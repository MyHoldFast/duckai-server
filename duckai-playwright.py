import asyncio
import json
import base64
import re
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from playwright.async_api import async_playwright
from contextlib import asynccontextmanager
import aiohttp

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DDGChat:
    def __init__(self, headless: bool = True):
        self.browser = None
        self.context = None
        self.page = None
        self.api_headers_cache = {}
        self.messages = []
        self.headless = headless
        self.pw = None
        self.ready_event = asyncio.Event()
        self.session = None

    async def start(self):
        """Запускает браузер и инициализирует сессию"""
        self.session = aiohttp.ClientSession()
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--window-size=1920,1080",
                "--start-maximized",
                "--disable-extensions",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-software-rasterizer",
                "--mute-audio",
                "--hide-scrollbars",
            ],
        )
        
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
        
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="ru-RU",
            user_agent=user_agent,
            ignore_https_errors=True
        )
        
        self.page = await self.context.new_page()
        
        # Скрываем автоматизацию
        await self.page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'languages', { get: () => ['ru-RU','ru'] });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(Notification, 'permission', { get: () => 'granted' });
        """)

        # Захватываем заголовки API запросов
        async def capture_api_headers(request):
            if "duck.ai/duckchat/v1/chat" in request.url:
                try:
                    headers = request.headers
                    self.api_headers_cache = dict(headers)
                    logger.debug(f"Captured API headers: {list(headers.keys())}")
                except Exception as e:
                    logger.debug(f"Error capturing headers: {e}")

        self.page.on("request", capture_api_headers)
        
        # Загружаем страницу и настраиваем
        await self.page.goto("https://duck.ai", wait_until="domcontentloaded")
        
        # Настраиваем localStorage
        await self.page.evaluate("""() => {
            localStorage.setItem('preferredDuckaiModel', '"203"');
            localStorage.setItem('duckaiHasAgreedToTerms', 'true');
            localStorage.setItem('isRecentChatsOn', '"1"');
        }""")
        
        await self.page.goto(
            "https://duck.ai/chat?q=test&duckai=1",
            wait_until="domcontentloaded",
        )
        
        # Инициализируем API сессию
        await self._initialize_api_session()
        
        await self._wait_for_input()
        self.ready_event.set()
        logger.info("Browser initialized and ready")

    async def _initialize_api_session(self):
        """Выполняет тестовый запрос для захвата заголовков"""
        try:
            logger.info("Initializing API session...")
            
            # Выполняем тестовый запрос через браузер
            await self.page.evaluate("""async () => {
                try {
                    const response = await fetch('https://duck.ai/duckchat/v1/chat', {
                        method: 'POST',
                        headers: {
                            'accept': 'text/event-stream',
                            'accept-language': 'en-US,en;q=0.9',
                            'cache-control': 'no-cache',
                            'content-type': 'application/json',
                            'pragma': 'no-cache',
                            'priority': 'u=1, i',
                            'sec-fetch-dest': 'empty',
                            'sec-fetch-mode': 'cors',
                            'sec-fetch-site': 'same-origin',
                            'x-fe-version': 'serp_20250401_100419_ET-19d438eb199b2bf7c300',
                        },
                        body: JSON.stringify({
                            model: 'gpt-4o-mini',
                            messages: [{role: 'user', content: 'hello'}],
                            canUseTools: true,
                            canUseApproxLocation: false,
                            metadata: {toolChoice: {WebSearch: false}}
                        })
                    });
                } catch (e) {
                    console.log('Test API call failed:', e);
                }
            }""")
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.warning(f"Failed to initialize API session: {e}")

    async def stop(self):
        """Останавливает браузер и очищает ресурсы"""
        try:
            if self.browser:
                await self.browser.close()
            if self.pw:
                await self.pw.stop()
            if self.session:
                await self.session.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self.ready_event.clear()

    async def _wait_for_input(self):
        """Ожидает появление поля ввода"""
        selectors = [
            'textarea[name="user-prompt"]',
            'div[contenteditable="true"]',
            '[data-testid="chat-input"]',
            '.chat-input',
            'input[type="text"]'
        ]
        
        for selector in selectors:
            try:
                element = await self.page.wait_for_selector(selector, timeout=3000)
                if element:
                    logger.debug(f"Found input field with selector: {selector}")
                    return element
            except:
                continue
        
        logger.warning("No input field found")
        return None

    async def _refresh_headers(self):
        """Обновляет API заголовки через браузер"""
        try:
            logger.debug("Refreshing API headers...")
            
            # Выполняем запрос через браузер для получения свежих заголовков
            await self.page.evaluate("""async () => {
                try {
                    const response = await fetch('https://duck.ai/duckchat/v1/chat', {
                        method: 'POST',
                        headers: {
                            'accept': 'text/event-stream',
                            'content-type': 'application/json',
                        },
                        body: JSON.stringify({
                            model: 'gpt-4o-mini',
                            messages: [{role: 'user', content: 'test'}],
                            canUseTools: true,
                            canUseApproxLocation: false,
                            metadata: {toolChoice: {WebSearch: false}}
                        })
                    });
                } catch (e) {
                    console.error('Failed to refresh headers:', e);
                }
            }""")
            
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error refreshing headers: {e}")

    def _get_api_headers(self):
        """Возвращает заголовки для API запроса"""
        if not self.api_headers_cache:
            logger.warning("No API headers cached, using defaults")
            return {
                "accept": "text/event-stream",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "no-cache",
                "content-type": "application/json",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "x-fe-version": "serp_20250401_100419_ET-19d438eb199b2bf7c300",
            }
        
        # Фильтруем заголовки, оставляя только нужные
        filtered_headers = {}
        important_headers = [
            'accept', 'accept-language', 'cache-control', 'content-type',
            'pragma', 'priority', 'sec-fetch-dest', 'sec-fetch-mode',
            'sec-fetch-site', 'x-fe-version', 'x-vqd-hash-1', 'x-vqd-hash',
            'user-agent', 'referer', 'origin'
        ]
        
        for header in important_headers:
            for key, value in self.api_headers_cache.items():
                if key.lower() == header.lower():
                    filtered_headers[key] = value
                    break
        
        # Добавляем обязательные заголовки, если их нет
        if 'accept' not in filtered_headers:
            filtered_headers['accept'] = 'text/event-stream'
        if 'content-type' not in filtered_headers:
            filtered_headers['content-type'] = 'application/json'
        
        return filtered_headers

    async def _send_via_browser(self, messages: List[Dict[str, str]], model: str = "gpt-5-mini"):
        """Отправляет запрос через браузерную сессию"""
        try:
            payload = {
                "model": model,
                "metadata": {
                    "toolChoice": {
                        "NewsSearch": False,
                        "VideosSearch": False,
                        "LocalSearch": False,
                        "WeatherForecast": False,
                        "WebSearch": False
                    }
                },
                "messages": messages,
                "canUseTools": True,
                "canUseApproxLocation": False,
            }
            
            logger.debug(f"Sending request via browser with model: {model}")
            
            # Выполняем запрос через браузер
            result = await self.page.evaluate("""async (payload) => {
                try {
                    const response = await fetch('https://duck.ai/duckchat/v1/chat', {
                        method: 'POST',
                        headers: {
                            'accept': 'text/event-stream',
                            'accept-language': 'en-US,en;q=0.9',
                            'cache-control': 'no-cache',
                            'content-type': 'application/json',
                            'pragma': 'no-cache',
                            'priority': 'u=1, i',
                            'sec-fetch-dest': 'empty',
                            'sec-fetch-mode': 'cors',
                            'sec-fetch-site': 'same-origin',
                            'x-fe-version': 'serp_20250401_100419_ET-19d438eb199b2bf7c300',
                        },
                        body: JSON.stringify(payload)
                    });
                    
                    if (response.status === 418) {
                        return { error: 'CAPTCHA_REQUIRED' };
                    }
                    
                    if (!response.ok) {
                        return { error: `HTTP error ${response.status}: ${response.statusText}` };
                    }
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let fullAnswer = '';
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const dataLine = line.slice(6);
                                if (dataLine === '[DONE]') break;
                                
                                try {
                                    const obj = JSON.parse(dataLine);
                                    if (obj.message) {
                                        fullAnswer += obj.message;
                                    } else if (obj.action === 'error' && obj.type === 'ERR_CHALLENGE') {
                                        return { error: 'CAPTCHA_REQUIRED' };
                                    }
                                } catch (e) {
                                    // Пропускаем невалидный JSON
                                }
                            }
                        }
                    }
                    
                    return { answer: fullAnswer };
                    
                } catch (error) {
                    return { error: error.toString() };
                }
            }""", payload)
            
            if 'error' in result:
                if result['error'] == 'CAPTCHA_REQUIRED':
                    return 'CAPTCHA_REQUIRED'
                else:
                    raise Exception(f"Browser request error: {result['error']}")
            
            return result.get('answer', '')
            
        except Exception as e:
            logger.error(f"Error in browser request: {e}")
            raise

    async def _take_captcha_screenshot(self):
        """Делает скриншот капчи"""
        try:
            await self.page.screenshot(path='captcha_full.png', full_page=True)
            logger.info("Captcha screenshot saved")
            return True
        except Exception as e:
            logger.error(f"Error taking captcha screenshot: {e}")
            return False

    async def solve_captcha_with_gemini(self, image_path: str):
        """Решает капчу через Gemini API"""
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your_free_gemini_key")
        MODEL_NAME = "gemini-2.5-flash"
        BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
        URL_PROXY = os.environ.get("URL_PROXY")
        if URL_PROXY:
            BASE_URL = URL_PROXY + BASE_URL

        url = f"{BASE_URL}/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
        
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        prompt = "where is the duck/duck on the captcha, give the answer as a 3*3 matrix in json"
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": img_base64}}]}],
            "generationConfig": {"responseModalities": ["TEXT"]}
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        logger.error("No candidates in Gemini response")
                        return None
                    
                    text_response = candidates[0]["content"]["parts"][0]["text"].strip()
                    logger.info(f"Raw Gemini response: {text_response}")
                    
                    matrix = self._parse_gemini_response(text_response)
                    
                    if matrix:
                        logger.info(f"Parsed matrix: {matrix}")
                        return matrix
                    else:
                        logger.error("Could not parse matrix from response")
                        return None
                else:
                    logger.error(f"Gemini HTTP error {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return None

    def _parse_gemini_response(self, text_response: str):
        """Парсит ответ от Gemini для извлечения матрицы"""
        clean_text = re.sub(
            r"^```json\s*|^```\s*|\s*```$",
            "",
            text_response.strip(),
            flags=re.IGNORECASE
        )
        
        try:
            parsed = json.loads(clean_text)
            return self._extract_matrix_from_json(parsed)
        except json.JSONDecodeError:
            pass
        
        matrix_pattern = r'\[\[[^\]]*\],\s*\[[^\]]*\],\s*\[[^\]]*\]\]'
        match = re.search(matrix_pattern, clean_text)
        if match:
            try:
                matrix_str = match.group(0)
                parsed = json.loads(matrix_str)
                if self._is_valid_matrix(parsed):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        rows = re.findall(r'\[[^\]]*\]', clean_text)
        if len(rows) >= 3:
            matrix = []
            for row in rows[:3]:
                try:
                    row_data = json.loads(row)
                    if len(row_data) == 3:
                        matrix.append(row_data)
                except json.JSONDecodeError:
                    numbers = re.findall(r'[01]', row)
                    if len(numbers) >= 3:
                        matrix.append([int(num) for num in numbers[:3]])
            
            if len(matrix) == 3 and all(len(row) == 3 for row in matrix):
                return matrix
        
        return None

    def _extract_matrix_from_json(self, parsed_data):
        """Извлекает матрицу из JSON ответа"""
        if self._is_valid_matrix(parsed_data):
            return parsed_data
        
        if isinstance(parsed_data, dict):
            for key in ['captcha_solution', 'matrix', 'answer', 'solution', 'grid']:
                if key in parsed_data:
                    matrix = parsed_data[key]
                    if self._is_valid_matrix(matrix):
                        return matrix
        
        if isinstance(parsed_data, list) and len(parsed_data) == 1:
            if self._is_valid_matrix(parsed_data[0]):
                return parsed_data[0]
        
        return None

    def _is_valid_matrix(self, matrix):
        """Проверяет валидность матрицы"""
        if not isinstance(matrix, list) or len(matrix) != 3:
            return False
        for row in matrix:
            if not isinstance(row, list) or len(row) != 3:
                return False
            for cell in row:
                if not isinstance(cell, (int, float)) or cell not in [0, 1]:
                    return False
        return True

    async def click_captcha(self, matrix):
        """Кликает по капче согласно матрице"""
        VIEW_W, VIEW_H = 1920, 1080
        GRID_START_X = 780
        GRID_START_Y = 380
        GRID_STEP_X = 114
        GRID_STEP_Y = 114
        GRID_CENTER_OFFSET = 57
        SUBMIT_X = 960
        SUBMIT_Y = 735
        CLICK_DELAY = 0.25
        SUBMIT_DELAY = 1.5

        try:
            if not self._is_valid_matrix(matrix):
                logger.error(f"Invalid matrix format: {matrix}")
                return False
            
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    if int(value) == 1:
                        cx = GRID_START_X + j * GRID_STEP_X + GRID_CENTER_OFFSET
                        cy = GRID_START_Y + i * GRID_STEP_Y + GRID_CENTER_OFFSET
                        logger.info(f"Clicking tile ({i},{j}) at ({cx},{cy})")
                        await self.page.mouse.click(cx, cy)
                        await asyncio.sleep(CLICK_DELAY)
            
            await self.page.mouse.click(SUBMIT_X, SUBMIT_Y + 50)
            await asyncio.sleep(SUBMIT_DELAY)
            logger.info("Captcha submitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clicking captcha: {e}")
            return False

    async def ask(self, messages: List[Dict[str, str]], model: str = "gpt-5-mini"):
        """Основной метод для отправки запроса"""
        AVAILABLE_MODELS = [
            "claude-3-5-haiku-latest",
            "mistralai/Mistral-Small-24B-Instruct-2501", 
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "gpt-4o-mini",
            "gpt-5-mini"
        ]

        if model not in AVAILABLE_MODELS:
            logger.warning(f"Model {model} not available, using default")
            model = "gpt-5-mini"
            
        answer = await self._send_via_browser(messages, model)
        
        if answer == "CAPTCHA_REQUIRED":
            logger.info("Captcha detected, attempting to solve...")
            
            if await self._take_captcha_screenshot():
                matrix = await self.solve_captcha_with_gemini("captcha_full.png")
                if matrix:
                    ok = await self.click_captcha(matrix)
                    if ok:
                        logger.info("Captcha solved successfully, retrying request...")
                        await asyncio.sleep(2.0)
                        return await self.ask(messages, model)
                    else:
                        raise Exception("Failed to click captcha tiles")
                else:
                    raise Exception("Failed to parse matrix from Gemini")
            else:
                raise Exception("Failed to take captcha screenshot")
        
        # Очищаем историю чатов
        try:
            await self.page.evaluate("localStorage.removeItem('savedAIChats')")
        except:
            pass
        
        return answer


bot = DDGChat(headless=False)


class Query(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = "gpt-5-mini"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager для FastAPI"""
    try:
        await bot.start()
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        await bot.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/ask")
async def ask_question(query: Query):
    """Endpoint для запросов к DuckDuckGo AI"""
    if not bot.ready_event.is_set():
        raise HTTPException(status_code=503, detail="Bot not ready yet")
    
    try:
        AVAILABLE_MODELS = [
            "claude-3-5-haiku-latest",
            "mistralai/Mistral-Small-24B-Instruct-2501", 
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "gpt-4o-mini",
            "gpt-5-mini"
        ]
        
        actual_model = query.model if query.model in AVAILABLE_MODELS else "gpt-5-mini"
        answer = await bot.ask(query.messages, actual_model)
        
        return {"answer": answer, "model": actual_model}
    except Exception as e:
        if "CAPTCHA" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import sys

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")