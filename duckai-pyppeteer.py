import asyncio
import json
import base64
import re
import os
import logging
import argparse
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime
from enum import Enum

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyppeteer import launch
from pyppeteer_stealth import stealth
from dotenv import load_dotenv


class LogLevel(Enum):
    SILENT = "silent"
    ERROR = "error"
    INFO = "info"
    DEBUG = "debug"

load_dotenv()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()
if LOG_LEVEL == "SILENT":
    logging.basicConfig(level=logging.CRITICAL + 1)
elif LOG_LEVEL == "ERROR":
    logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
elif LOG_LEVEL == "INFO":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your_free_gemini_key")
MODEL_NAME = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
URL_PROXY = os.environ.get("URL_PROXY")
if URL_PROXY:
    BASE_URL = URL_PROXY + BASE_URL

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

AVAILABLE_MODELS = [
    "claude-3-5-haiku-latest",
    "mistralai/Mistral-Small-24B-Instruct-2501", 
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "gpt-4o-mini",
    "gpt-5-mini"
]
DEFAULT_MODEL = "gpt-5-mini"


class DDGChat:
    def __init__(self, headless: bool = True):
        self.browser = None
        self.page = None
        self.headers = None
        self.headless = headless
        self.ready_event = asyncio.Event()
        self.session = None
        self.input_field = None
        self.api_headers_cache = {}

    async def start(self):
        try:
            logger.info("Starting browser...")
            self.session = aiohttp.ClientSession()
            
            self.browser = await launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",
                    "--window-size=1920,1080",
                    "--disable-blink-features=AutomationControlled",
                ],
                ignoreHTTPSErrors=True,
                userDataDir='./user_data',
                autoClose=False,
            )
            
            self.page = await self.browser.newPage()
            await self._configure_browser_stealth()
            
            await self.page.goto("https://duck.ai/", waitUntil='networkidle0', timeout=30000)
            await self._set_local_storage_preferences()
            
            await self.page.goto(
                "https://duck.ai/chat?q=test&duckai=1",
                waitUntil='networkidle0',
                timeout=30000
            )
            
            self.input_field = await self._wait_for_input(10000)
            if not self.input_field:
                raise Exception("Input field not found")
            
            # Инициализируем API сессию в браузере
            await self._initialize_api_session()
                
            logger.info("Input field found! Bot is ready!")
            self.ready_event.set()
            
        except Exception as e:
            logger.error("Error during startup: %s", e)
            await self._cleanup()
            raise

    async def _configure_browser_stealth(self):
        await self.page.setUserAgent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        await self.page.setViewport({'width': VIEW_W, 'height': VIEW_H})
        await stealth(self.page)
        
        await self.page.evaluateOnNewDocument("""
            () => {
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
            }
        """)
        
        # Захватываем заголовки API запросов
        self.page.on('request', lambda req: asyncio.ensure_future(self._capture_api_headers(req)))

    async def _capture_api_headers(self, request):
        try:
            if "duck.ai/duckchat/v1/chat" in request.url:
                headers = request.headers
                # Сохраняем все заголовки, а не только некоторые
                self.api_headers_cache = dict(headers)
                logger.debug("API headers captured: %s", list(headers.keys()))
                
                # Также получаем VQD hash
                vqd_hash = headers.get('x-vqd-hash-1') or headers.get('x-vqd-hash')
                if vqd_hash:
                    logger.debug("VQD hash captured: %s...", vqd_hash[:20])
        except Exception as e:
            logger.debug("Error capturing headers: %s", e)

    async def _set_local_storage_preferences(self):
        await self.page.evaluate("""() => {
            try {
                localStorage.setItem('duckaiHasAgreedToTerms', 'true');
                localStorage.setItem('preferredDuckaiModel', '"203"');
                localStorage.setItem('isRecentChatsOn', '"1"');
            } catch (e) {}
        }""")

    async def _initialize_api_session(self):
        """Выполняет тестовый запрос для захвата заголовков"""
        try:
            logger.info("Initializing API session...")
            
            # Выполняем небольшой запрос для получения заголовков
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
                            messages: [{role: 'user', content: 'hello'}],
                            canUseTools: true,
                            canUseApproxLocation: false,
                            metadata: {toolChoice: {WebSearch: false}}
                        })
                    });
                    // Не читаем ответ, просто отправляем запрос для захвата заголовков
                } catch (e) {
                    console.log('Test API call failed:', e);
                }
            }""")
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.warning("Failed to initialize API session: %s", e)

    async def _take_captcha_screenshot(self):
        try:
            await self.page.screenshot({'path': 'captcha_full.png', 'fullPage': True})
            logger.info("Captcha screenshot saved")
            return True
        except Exception as e:
            logger.error("Error taking captcha screenshot: %s", e)
            return False

    async def solve_captcha_with_gemini(self, image_path: str):
        url = f"{BASE_URL}/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
        
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        prompt = "where is the duck/duck on the captcha, return a JSON object with a key named 'matrix' containing a 3*3 grid"
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": img_base64}}]}],
            "generationConfig": {"responseModalities": ["TEXT"]}
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            logger.error("No candidates in Gemini response")
                            return None
                        
                        text_response = candidates[0]["content"]["parts"][0]["text"].strip()
                        logger.info(f"Raw Gemini response:\n{text_response}")

                        matrix = self._parse_gemini_response(text_response)
                        
                        if matrix:
                            logger.info(f"Parsed matrix: {matrix}")
                            return matrix
                        else:
                            logger.error(f"Could not parse matrix from response: {text_response}")
                            return None
                            
                    except Exception as e:
                        logger.error(f"Error parsing Gemini response: {e}\nRaw response: {response_text}")
                        return None
                else:
                    logger.error(f"Gemini HTTP error {response.status}: {response_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def _parse_gemini_response(self, text_response: str):
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
        if self._is_valid_matrix(parsed_data):
            return parsed_data
        
        if isinstance(parsed_data, dict):
            for key in ['captcha_solution', 'matrix', 'answer', 'solution', 'grid', 'duck_captcha']:
                if key in parsed_data:
                    matrix = parsed_data[key]
                    if self._is_valid_matrix(matrix):
                        return matrix
        
        if isinstance(parsed_data, list) and len(parsed_data) == 1:
            if self._is_valid_matrix(parsed_data[0]):
                return parsed_data[0]
        
        return None

    def _is_valid_matrix(self, matrix):
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
        try:
            if not self._is_valid_matrix(matrix):
                logger.error(f"Invalid matrix format: {matrix}")
                return False

            for i in range(3):
                for j in range(3):
                    try:
                        if int(matrix[i][j]) == 1:
                            cx = GRID_START_X + j * GRID_STEP_X + GRID_CENTER_OFFSET
                            cy = GRID_START_Y + i * GRID_STEP_Y + GRID_CENTER_OFFSET
                            logger.info("Clicking tile (%d,%d) at (%d,%d)", i, j, cx, cy)
                            await self.page.mouse.click(cx, cy)
                            await asyncio.sleep(CLICK_DELAY)
                    except Exception as e:
                        logger.error(f"Error processing cell [{i}][{j}]: {e}")

            await self.page.mouse.click(SUBMIT_X, SUBMIT_Y + 50)
            await asyncio.sleep(SUBMIT_DELAY)
            logger.info("Captcha submitted successfully")
            return True
            
        except Exception as e:
            logger.error("Error clicking captcha: %s", e)
            return False

    async def _cleanup(self):
        try:
            if self.page:
                await self.page.close()
        except:
            pass
        try:
            if self.browser:
                await self.browser.close()
        except:
            pass
        self.ready_event.clear()

    async def stop(self):
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
        except Exception as e:
            logger.error("Error during shutdown: %s", e)
        finally:
            if self.session:
                await self.session.close()
            self.page = None
            self.browser = None

    async def _wait_for_input(self, timeout=15000):
        selectors = [
            'textarea[name="user-prompt"]',
            'div[contenteditable="true"]',
            '[data-testid="chat-input"]',
            '.chat-input',
            'input[type="text"]'
        ]
        
        for selector in selectors:
            try:
                element = await self.page.waitForSelector(selector, timeout=5000)
                if element:
                    return element
            except:
                continue
        return None

    async def _refresh_headers(self):
        """Обновляет API заголовки, выполняя новый запрос через браузер"""
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
            logger.error("Error refreshing headers: %s", e)

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

    async def _send_via_browser(self, messages: List[Dict[str, str]], model: str = DEFAULT_MODEL):
        """Отправляет запрос через браузерную сессию"""
        try:
            payload = {
                "model": model,
                "metadata": {"toolChoice": {"WebSearch": False}},
                "messages": messages,
                "canUseTools": True,
                "canUseApproxLocation": False,
            }
            
            logger.debug("Sending request via browser: %s", payload)
            
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
                        return { error: `HTTP error ${response.status}` };
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
                                } catch (e) {}
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
            logger.error("Error in browser request: %s", e)
            raise

    async def ask(self, messages: List[Dict[str, str]], model: str = DEFAULT_MODEL):
        if model not in AVAILABLE_MODELS:
            logger.warning("Model %s not available, using default", model)
            model = DEFAULT_MODEL
            
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
        
        return answer


class Query(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = DEFAULT_MODEL


bot = DDGChat(headless=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await bot.start()
        yield
    except Exception as e:
        logger.error("Startup failed: %s", e)
        raise
    finally:
        await bot.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/ask")
async def ask_question(query: Query):
    if not bot.ready_event.is_set():
        raise HTTPException(status_code=503, detail="Bot not ready")
    
    try:
        actual_model = query.model if query.model in AVAILABLE_MODELS else DEFAULT_MODEL
        answer = await bot.ask(query.messages, actual_model)
        return {"answer": answer, "model": actual_model}
    except Exception as e:
        if "CAPTCHA" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

class CLIChat:
    def __init__(self, conversation_file: str = "conversation.json"):
        self.conversation_file = conversation_file
        self.conversation = self._load_conversation()
        self.chat_bot = DDGChat(headless=True)
        self.current_model = DEFAULT_MODEL
    
    def _load_conversation(self):
        try:
            if os.path.exists(self.conversation_file):
                with open(self.conversation_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error("Error loading conversation: %s", e)
        return []
    
    def _save_conversation(self):
        try:
            with open(self.conversation_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Error saving conversation: %s", e)
    
    def _add_message(self, role: str, content: str):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation.append(message)
        self._save_conversation()
    
    async def start(self):
        try:
            print("Initializing DuckDuckGo chat...")
            await self.chat_bot.start()
            
            print("\n=== DuckDuckGo CLI Chat ===")
            print("Type 'quit' to exit, 'clear' to reset conversation")
            print("Type 'model' to change AI model")
            print(f"Current model: {self.current_model}")
            print("=" * 30)
            
            while True:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'clear':
                    self.conversation = []
                    self._save_conversation()
                    print("Conversation cleared")
                    continue
                elif user_input.lower() == 'model':
                    print("\nAvailable models:")
                    for i, model_name in enumerate(AVAILABLE_MODELS, 1):
                        print(f"{i}. {model_name}")
                    
                    try:
                        choice = input(f"\nSelect model by number (1-{len(AVAILABLE_MODELS)}): ").strip()
                        if choice:
                            model_index = int(choice) - 1
                            if 0 <= model_index < len(AVAILABLE_MODELS):
                                self.current_model = AVAILABLE_MODELS[model_index]
                                print(f"Selected model: {self.current_model}")
                            else:
                                print("Invalid model number")
                    except ValueError:
                        print("Please enter a valid number")
                    continue
                
                if not user_input:
                    continue
                
                self._add_message("user", user_input)
                
                print("AI: ", end="", flush=True)
                try:
                    response = await self.chat_bot.ask(self.conversation, self.current_model)
                    print(response)
                    self._add_message("assistant", response)
                except Exception as e:
                    print(f"Error: {e}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
        except Exception as e:
            logger.error("Chat error: %s", e)
        finally:
            await self.chat_bot.stop()

def main():
    parser = argparse.ArgumentParser(description="DuckDuckGo Chat Interface")
    parser.add_argument('--mode', choices=['server', 'cli'], default='server',
                       help='Run mode: server (FastAPI) or cli (interactive)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (server mode only)')
    parser.add_argument('--port', type=int, default=8000, help='Server port (server mode only)')
    parser.add_argument('--conversation', default='conversation.json',
                       help='Conversation file (cli mode only)')
    parser.add_argument('--log-level', choices=['silent', 'error', 'info', 'debug'],
                       default='info', help='Logging level')
    
    args = parser.parse_args()
    
    os.environ["LOG_LEVEL"] = args.log_level
    
    if args.mode == 'cli':
        cli_chat = CLIChat(args.conversation)
        asyncio.run(cli_chat.start())
    else:
        import uvicorn
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()