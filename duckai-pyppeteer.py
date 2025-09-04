import asyncio
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from pyppeteer import launch
from pyppeteer_stealth import stealth
from contextlib import asynccontextmanager
import logging
import aiohttp
import base64
import re
import psutil
import sys, os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = "your free gemini key"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
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
MAX_CAPTCHA_ATTEMPTS = 2

CLICK_DELAY = 0.25
SUBMIT_DELAY = 1.5

class DDGChat:
    def __init__(self, headless: bool = True):
        self.browser = None
        self.page = None
        self.headers = None
        self.messages = []
        self.headless = headless
        self.ready_event = asyncio.Event()
        self._captcha_attempts = 0
        self.session: aiohttp.ClientSession | None = None
        self.input_field = None

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
            self.page.on('request', lambda req: asyncio.ensure_future(self._log_request(req)))
            logger.info("Navigating to DuckDuckGo...")
            await self.page.goto("https://duckduckgo.com", waitUntil='networkidle0', timeout=30000)
            await self.page.evaluate("""() => {
                try {
                    localStorage.setItem('duckaiHasAgreedToTerms', 'true');
                    localStorage.setItem('preferredDuckaiModel', '"203"');
                    localStorage.setItem('isRecentChatsOn', '"1"');
                } catch (e) {}
            }""")
            logger.info("Navigating to chat page...")
            await self.page.goto(
                "https://duckduckgo.com/?q=test&ia=chat&duckai=1",
                waitUntil='networkidle0',
                timeout=30000
            )
            self.input_field = await self._wait_for_input(10000)
            if self.input_field:
                logger.info("Input field found! Bot is ready!")
                self.ready_event.set()
            else:
                raise Exception("Input field not found")
        except Exception as e:
            logger.error(f"Error during startup: {e}")
            await self._cleanup()
            raise

    async def _take_captcha_screenshot(self):
        try:
            await self.page.screenshot({'path': 'captcha_full.png', 'fullPage': True})
            logger.info("Captcha screenshot saved (captcha_full.png)")
            return True
        except Exception as e:
            logger.error(f"Error taking captcha screenshot: {e}")
            return False

    async def solve_captcha_with_gemini(self, image_path: str):
        url = f"{BASE_URL}/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        prompt = "where is the duck/duck on the captcha, give the answer as a 3*3 matrix in json"
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": img_base64}}]}],
            "generationConfig": {"responseModalities": ["TEXT"]}
        }
        headers = {"Content-Type": "application/json"}
        async with self.session.post(url, json=payload, headers=headers) as response:
            response_text = await response.text()
            if response.status == 200:
                try:
                    data = await response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return None
                    text_response = candidates[0]["content"]["parts"][0]["text"].strip()

                    clean = re.sub(
                        r"^```json\s*|^```\s*|\s*```$",
                        "",
                        text_response.strip(),
                        flags=re.IGNORECASE
                    )

                    logger.info(f"Cleaned Gemini response:\n{clean}")

                    try:
                        parsed = json.loads(clean)
                    except json.JSONDecodeError:
                        m = re.search(r"\[\s*\[.*?\]\s*,\s*\[.*?\]\s*,\s*\[.*?\]\s*\]", clean, re.DOTALL)
                        if not m:
                            raise
                        parsed = json.loads(m.group(0))

                    if isinstance(parsed, dict):
                        if "matrix" in parsed:
                            matrix = parsed["matrix"]
                        elif "answer" in parsed:
                            matrix = parsed["answer"]
                        else:
                            raise ValueError(f"Unexpected Gemini response format: {parsed}")
                    else:
                        matrix = parsed

                    matrix = [[int(x) for x in row] for row in matrix]

                    logger.info(f"Gemini matrix: {matrix}")
                    return matrix
                except Exception as e:
                    logger.error(f"Error parsing Gemini response: {e}\nRaw response: {response_text}")
                    return None
            else:
                logger.error(f"Gemini HTTP error {response.status}: {response_text}")
                return None


    async def click_captcha(self, matrix):
        try:
            if not isinstance(matrix, list) or len(matrix) != 3 or not all(isinstance(row, list) and len(row) == 3 for row in matrix):
                logger.error(f"Invalid matrix format: {matrix}")
                return False

            for i in range(3):
                for j in range(3):
                    try:
                        if int(matrix[i][j]) == 1:
                            cx = GRID_START_X + j * GRID_STEP_X + GRID_CENTER_OFFSET
                            cy = GRID_START_Y + i * GRID_STEP_Y + GRID_CENTER_OFFSET
                            logger.info(f"Clicking tile ({i},{j}) at ({cx},{cy})")
                            await self.page.mouse.click(cx, cy)
                            await asyncio.sleep(CLICK_DELAY)
                    except Exception as e:
                        logger.error(f"Error processing cell [{i}][{j}]: {e}")

            await self.page.mouse.click(SUBMIT_X, SUBMIT_Y + 50)
            await asyncio.sleep(SUBMIT_DELAY)
            logger.info("Captcha submitted")
            return True
        except Exception as e:
            logger.error(f"Error clicking captcha: {e}")
            return False


    async def _cleanup(self):
        try:
            if self.page:
                await self.page.close()
                self.page = None
        except:
            pass
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
        except:
            pass
        self.ready_event.clear()

    async def _log_request(self, request):
        try:
            if "duckduckgo.com/duckchat/v1/chat" in request.url:
                self.headers = request.headers
                logger.info("Headers captured from API request")
        except:
            pass

    async def stop(self):
        try:
            if self.page:
                try:
                    await self.page.close()
                except:
                    pass
            if self.browser:
                try:
                    proc = self.browser.process
                    await self.browser.close()
                    if proc and psutil.pid_exists(proc.pid):
                        proc.kill()
                except:
                    pass
        finally:
            if self.session:
                await self.session.close()
                self.session = None
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
        try:
            buttons = await self.page.querySelectorAll('button[type="button"]')
            clicked = False
            for b in buttons:
                try:
                    text = await self.page.evaluate('(el) => el.innerText', b)
                    if text and text.strip().lower() in ["новый чат", "запустить чат", "new chat", "start chat"]:
                        logger.info(f"Found chat button: {text.strip()}, clicking...")
                        await b.click()
                        await asyncio.sleep(0.5)
                        clicked = True
                        break
                except Exception as e:
                    logger.debug(f"Error reading button text: {e}")
                    continue
            if not clicked:
                logger.debug("No 'New Chat' button found")
        except Exception as e:
            logger.debug(f"No 'New Chat' button found: {e}")

        try:
            el = await self._wait_for_input()
            if not el:
                raise Exception("Input field not found for header refresh")
            await el.click()
            await el.type(" ")
            await self.page.keyboard.press("Enter")
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error refreshing headers: {e}")
            raise



    def _filtered_headers(self):
        if not self.headers:
            return {}
        return {
            k: v for k, v in self.headers.items()
            if k.lower() in [
                "accept", "content-type", "origin", "referer",
                "user-agent", "x-fe-signals", "x-fe-version",
                "x-vqd-hash-1", "sec-ch-ua", "sec-ch-ua-mobile",
                "sec-ch-ua-platform",
            ]
        }

    async def _send(self, messages: List[Dict[str, str]]):
        payload = {
            "model": "gpt-5-mini",
            "metadata": {"toolChoice": {"WebSearch": False}},
            "messages": messages,
            "canUseTools": True,
            "canUseApproxLocation": False,
        }
        headers = self._filtered_headers()
        full_answer = []
        try:
            async with self.session.post(
                "https://duckduckgo.com/duckchat/v1/chat",
                headers=headers,
                json=payload,
                timeout=30
            ) as r:
                if r.status == 418:
                    return "CAPTCHA_REQUIRED"
                if r.status != 200:
                    text = await r.text()
                    return f"HTTP error {r.status}: {text}"
                async for line in r.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_line = line[len("data: "):]
                    if data_line == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_line)
                        if "message" in obj:
                            full_answer.append(obj["message"])
                        elif obj.get("action") == "error" and obj.get("type") == "ERR_CHALLENGE":
                            return "CAPTCHA_REQUIRED"
                    except:
                        continue
        except Exception as e:
            return f"Request error: {str(e)}"
        return "".join(full_answer)

    async def ask(self, messages: List[Dict[str, str]]):
        await self._refresh_headers()
        answer = await self._send(messages)
        if answer == "CAPTCHA_REQUIRED":
            if self._captcha_attempts >= MAX_CAPTCHA_ATTEMPTS:
                raise Exception("Captcha could not be solved after max attempts")
            self._captcha_attempts += 1
            await self._take_captcha_screenshot()
            matrix = await self.solve_captcha_with_gemini("captcha_full.png")
            if matrix:
                ok = await self.click_captcha(matrix)
                if not ok:
                    raise Exception("Failed to click captcha tiles")
                await asyncio.sleep(1.0)
                return await self.ask(messages)
            else:
                raise Exception("Failed to parse matrix from Gemini")
        self._captcha_attempts = 0
        try:
            await self.page.evaluate("localStorage.removeItem('savedAIChats')")
        except:
            pass
        return answer

bot = DDGChat(headless=True)

class Query(BaseModel):
    messages: List[Dict[str, str]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await bot.start()
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        yield
    finally:
        await bot.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/ask")
async def ask_question(query: Query):
    if not bot.ready_event.is_set():
        raise HTTPException(status_code=503, detail="Bot not ready yet")
    try:
        answer = await bot.ask(query.messages)
        return {"answer": answer}
    except Exception as e:
        if "CAPTCHA" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ready" if bot.ready_event.is_set() else "not ready"}

def handle_exit(*args):
    logger.info("Shutting down gracefully...")
    loop = asyncio.get_event_loop()
    loop.create_task(bot.stop())
    sys.exit(0)

if __name__ == "__main__":
    import uvicorn, sys, asyncio

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        loop="asyncio",
    )