import asyncio
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from playwright.async_api import async_playwright
from contextlib import asynccontextmanager
import base64
import re
import os
import aiohttp


class DDGChat:
    def __init__(self, headless: bool = True):
        self.browser = None
        self.context = None
        self.page = None
        self.headers = None
        self.messages = []
        self.headless = headless
        self.pw = None
        self.ready_event = asyncio.Event()
        self._captcha_attempts = 0
        self.session = None

    async def start(self):
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
        )
        self.page = await self.context.new_page()
        await self.page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'languages', { get: () => ['ru-RU','ru'] });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(Notification, 'permission', { get: () => 'granted' });
        """)

        async def log_request(request):
            if "duckduckgo.com/duckchat/v1/chat" in request.url:
                self.headers = request.headers

        self.page.on("request", log_request)
        await self.page.goto("https://duckduckgo.com", wait_until="domcontentloaded")
        await self.page.evaluate(
            "localStorage.setItem('preferredDuckaiModel', '\"203\"')"
        )
        await self.page.goto(
            "https://duckduckgo.com/?q=DuckDuckGo+AI+Chat&ia=chat&duckai=1",
            wait_until="domcontentloaded",
        )
        await self._wait_for_input()
        self.ready_event.set()

    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()
        if self.session:
            await self.session.close()
        self.ready_event.clear()

    async def _wait_for_input(self):
        try:
            return await self.page.wait_for_selector(
                'textarea[name="user-prompt"]', timeout=6000
            )
        except:
            return await self.page.wait_for_selector(
                'div[contenteditable="true"]', timeout=6000
            )

    async def _refresh_headers(self):
        try:
            btn = self.page.locator(
                'button[type="button"]:has-text("Новый чат"), '
                'button[type="button"]:has-text("Запустить чат"), '
                'button[type="button"]:has-text("New Chat"), '
                'button[type="button"]:has-text("Start chat")'
            )
            await btn.first.wait_for(timeout=3000)
            await btn.first.click()
        except:
            pass
        el = await self._wait_for_input()
        await el.click()
        await el.type(" ")
        await self.page.keyboard.press("Enter")
        await self.page.wait_for_timeout(500)

    def _filtered_headers(self):
        return {
            k: v
            for k, v in self.headers.items()
            if k.lower()
            in [
                "accept",
                "content-type",
                "origin",
                "referer",
                "user-agent",
                "x-fe-signals",
                "x-fe-version",
                "x-vqd-hash-1",
                "sec-ch-ua",
                "sec-ch-ua-mobile",
                "sec-ch-ua-platform",
            ]
        }

    async def _take_captcha_screenshot(self):
        try:
            await self.page.screenshot(path='captcha_full.png', full_page=True)
            return True
        except:
            return False

    async def solve_captcha_with_gemini(self, image_path: str):
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your_free_gemini_key")
        MODEL_NAME = "gemini-2.5-flash-preview-05-20"
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
                        return None
                    
                    text_response = candidates[0]["content"]["parts"][0]["text"].strip()
                    
                    clean = re.sub(
                        r"^```json\s*|^```\s*|\s*```$",
                        "",
                        text_response.strip(),
                        flags=re.IGNORECASE
                    )

                    try:
                        parsed = json.loads(clean)
                    except json.JSONDecodeError:
                        m = re.search(r"\[\s*\[.*?\]\s*,\s*\[.*?\]\s*,\s*\[.*?\]\s*\]", clean, re.DOTALL)
                        if not m:
                            return None
                        parsed = json.loads(m.group(0))

                    if isinstance(parsed, dict):
                        if "matrix" in parsed:
                            matrix = parsed["matrix"]
                        elif "answer" in parsed:
                            matrix = parsed["answer"]
                        elif "response" in parsed:
                            matrix = parsed["response"]
                        else:
                            for key, value in parsed.items():
                                if isinstance(value, list) and len(value) == 3:
                                    matrix = value
                                    break
                            else:
                                return None
                    else:
                        matrix = parsed

                    matrix = [[int(x) for x in row] for row in matrix]
                    return matrix
                else:
                    return None
                    
        except:
            return None

    async def click_captcha(self, matrix):
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
            if not isinstance(matrix, list) or len(matrix) != 3:
                return False
            
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    if int(value) == 1:
                        cx = GRID_START_X + j * GRID_STEP_X + GRID_CENTER_OFFSET
                        cy = GRID_START_Y + i * GRID_STEP_Y + GRID_CENTER_OFFSET
                        await self.page.mouse.click(cx, cy)
                        await asyncio.sleep(CLICK_DELAY)
            
            await self.page.mouse.click(SUBMIT_X, SUBMIT_Y + 50)
            await asyncio.sleep(SUBMIT_DELAY)
            return True
            
        except:
            return False

    def _send(self, messages: List[Dict[str, str]], model: str):
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
        full_answer = []
        with requests.post(
            "https://duckduckgo.com/duckchat/v1/chat",
            headers=self._filtered_headers(),
            json=payload,
            stream=True,
        ) as r:
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_line = line[len("data: ") :]
                if data_line.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data_line)
                    if "message" in obj:
                        full_answer.append(obj["message"])
                    elif obj.get("action") == "error" and obj.get("type") == "ERR_CHALLENGE":
                        return "CAPTCHA_REQUIRED"
                except:
                    continue
        return "".join(full_answer)

    async def ask(self, messages: List[Dict[str, str]], model: str = "gpt-5-mini"):
        AVAILABLE_MODELS = [
            "claude-3-5-haiku-latest",
            "mistralai/Mistral-Small-24B-Instruct-2501", 
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "gpt-4o-mini",
            "gpt-5-mini"
        ]
        MAX_CAPTCHA_ATTEMPTS = 2

        if model not in AVAILABLE_MODELS:
            model = "gpt-5-mini"
            
        await self._refresh_headers()
        answer = self._send(messages, model)
        
        if answer == "CAPTCHA_REQUIRED":
            if self._captcha_attempts >= MAX_CAPTCHA_ATTEMPTS:
                raise Exception("Max captcha attempts exceeded")
                
            self._captcha_attempts += 1
            
            if await self._take_captcha_screenshot():
                matrix = await self.solve_captcha_with_gemini("captcha_full.png")
                if matrix and await self.click_captcha(matrix):
                    await asyncio.sleep(1.0)
                    return await self.ask(messages, model)
            
            raise Exception("Failed to solve captcha")
        try:
            await self.page.evaluate("localStorage.removeItem('savedAIChats')")
        except:
            pass        
        self._captcha_attempts = 0
        return answer


bot = DDGChat(headless=False)


class Query(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = "gpt-5-mini"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await bot.start()
    yield
    await bot.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/ask")
async def ask_question(query: Query):
    if not bot.ready_event.is_set():
        raise HTTPException(status_code=503, detail="Bot not ready yet")
    answer = await bot.ask(query.messages, query.model)
    return {"answer": answer, "model": query.model}


if __name__ == "__main__":
    import uvicorn, sys

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")