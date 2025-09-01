import asyncio
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from playwright.async_api import async_playwright
from contextlib import asynccontextmanager


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

    async def start(self):
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
        try:
            await self.page.wait_for_selector(
                "button:has-text('Запустить чат'), button:has-text('Start chat')",
                timeout=5000,
            )
            await self.page.locator(
                "button:has-text('Запустить чат'), button:has-text('Start chat')"
            ).click()
        except:
            pass
        await self._wait_for_input()
        self.ready_event.set() 

    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()
        self.ready_event.clear()

    async def _wait_for_input(self):
        try:
            return await self.page.wait_for_selector(
                'textarea[name="user-prompt"]', timeout=4000
            )
        except:
            return await self.page.wait_for_selector(
                'div[contenteditable="true"]', timeout=4000
            )

    async def _refresh_headers(self):
        try:
            btn = self.page.locator(
                'button[type="button"]:has-text("Новый чат"), '
                'button[type="button"]:has-text("Запустить чат"), '
                'button[type="button"]:has-text("New chat"), '
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

    def _send(self, messages: List[Dict[str, str]]):
        payload = {
            "model": "gpt-5-mini",
            "metadata": {
                "toolChoice": {
                    "NewsSearch": False,
                    "VideosSearch": False,
                    "LocalSearch": False,
                    "WeatherForecast": False,
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
                except:
                    continue
        return "".join(full_answer)

    async def ask(self, messages: List[Dict[str, str]]):
        await self._refresh_headers()
        return self._send(messages)


bot = DDGChat(headless=False)


class Query(BaseModel):
    messages: List[Dict[str, str]]


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
    answer = await bot.ask(query.messages)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn, sys

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")