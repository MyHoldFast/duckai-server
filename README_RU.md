# DuckAI-server

Простой FastAPI-сервер, который поднимает DuckDuckGo AI через Playwright.

---

## Установка зависимостей

Рекомендуется использовать виртуальное окружение:

```bash
python3 -m venv venv
source venv/bin/activate   # для Linux/macOS
venv\Scripts\activate      # для Windows PowerShell

pip install fastapi uvicorn playwright requests pydantic
playwright install chromium
```


## Запуск сервера

На Linux с виртуальным X-сервером (xvfb):
```bash
nohup xvfb-run -a -s "-screen 0 1920x1080x24" python server.py > server.log 2>&1 & disown
```

## Примеры запросов
Linux (Bash)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Привет, кто ты?"}
    ]
  }'
```
Windows (PowerShell)
```powershell
curl -X POST "http://localhost:8000/ask" ^
-H "Content-Type: application/json" ^
-d "{\"messages\": [{\"role\": \"user\", \"content\": \"Привет, кто ты?\"}]}"
```

