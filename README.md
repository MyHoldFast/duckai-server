# DuckAI-server

A simple FastAPI server that runs DuckDuckGo AI via Playwright.

---

## Installing dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate # for Linux/macOS
venv\Scripts\activate # for Windows PowerShell

pip install fastapi uvicorn playwright requests pydantic
playwright install chromium
```

## Running the server

On Linux with a virtual X server (xvfb):
```bash
nohup xvfb-run -a -s "-screen 0 1920x1080x24" python server.py > server.log 2>&1 & disown
```

## Request examples
Linux (Bash)
```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{
"messages": [
{"role": "user", "content": "Hello, who are you?"}
]
}'
```
Windows (PowerShell)
```powershell
curl -Method POST "http://localhost:8000/ask" `
-ContentType "application/json" `
-Body '{
"messages": [
{"role": "user", "content": "Hello, who are you?"}
]
}'
```