# DuckAI-server

A simple FastAPI server that runs DuckDuckGo AI.
Two supported backends:

* **Playwright** (default, more stable)
* **Pyppeteer** (alternative, experimental)

---

## Installing dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows PowerShell
```

### For Playwright version

```bash
pip install fastapi uvicorn playwright requests pydantic
playwright install chromium
```

### For Pyppeteer version

```bash
pip install fastapi uvicorn pyppeteer pyppeteer_stealth aiohttp requests pydantic
```

**Important:** For the Pyppeteer version, set your Gemini API key in the script:

```python
GEMINI_API_KEY = "your free gemini key"
```

---

## Running the server

### Playwright version

On Linux with a virtual X server (xvfb):

```bash
nohup xvfb-run -a -s "-screen 0 1920x1080x24" python duckai-server.py > server.log 2>&1 & disown
```

To stop:

```bash
pkill -f "python.*duckai-server.py"
```

### Pyppeteer version

Run normally:

```bash
python duckai-pyppeteer.py
```

To stop, use **Ctrl+\\** (SIGQUIT) instead of Ctrl+C, because Ctrl+C may not terminate the server properly.

On Linux server, to keep it running after closing the SSH session:

```bash
nohup python duckai-pyppeteer.py > server.log 2>&1 &
disown
```

To stop:

```bash
pkill -f "python.*duckai-pyppeteer.py"
```

---

## Request examples

### Linux (Bash)

```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{
"messages": [
{"role": "user", "content": "Hello, who are you?"}
]
}'
```

### Windows (PowerShell)

```powershell
curl -X POST "http://localhost:8000/ask" ^
-H "Content-Type: application/json" ^
-d "{\"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}]}"
```
