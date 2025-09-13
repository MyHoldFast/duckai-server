# DuckAI Server

A simple **FastAPI and Pyppeteer** server that runs DuckDuckGo AI.  

---

## Installation

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

```bash
pip install fastapi uvicorn pyppeteer pyppeteer_stealth aiohttp requests pydantic
```

#### Gemini API Key (for CAPTCHA solving)

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **Get API key** → **Create API key**
4. Copy your generated key
5. Set it in the script:

```python
GEMINI_API_KEY = "your free gemini key"
```

---

## Running the Server

Normal run:

```bash
python duckai-pyppeteer.py
```

On Linux server (background mode):

```bash
nohup python duckai-pyppeteer.py > server.log 2>&1 & disown
```

Stop the server:

```bash
pkill -f "python.*duckai-pyppeteer.py"
pkill -f "chrome"
```

> Note: On some systems, use **Ctrl+\\** (SIGQUIT) instead of Ctrl+C to stop it properly.

---

## Basic Requests

### Request with default model (gpt-5-mini)

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "user", "content": "Hello, who are you?"}
  ]
}'
```
### Request with specific model
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "user", "content": "Hello, who are you?"}
  ],
  "model": "gpt-4o-mini"
}'
```

### Windows (CMD)

```powershell
curl -X POST "http://localhost:8000/ask" ^
-H "Content-Type: application/json" ^
-d "{\"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}], \"model\": \"gpt-5-mini\"}"
```
## Available Models

- `claude-3-5-haiku-latest`
- `mistralai/Mistral-Small-24B-Instruct-2501`
- `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- `gpt-4o-mini`
- `gpt-5-mini` (default model)