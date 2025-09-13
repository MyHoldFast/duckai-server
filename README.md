# DuckAI Server

A **FastAPI + Pyppeteer** server for running DuckDuckGo AI with both **API** and **CLI** modes.

---

## Installation

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows

pip install fastapi uvicorn pyppeteer pyppeteer_stealth aiohttp requests pydantic python-dotenv
```

### Gemini API Key (for CAPTCHA solving)

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create and copy your API key
4. Set it as an environment variable:

```bash
export GEMINI_API_KEY="your_free_gemini_key"
```
or edit
```bash
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your_free_gemini_key")
```

---

## Running Modes

### Server Mode (default)

```bash
python duckai-pyppeteer.py
```

```bash
python duckai-pyppeteer.py --host 0.0.0.0 --port 8000
```

Run in background (Linux):

```bash
nohup python duckai-pyppeteer.py > server.log 2>&1 & disown
```

Stop the server:

```bash
pkill -f "python.*duckai-pyppeteer.py"
pkill -f "chrome"
```

---

### CLI Mode (Interactive Chat)

```bash
python duckai-pyppeteer.py --mode cli
```

**Available commands inside CLI:**

- `quit` / `exit` → Exit chat  
- `clear` → Clear conversation history  
- `model` → Change AI model  

---

## API Usage

Send requests to the FastAPI server:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hello, who are you?"}]
}'
```

With specific model:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hello"}],
  "model": "gpt-4o-mini"
}'
```

### Windows (CMD)

```powershell
curl -X POST "http://localhost:8000/ask" ^
-H "Content-Type: application/json" ^
-d "{\"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}], \"model\": \"gpt-5-mini\"}"
```

---

## Available Models

- `claude-3-5-haiku-latest`
- `mistralai/Mistral-Small-24B-Instruct-2501`
- `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- `gpt-4o-mini`
- `gpt-5-mini` (default)
