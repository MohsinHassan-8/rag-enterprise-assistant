"""Smoke test: one chat completion, to confirm key, model ID, and response shape."""
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("LLM_API_KEY", "")
base = os.getenv("LLM_BASE_URL", "").rstrip("/")
model = os.getenv("LLM_MODEL", "")
if not (key and base and model) or key.startswith(("paste_", "your_")):
    sys.exit("Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in .env first")

payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
    "temperature": 0,
    "max_tokens": 200,
}
t0 = time.perf_counter()
resp = httpx.post(
    f"{base}/chat/completions",
    json=payload,
    headers={"Authorization": f"Bearer {key}"},
    timeout=60,
)
elapsed = (time.perf_counter() - t0) * 1000
print(f"HTTP {resp.status_code} in {elapsed:.0f} ms")
if resp.status_code != 200:
    print(resp.text[:500])
    sys.exit(1)

data = resp.json()
choice = data["choices"][0]
msg = choice["message"]
print("model:", data.get("model"))
print("content:", repr(msg.get("content")))
print("finish_reason:", choice.get("finish_reason"))
print("message keys:", sorted(msg.keys()))
print("usage:", data.get("usage"))
for name, value in resp.headers.items():
    if name.lower().startswith(("x-ratelimit", "retry-after")):
        print(f"{name}: {value}")