"""List the models available to your API key, so you can pick a live model ID."""
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("LLM_API_KEY", "")
base = os.getenv("LLM_BASE_URL", "").rstrip("/")
if not key or key.startswith(("paste_", "your_")) or not base:
    sys.exit("Set LLM_API_KEY and LLM_BASE_URL in .env first")

resp = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
print("HTTP", resp.status_code)
resp.raise_for_status()
for m in sorted(resp.json()["data"], key=lambda m: m["id"]):
    print(f"{m['id']:60s} context={m.get('context_window')}")