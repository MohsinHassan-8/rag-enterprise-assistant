"""Call a running API with good and bad requests and print what comes back (Week 5).

Start the server first:  uvicorn src.api:app --port 8000
Usage: python scripts/smoke_api.py [--base http://127.0.0.1:8000]
"""
import argparse
import sys

import httpx

sys.stdout.reconfigure(errors="replace")

CASES = [
    ("GET", "/health", None),
    ("GET", "/themes", None),
    ("POST", "/ask", {"question": "Who was the first person to walk on the moon?"}),
    ("POST", "/ask", {"question": "Who was Henry VIII?", "theme": "History", "place": "England"}),
    ("POST", "/ask", {"question": "kubernetes pod autoscaling configuration"}),
    ("POST", "/ask", {"question": "   "}),
    ("POST", "/ask", {"question": "q", "theme": "No such theme"}),
    ("POST", "/ask", {}),
    ("POST", "/ask", {"question": "x" * 1001}),
    ("GET", "/nope", None),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with httpx.Client(base_url=args.base, timeout=90) as client:
        for method, path, body in CASES:
            r = client.request(method, path, json=body)
            j = r.json()
            label = (body or {}).get("question", "")[:45] if body is not None else ""
            if path == "/themes":
                summary = "%d themes" % len(j["themes"])
            elif r.status_code == 200 and path == "/ask":
                summary = "status=%s sources=%d top=%s answer=%r" % (
                    j["status"], len(j["sources"]), j["top_score"], j["answer"][:70])
            else:
                summary = "error=%s detail=%s" % (j.get("error", j.get("status")), str(j.get("detail", ""))[:80])
            print("%d %-5s %-7s %-45s %s | id=%s" % (
                r.status_code, method, path, repr(label), summary, r.headers.get("X-Request-ID")))


if __name__ == "__main__":
    main()
