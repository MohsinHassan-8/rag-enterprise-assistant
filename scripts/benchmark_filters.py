"""Benchmark filtered retrieval and check whether filters help (Week 4).

Writes reports/filter_benchmark.json. Run after scripts/enrich_chroma.py.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.vectorstore import DEFAULT_COLLECTION, build_where, get_collection, search

sys.stdout.reconfigure(errors="replace")
LOG_PATH = ROOT / "reports" / "filter_benchmark.json"
REPS = 10

# (question, theme, place or None, expected article title)
CASES = [
    ("Who was Henry VIII?", "History", "England", "Henry VIII"),
    ("What is the capital of France?", "Europe", "France", "France"),
    ("How do tornadoes form?", "Earth, space and weather", None, "Tornado"),
    ("What is a neutron star?", "Earth, space and weather", None, "Neutron star"),
    ("Who wrote Oliver Twist?", "Arts and literature", None, "Charles Dickens"),
    ("What is the Rhine?", "Europe", "Germany", "Rhine"),
    ("What does the oesophagus do?", "Human body and medicine", None, "Oesophagus"),
    ("What is Crohn's disease?", "Human body and medicine", None, "Crohn's disease"),
    ("Who is Sting?", "Music", None, "Sting"),
    ("What is StarCraft?", "Games", None, "StarCraft"),
    ("Where is Iquitos?", "Latin America and Caribbean", "Peru", "Iquitos"),
    ("What was Operation Overlord?", "History", None, "Operation Overlord"),
]


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000, out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    col = get_collection(ROOT / "chroma_db", DEFAULT_COLLECTION)
    vecs = model.encode([c[0] for c in CASES], normalize_embeddings=True)

    # Part 1: latency by filter type, and matching-set size
    variants = {"unfiltered": lambda c: None,
                "theme": lambda c: build_where(theme=c[1]),
                "place": lambda c: build_where(place=c[2]) if c[2] else None,
                "theme+place": lambda c: build_where(theme=c[1], place=c[2]) if c[2] else None}
    lat, matched = {}, {}
    search(col, vecs[0], 5)  # warm-up
    for name, mk in variants.items():
        times, sizes = [], []
        for c, v in zip(CASES, vecs):
            w = mk(c)
            if name != "unfiltered" and w is None:
                continue
            for _ in range(REPS):
                ms, _ = timed(lambda: search(col, v, 5, where=w))
                times.append(ms)
            if w is not None:
                sizes.append(len(col.get(where=w, limit=100000, include=[])["ids"]))
        lat[name] = {"n": len(times), "p50": round(float(np.percentile(times, 50)), 2),
                     "p95": round(float(np.percentile(times, 95)), 2)}
        matched[name] = {"min": min(sizes), "max": max(sizes)} if sizes else None
        print("%-12s n=%3d p50 %6.2f ms p95 %6.2f ms | chunks matched: %s"
              % (name, len(times), lat[name]["p50"], lat[name]["p95"], matched[name]))

    # Part 2: does the filter keep or hide the expected article?
    print("\nquestion | unfiltered top5 has expected | filtered top5 has expected | top score unfiltered -> filtered")
    rows, kept, hid, gained = 0, 0, 0, 0
    for c, v in zip(CASES, vecs):
        q, theme, place, expected = c
        base = search(col, v, 5)
        filt = search(col, v, 5, where=build_where(theme=theme, place=place))
        b = any(h["metadata"]["title"] == expected for h in base)
        f = any(h["metadata"]["title"] == expected for h in filt)
        rows += 1
        kept += b and f
        hid += b and not f
        gained += f and not b
        print("%s | %s | %s | %.3f -> %.3f" % (
            q, b, f, base[0]["score"], filt[0]["score"] if filt else float("nan")))
    print("\ncases %d | kept %d | filter hid the expected article %d | filter surfaced it %d"
          % (rows, kept, hid, gained))

    LOG_PATH.write_text(json.dumps({"latency_ms": lat, "chunks_matched": matched,
                                    "cases": rows, "kept": int(kept), "hidden": int(hid),
                                    "surfaced": int(gained), "reps": REPS}, indent=2),
                        encoding="utf-8")
    print("saved reports/filter_benchmark.json")


if __name__ == "__main__":
    main()
