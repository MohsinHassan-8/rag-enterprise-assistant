"""Benchmark retrieval latency and run a small relevance smoke check.

Output: reports/retrieval_benchmark.json (committed)
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.vectorstore import DEFAULT_COLLECTION, get_collection, search

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_PATH = ROOT / "reports" / "retrieval_benchmark.json"

# (type, query, expected title substrings or None)
QUERIES = [
    ("question", "How do solar eclipses happen?", ["Solar eclipse"]),
    ("question", "What causes the greenhouse effect?",
     ["Greenhouse gas", "Water vapor", "Global warming"]),
    ("question", "Who was the first person to walk on the moon?",
     ["Neil Armstrong", "Buzz Aldrin"]),
    ("question", "How does photosynthesis work in plants?",
     ["Chloroplast", "Photosynthesis"]),
    ("question", "What is the capital city of Japan?", ["Tokyo"]),
    ("question", "Why do volcanoes erupt?", None),
    ("question", "How do vaccines protect people from disease?", None),
    ("keyword", "volcano", None),
    ("keyword", "Mozart", None),
    ("keyword", "democracy", None),
    ("keyword", "football", None),
    ("long", "I want to understand how the water cycle moves water between oceans, "
             "clouds, rivers and the ground over time", None),
    ("long", "Explain the main causes and consequences of the Second World War "
             "in Europe and the Pacific", None),
    ("long", "What are the differences between mammals, reptiles, birds and amphibians "
             "in terms of body temperature and reproduction", None),
    ("short_phrase", "largest planet in the solar system", None),
    ("short_phrase", "history of the printing press", None),
    ("numeric", "1969", None),
    ("numeric", "World War 2 1939 1945", None),
    ("out_of_domain", "kubernetes pod autoscaling configuration", None),
    ("nonsense", "asdfgh qwerty zxcvb", None),
]
K_SWEEP = (1, 5, 10, 20, 50)


def pct(values, p: float) -> float:
    return round(float(np.percentile(values, p)), 2)


def summarize(values) -> dict:
    return {
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "mean": round(float(np.mean(values)), 2),
    }


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20, help="repetitions per query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    model = SentenceTransformer(MODEL_NAME, device=device)
    collection = get_collection(ROOT / "chroma_db", DEFAULT_COLLECTION)
    n_chunks = collection.count()
    assert n_chunks > 0, "collection is empty; run scripts/ingest_chroma.py first"
    print(f"Device: {device}" + (f" ({gpu})" if gpu else "")
          + f" | collection: {n_chunks} chunks | k={args.k} | reps={args.reps}")

    # Cold first query (lazy initialisation, before any warm-up)
    t0 = time.perf_counter()
    vec = model.encode(QUERIES[0][1], normalize_embeddings=True)
    sync(device)
    search(collection, vec, n_results=args.k)
    cold_first_query_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Warm-up
    for _, q, _ in QUERIES[:5]:
        search(collection, model.encode(q, normalize_embeddings=True), n_results=args.k)

    # Main benchmark
    per_query = []
    all_embed, all_search, all_total = [], [], []
    total_by_type = defaultdict(list)
    relevance_checked = relevance_hits = 0

    for qtype, query, expected in QUERIES:
        embed_ms, search_ms, total_ms = [], [], []
        hits = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            vec = model.encode(query, normalize_embeddings=True)
            sync(device)
            t1 = time.perf_counter()
            hits = search(collection, vec, n_results=args.k)
            t2 = time.perf_counter()
            embed_ms.append((t1 - t0) * 1000)
            search_ms.append((t2 - t1) * 1000)
            total_ms.append((t2 - t0) * 1000)

        assert len(hits) == min(args.k, n_chunks), f"wrong result count for: {query}"
        titles = [h["metadata"]["title"] for h in hits]
        hit = None
        if expected:
            hit = any(e.lower() in t.lower() for e in expected for t in titles)
            relevance_checked += 1
            relevance_hits += int(hit)

        all_embed += embed_ms
        all_search += search_ms
        all_total += total_ms
        total_by_type[qtype] += total_ms
        per_query.append({
            "type": qtype,
            "query": query,
            "embed_ms_p50": pct(embed_ms, 50),
            "search_ms_p50": pct(search_ms, 50),
            "total_ms_p50": pct(total_ms, 50),
            "total_ms_p95": pct(total_ms, 95),
            "top1_score": round(hits[0]["score"], 3),
            "top1_margin": round(hits[0]["score"] - hits[1]["score"], 3) if len(hits) > 1 else None,
            "top1_title": titles[0],
            "top_titles": titles,
            "relevance_hit": hit,
        })

    assert np.isfinite(all_total).all(), "non-finite latency"

    # k sweep (search only, embeddings precomputed)
    vecs = [model.encode(q, normalize_embeddings=True) for _, q, _ in QUERIES]
    reps_k = max(3, args.reps // 4)
    k_sweep = {}
    for k in K_SWEEP:
        times = []
        for v in vecs:
            for _ in range(reps_k):
                t0 = time.perf_counter()
                search(collection, v, n_results=k)
                times.append((time.perf_counter() - t0) * 1000)
        k_sweep[str(k)] = summarize(times)

    # Metadata-filtered vs unfiltered search, same query vector
    q_vec = vecs[0]
    title = search(collection, q_vec, n_results=1)[0]["metadata"]["title"]

    def time_search(where):
        times, res = [], []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            res = search(collection, q_vec, n_results=args.k, where=where)
            times.append((time.perf_counter() - t0) * 1000)
        return res, times

    res_f, ts_f = time_search({"title": title})
    _, ts_u = time_search(None)
    assert res_f and all(h["metadata"]["title"] == title for h in res_f), "filter not applied"
    res_none = search(collection, q_vec, n_results=args.k, where={"title": "__no_such_title__"})
    assert res_none == [], "filter matching nothing should return an empty list"

    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "gpu": gpu,
        "model": MODEL_NAME,
        "collection_chunks": n_chunks,
        "k": args.k,
        "reps_per_query": args.reps,
        "n_queries": len(QUERIES),
        "cold_first_query_ms": cold_first_query_ms,
        "latency_ms": {
            "embed_query": summarize(all_embed),
            "chroma_search": summarize(all_search),
            "total": summarize(all_total),
        },
        "total_latency_by_query_type_ms": {t: summarize(v) for t, v in total_by_type.items()},
        "search_latency_by_k_ms": k_sweep,
        "filtered_search": {
            "filter": {"title": title},
            "filtered_ms": summarize(ts_f),
            "unfiltered_ms": summarize(ts_u),
        },
        "relevance_smoke_check": {
            "note": "5 queries with titles verified in the manual demo; not a full evaluation",
            "checked": relevance_checked,
            "hits_in_top_k": relevance_hits,
        },
        "per_query": per_query,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\n=== Latency (ms) ===")
    for name, s in log["latency_ms"].items():
        print(f"{name:14s} p50 {s['p50']:7.2f} | p95 {s['p95']:7.2f} | p99 {s['p99']:7.2f}")
    print(f"cold first query: {cold_first_query_ms} ms")

    print("\n=== By query type (total ms) ===")
    for t, s in log["total_latency_by_query_type_ms"].items():
        print(f"{t:14s} p50 {s['p50']:7.2f} | p95 {s['p95']:7.2f}")

    print("\n=== Chroma search latency by k (ms) ===")
    for k, s in k_sweep.items():
        print(f"k={k:>3s}  p50 {s['p50']:6.2f} | p95 {s['p95']:6.2f}")

    f = log["filtered_search"]
    print(f"\nFiltered (title='{title}') p50 {f['filtered_ms']['p50']} ms "
          f"vs unfiltered p50 {f['unfiltered_ms']['p50']} ms")

    print("\n=== Per query ===")
    for r in per_query:
        print(f"{r['type']:13s} {r['query'][:38]:38s} total p50 {r['total_ms_p50']:6.2f} "
              f"| top1 {r['top1_score']:.3f} (margin {r['top1_margin']}) {r['top1_title']}")

    print(f"\nRelevance smoke check: {relevance_hits}/{relevance_checked} queries "
          f"had an expected article in the top {args.k}")
    print(f"Saved: {OUT_PATH}")
    print("All validation checks passed")


if __name__ == "__main__":
    main()