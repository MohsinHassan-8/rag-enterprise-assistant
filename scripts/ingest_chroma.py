"""Ingest embedded chunks into a persistent ChromaDB collection, then verify it.

Input : data/processed/chunks_embedded.parquet, data/processed/embeddings.npy
Output: chroma_db/ (git-ignored), reports/ingest_log.json (committed)
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.vectorstore import DEFAULT_COLLECTION, get_collection, search, upsert_chunks

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks_embedded.parquet"
EMB_PATH = ROOT / "data" / "processed" / "embeddings.npy"
DB_PATH = ROOT / "chroma_db"
LOG_PATH = ROOT / "reports" / "ingest_log.json"


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="delete the collection first and rebuild from scratch")
    args = parser.parse_args()

    chunks = pd.read_parquet(CHUNKS_PATH)
    emb = np.load(EMB_PATH)
    assert len(chunks) == emb.shape[0], "chunks and embeddings are misaligned"
    assert chunks["chunk_id"].is_unique, "duplicate chunk ids"
    print(f"Chunks: {len(chunks)} | embeddings: {emb.shape}")

    ids = chunks["chunk_id"].tolist()
    docs = chunks["text"].tolist()
    metas = [
        {"doc_id": str(d), "chunk_index": int(i), "title": str(t), "url": str(u)}
        for d, i, t, u in zip(chunks["doc_id"], chunks["chunk_index"],
                              chunks["title"], chunks["url"])
    ]

    collection = get_collection(DB_PATH, DEFAULT_COLLECTION, reset=args.reset)
    print(f"Collection '{DEFAULT_COLLECTION}' opened with {collection.count()} existing items")

    t0 = time.perf_counter()
    stats = upsert_chunks(collection, ids, emb, docs, metas)
    ingest_seconds = time.perf_counter() - t0
    count = collection.count()
    print(f"Upserted {stats['upserted']} chunks in {ingest_seconds:.1f}s "
          f"(skipped empty: {stats['skipped_empty']})")

    # Check 1: count matches. A higher count means stale ids from an older build.
    expected = len(chunks) - stats["skipped_empty"]
    assert count == expected, (
        f"collection has {count} items, expected {expected}; "
        "re-run with --reset to clear stale chunks"
    )

    # Check 2: re-upserting existing ids does not create duplicates
    upsert_chunks(collection, ids[:100], emb[:100], docs[:100], metas[:100])
    assert collection.count() == count, "upsert created duplicates"

    # Check 3: stored vectors and text round-trip unchanged
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    rng = np.random.default_rng(0)
    sample = [ids[i] for i in rng.choice(len(ids), size=20, replace=False)]
    got = collection.get(ids=sample, include=["embeddings", "documents"])
    for cid, vec, doc in zip(got["ids"], got["embeddings"], got["documents"]):
        i = id_to_idx[cid]
        assert np.allclose(np.asarray(vec), emb[i], atol=1e-5), f"vector mismatch {cid}"
        assert doc == docs[i], f"text mismatch {cid}"

    # Check 4: querying with a stored vector returns itself first, and the
    # reported distances match true cosine distance (confirms the metric)
    hits = search(collection, emb[0], n_results=5)
    assert hits[0]["id"] == ids[0], "stored vector did not retrieve itself"
    assert hits[0]["distance"] < 1e-3, "self-distance should be ~0"
    for h in hits:
        expected_dist = 1.0 - float(emb[0] @ emb[id_to_idx[h["id"]]])
        assert abs(h["distance"] - expected_dist) < 1e-3, "distance is not cosine"

    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": DEFAULT_COLLECTION,
        "distance_space": "cosine",
        "chunks_in_collection": count,
        "embedding_dim": int(emb.shape[1]),
        "ingest_seconds": round(ingest_seconds, 2),
        "chunks_per_second": round(stats["upserted"] / ingest_seconds, 1),
        "chroma_db_size_mb": round(dir_size_mb(DB_PATH), 1),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\n=== Ingest summary ===")
    for key, value in log.items():
        print(f"{key}: {value}")
    print(f"Saved: {LOG_PATH}")
    print("All validation checks passed")


if __name__ == "__main__":
    main()