"""Chunk the clean corpus: clean_corpus.parquet -> data/processed/chunks.parquet."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.chunking import chunk_document

IN_PATH = ROOT / "data" / "processed" / "clean_corpus.parquet"
OUT_PATH = ROOT / "data" / "processed" / "chunks.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--min-chars", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_parquet(IN_PATH)
    print(f"Documents: {len(df)} | chunk_size={args.chunk_size}, overlap={args.chunk_overlap}")

    start = time.perf_counter()
    rows: list[dict] = []
    for doc in tqdm(df.itertuples(index=False), total=len(df), desc="Chunking"):
        rows.extend(
            chunk_document(
                str(doc.id),
                doc.text,
                title=doc.title,
                url=doc.url,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        )
    elapsed = time.perf_counter() - start

    chunks = pd.DataFrame(rows)
    chunks["char_count"] = chunks["text"].str.len()
    n_before = len(chunks)
    chunks = chunks[chunks["char_count"] >= args.min_chars].reset_index(drop=True)
    print(f"Dropped {n_before - len(chunks)} chunks under {args.min_chars} chars")

    # Validate before saving
    assert chunks["chunk_id"].is_unique, "duplicate chunk ids"
    assert (chunks["char_count"] > 0).all(), "empty chunks"
    assert (chunks["char_count"] <= args.chunk_size).all(), "oversized chunks"
    assert chunks["doc_id"].nunique() == len(df), "some documents produced no chunks"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(OUT_PATH, index=False)

    per_doc = chunks.groupby("doc_id").size()
    cc = chunks["char_count"]
    print("\n=== Chunk build summary ===")
    print(f"Chunking time:        {elapsed:.1f}s")
    print(f"Total chunks:         {len(chunks)}")
    print(f"Chunks per doc:       min {per_doc.min()}, median {int(per_doc.median())}, "
          f"mean {per_doc.mean():.1f}, max {per_doc.max()}")
    print(f"Chars per chunk:      min {cc.min()}, p5 {int(cc.quantile(0.05))}, "
          f"median {int(cc.median())}, mean {int(cc.mean())}, max {cc.max()}")
    print(f"Chunks under 100 chars: {int((cc < 100).sum())} "
          f"({(cc < 100).mean() * 100:.1f}%)")
    print(f"Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print("All validation checks passed")

    mid = chunks.iloc[len(chunks) // 2]
    print(f"\n--- Sample chunk ({mid['title']}, index {mid['chunk_index']}) ---")
    print(repr(mid["text"]))


if __name__ == "__main__":
    main()