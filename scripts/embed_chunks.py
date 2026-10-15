"""Embed chunks with sentence-transformers and log timing.

Input : data/processed/chunks.parquet
Output: data/processed/chunks_embedded.parquet (git-ignored)
        data/processed/embeddings.npy          (git-ignored)
        reports/embedding_log.json             (committed)
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
import torch
from sentence_transformers import SentenceTransformer

IN_PATH = ROOT / "data" / "processed" / "chunks.parquet"
OUT_CHUNKS = ROOT / "data" / "processed" / "chunks_embedded.parquet"
OUT_EMB = ROOT / "data" / "processed" / "embeddings.npy"
LOG_PATH = ROOT / "reports" / "embedding_log.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-chunk-chars", type=int, default=50)
    args = parser.parse_args()

    chunks = pd.read_parquet(IN_PATH)
    n_total = len(chunks)
    chunks = chunks[chunks["char_count"] >= args.min_chunk_chars].reset_index(drop=True)
    n_dropped = n_total - len(chunks)
    print(f"Chunks: {n_total} total, {n_dropped} dropped (< {args.min_chunk_chars} chars), "
          f"{len(chunks)} to embed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else ""))
    if device == "cpu":
        print("WARNING: running on CPU, this will be much slower")

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device=device)
    load_time = time.perf_counter() - t0
    max_seq = model.max_seq_length
    print(f"Model loaded in {load_time:.1f}s (max_seq_length={max_seq} tokens)")

    texts = chunks["text"].tolist()

    # How many chunks exceed the model's token limit (and would be truncated)?
    token_lengths = [
        len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]
    ]
    n_truncated = int(sum(1 for n in token_lengths if n > max_seq))
    print(f"Tokens per chunk: median {int(np.median(token_lengths))}, "
          f"max {max(token_lengths)}; over limit: {n_truncated} "
          f"({n_truncated / len(texts) * 100:.2f}%)")

    # Warm-up so one-time CUDA initialisation is not counted in the benchmark
    t0 = time.perf_counter()
    model.encode(texts[:64], batch_size=args.batch_size, show_progress_bar=False)
    sync(device)
    warmup_time = time.perf_counter() - t0

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sync(device)
    encode_time = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None

    # Validate before saving
    assert embeddings.shape[0] == len(chunks), "row count mismatch"
    assert np.isfinite(embeddings).all(), "NaN or inf in embeddings"
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "embeddings are not normalized"

    OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, embeddings.astype(np.float32))
    chunks.to_parquet(OUT_CHUNKS, index=False)

    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "device": device,
        "gpu": gpu_name,
        "batch_size": args.batch_size,
        "embedding_dim": int(embeddings.shape[1]),
        "max_seq_length_tokens": int(max_seq),
        "chunks_total": int(n_total),
        "chunks_dropped_short": int(n_dropped),
        "chunks_embedded": int(len(chunks)),
        "chunks_over_token_limit": n_truncated,
        "median_tokens_per_chunk": int(np.median(token_lengths)),
        "model_load_seconds": round(load_time, 2),
        "warmup_seconds": round(warmup_time, 2),
        "encode_seconds": round(encode_time, 2),
        "chunks_per_second": round(len(chunks) / encode_time, 1),
        "ms_per_chunk": round(encode_time / len(chunks) * 1000, 3),
        "peak_gpu_memory_gb": round(peak_gb, 2) if peak_gb is not None else None,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\n=== Embedding summary ===")
    for key, value in log.items():
        print(f"{key}: {value}")
    print(f"\nSaved: {OUT_EMB} ({OUT_EMB.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved: {OUT_CHUNKS}")
    print(f"Saved: {LOG_PATH}")
    print("All validation checks passed")


if __name__ == "__main__":
    main()