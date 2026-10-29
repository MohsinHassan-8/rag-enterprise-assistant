"""Average chunk embeddings into one normalized vector per document.

Used by topic modeling (Week 4). Verifies first that the stored embeddings
line up with chunks.parquet by re-embedding a random sample.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
DIM = 384


def main():
    chunks = pd.read_parquet(PROC / "chunks.parquet")
    corpus = pd.read_parquet(PROC / "clean_corpus.parquet")
    emb = np.load(PROC / "embeddings.npy")

    assert emb.shape == (len(chunks), DIM), emb.shape
    assert np.isfinite(emb).all(), "non-finite values in embeddings"
    assert set(chunks["doc_id"]) <= set(corpus["id"]), "chunk doc_id not in corpus"
    print("chunk embeddings:", emb.shape, "stored norm mean:", float(np.linalg.norm(emb, axis=1).mean()))

    # Alignment check: re-embed 30 random chunks and compare to stored rows
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(chunks), size=30, replace=False)
    fresh = model.encode(chunks["text"].iloc[idx].tolist(), normalize_embeddings=True)
    sims = (fresh * emb[idx]).sum(axis=1)
    print("alignment cosine: min %.4f mean %.4f" % (sims.min(), sims.mean()))
    assert sims.min() > 0.99, "stored embeddings do not match chunk order"

    # Average chunk vectors per document
    doc_ids = corpus["id"].tolist()
    pos = {d: i for i, d in enumerate(doc_ids)}
    rows = chunks["doc_id"].map(pos).to_numpy()
    sums = np.zeros((len(doc_ids), DIM), dtype=np.float64)
    counts = np.zeros(len(doc_ids), dtype=int)
    np.add.at(sums, rows, emb.astype(np.float64))
    np.add.at(counts, rows, 1)

    keep = counts > 0
    print("documents:", len(doc_ids), "| with zero chunks (excluded):", int((~keep).sum()))
    doc_emb = sums[keep] / counts[keep][:, None]
    doc_emb /= np.linalg.norm(doc_emb, axis=1, keepdims=True)
    doc_emb = doc_emb.astype(np.float32)

    index = corpus.loc[keep, ["id", "title"]].copy()
    index["n_chunks"] = counts[keep]
    index = index.reset_index(drop=True)
    assert len(index) == len(doc_emb)

    np.save(PROC / "doc_embeddings.npy", doc_emb)
    index.to_parquet(PROC / "doc_index.parquet", index=False)
    print("saved doc_embeddings.npy", doc_emb.shape, "and doc_index.parquet", index.shape)
    print("chunks per doc: median", int(index["n_chunks"].median()), "max", int(index["n_chunks"].max()))


if __name__ == "__main__":
    main()
