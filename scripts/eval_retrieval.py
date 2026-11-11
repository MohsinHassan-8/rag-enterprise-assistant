"""Retrieval evaluation: Precision@K, Recall@K, Hit@K and MRR (Week 5).

Questions were written from article titles in the corpus; relevance is judged by
article title. Writes reports/retrieval_eval.json and reports/retrieval_eval_cases.csv.
Run after the vector database is built.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.eval_metrics import (hit_at_k, precision_at_k, recall_at_k,
                              reciprocal_rank, wilson)
from src.vectorstore import DEFAULT_COLLECTION, get_collection, search

sys.stdout.reconfigure(errors="replace")
KS = (1, 3, 5, 10)

CASES = [
    ("What is a chord in music?", ["Chord"]),
    ("What is nihilism?", ["Nihilism"]),
    ("What is flash memory?", ["Flash memory"]),
    ("What is an adjective?", ["Adjective"]),
    ("What is the Japanese tea ceremony?", ["Japanese tea ceremony"]),
    ("How does electricity work?", ["Electricity"]),
    ("What is the British Museum?", ["British Museum", "Museum"]),
    ("What was the California Gold Rush?", ["California Gold Rush"]),
    ("How do you play Monopoly?", ["Monopoly (game)"]),
    ("What is a harmonica?", ["Harmonica"]),
    ("What is the game Tetris?", ["Tetris"]),
    ("What was the Russian Revolution?", ["Russian Revolution", "Vladimir Lenin"]),
    ("Who was Vladimir Lenin?", ["Vladimir Lenin", "Russian Revolution"]),
    ("Who was John Wilkes Booth?", ["John Wilkes Booth"]),
    ("Who was Charles Manson?", ["Charles Manson"]),
    ("What is a neuron?", ["Neuron"]),
    ("What is biodiversity?", ["Biodiversity"]),
    ("What is a gorilla?", ["Gorilla"]),
    ("What is Plato known for?", ["Plato"]),
    ("Where is Nova Scotia?", ["Nova Scotia"]),
    ("What is a greenhouse gas?", ["Greenhouse gas"]),
    ("What was the Winter War?", ["Winter War"]),
    ("What is a ferret?", ["Ferret"]),
    ("Who was Anne of Cleves?", ["Anne of Cleves", "Henry VIII"]),
    ("Where is Taiwan?", ["Taiwan"]),
    ("What is a tornado?", ["Tornado"]),
    ("Who is Elizabeth II?", ["Elizabeth II"]),
    ("What is a caracal?", ["Caracal"]),
    ("How is pizza made?", ["Pizza"]),
    ("What is a submarine?", ["Submarine"]),
    ("Who was Pierre Trudeau?", ["Pierre Trudeau"]),
    ("Who was Louis XII of France?", ["Louis XII"]),
    ("What is The Phantom of the Opera?", ["The Phantom of the Opera"]),
    ("What is humidity?", ["Humidity"]),
    ("What is the Denver Broncos?", ["Denver Broncos"]),
    ("Who was James K. Polk?", ["James K. Polk"]),
    ("What is whaling?", ["Whaling"]),
    ("How do beekeepers keep bees?", ["Beekeeping"]),
    ("What is the play Hamlet about?", ["Hamlet"]),
]


def main() -> None:
    """Score retrieval on the easy question set (Precision@K, Recall@K, Hit@K, MRR) and save the report."""
    corpus = pd.read_parquet(ROOT / "data" / "processed" / "clean_corpus.parquet")
    titles = set(corpus["title"])
    missing = sorted({t for _, rel in CASES for t in rel if t not in titles})
    if missing:
        print("These expected titles are not in the corpus:", missing)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    col = get_collection(ROOT / "chroma_db", DEFAULT_COLLECTION)
    vecs = model.encode([q for q, _ in CASES], normalize_embeddings=True)

    rows, agg = [], {k: {"p": [], "r": [], "hit": []} for k in KS}
    rr = []
    for (q, rel), v in zip(CASES, vecs):
        hits = search(col, v, max(KS))
        ret = [h["metadata"]["title"] for h in hits]
        rr.append(reciprocal_rank(ret, rel))
        first = next((i for i, t in enumerate(ret, 1) if t in set(rel)), None)
        for k in KS:
            agg[k]["p"].append(precision_at_k(ret, rel, k))
            agg[k]["r"].append(recall_at_k(ret, rel, k))
            agg[k]["hit"].append(hit_at_k(ret, rel, k))
        rows.append({"question": q, "relevant": " | ".join(rel), "first_relevant_rank": first,
                     "p_at_5": round(precision_at_k(ret, rel, 5), 3),
                     "r_at_5": round(recall_at_k(ret, rel, 5), 3),
                     "top5_titles": " | ".join(ret[:5]), "top1_score": round(hits[0]["score"], 3)})

    n = len(CASES)
    print("cases: %d | k values: %s | chunk-level precision, article-level recall\n" % (n, KS))
    print("%-4s %12s %10s %8s" % ("K", "Precision@K", "Recall@K", "Hit@K"))
    summary = {}
    for k in KS:
        p, r, h = (float(np.mean(agg[k]["p"])), float(np.mean(agg[k]["r"])),
                   float(np.mean(agg[k]["hit"])))
        summary[str(k)] = {"precision": round(p, 3), "recall": round(r, 3), "hit": round(h, 3)}
        print("%-4d %12.3f %10.3f %8.3f" % (k, p, r, h))
    mrr = float(np.mean(rr))
    h5 = int(sum(agg[5]["hit"]))
    lo, hi = wilson(h5, n)
    print("\nMRR: %.3f | Hit@5: %d/%d (95%% Wilson interval %.1f%% to %.1f%%)"
          % (mrr, h5, n, 100 * lo, 100 * hi))

    late = [r for r in rows if r["first_relevant_rank"] is None or r["first_relevant_rank"] > 1]
    print("\nQuestions where the first relevant chunk was not rank 1:")
    for r in late:
        print("  rank %s | %s | top3: %s" % (r["first_relevant_rank"], r["question"],
                                              " | ".join(r["top5_titles"].split(" | ")[:3])))

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    with open(out / "retrieval_eval_cases.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "retrieval_eval.json").write_text(json.dumps({
        "cases": n, "ks": list(KS), "metrics": summary, "mrr": round(mrr, 3),
        "hit_at_5": {"hits": h5, "n": n, "wilson_low": round(lo, 3), "wilson_high": round(hi, 3)},
    }, indent=2), encoding="utf-8")
    print("\nsaved reports/retrieval_eval.json and reports/retrieval_eval_cases.csv")


if __name__ == "__main__":
    main()
