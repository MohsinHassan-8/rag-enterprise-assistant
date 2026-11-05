"""Harder retrieval evaluation: paraphrased questions that avoid the article title (Week 5).

Reported separately from scripts/eval_retrieval.py. Writes reports/retrieval_eval_hard.json
and reports/retrieval_eval_hard_cases.csv.
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
    ("Which instrument do luthiers build?", ["Luthier"]),
    ("What do you call a disease that spreads quickly through a population?", ["Epidemic"]),
    ("Which animal is a small wild cat with long ear tufts?", ["Caracal"]),
    ("Why does the air feel sticky on hot summer days?", ["Humidity"]),
    ("Who assassinated Abraham Lincoln?", ["John Wilkes Booth"]),
    ("How did people hunt large sea animals for oil?", ["Whaling"]),
    ("What was the movement that started in 1917 and ended the Tsar's rule in Russia?",
     ["Russian Revolution", "Vladimir Lenin"]),
    ("What do bees make and who looks after the hives?", ["Beekeeping"]),
    ("What are the chemical gases that trap heat in the atmosphere?", ["Greenhouse gas"]),
    ("Which Canadian prime minister served in the 1970s?", ["Pierre Trudeau"]),
    ("What is a tube of rotating wind that touches the ground?", ["Tornado"]),
    ("Which brain cell sends electrical signals?", ["Neuron"]),
]


def main():
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

    rows, agg, rr = [], {k: {"p": [], "r": [], "hit": []} for k in KS}, []
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
                     "top5_titles": " | ".join(ret[:5]), "top1_score": round(hits[0]["score"], 3)})

    n = len(CASES)
    print("cases: %d (paraphrased, title not in the question)\n" % n)
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

    print("\nPer question (first relevant rank | top 3 titles):")
    for r in rows:
        print("  rank %s | %s\n      top3: %s" % (r["first_relevant_rank"], r["question"],
                                                 " | ".join(r["top5_titles"].split(" | ")[:3])))

    out = ROOT / "reports"
    with open(out / "retrieval_eval_hard_cases.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "retrieval_eval_hard.json").write_text(json.dumps({
        "cases": n, "ks": list(KS), "metrics": summary, "mrr": round(mrr, 3),
        "hit_at_5": {"hits": h5, "n": n, "wilson_low": round(lo, 3), "wilson_high": round(hi, 3)},
    }, indent=2), encoding="utf-8")
    print("\nsaved reports/retrieval_eval_hard.json and reports/retrieval_eval_hard_cases.csv")


if __name__ == "__main__":
    main()
