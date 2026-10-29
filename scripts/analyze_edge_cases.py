"""Do short or structurally unusual pages become topic outliers more often? (Week 4)

Reads data/processed/doc_themes.parquet and clean_corpus.parquet.
Writes reports/edge_case_analysis.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.stdout.reconfigure(errors="replace")
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
BINS = [0, 400, 800, 1500, 3000, np.inf]
LABELS = ["<400", "400-799", "800-1499", "1500-2999", "3000+"]


def row(name, x):
    n = len(x)
    out = int((x["theme"] == "Unclassified").sum())
    return {"group": name, "documents": n, "unclassified": out,
            "unclassified_pct": round(100.0 * out / n, 1) if n else None}


def main():
    corpus = pd.read_parquet(PROC / "clean_corpus.parquet")[["id", "char_count"]]
    th = pd.read_parquet(PROC / "doc_themes.parquet")
    d = th.merge(corpus, on="id", how="left")
    assert len(d) == len(th) and d["char_count"].notna().all()

    # Date and year pages were set aside by a title rule, so they are excluded here
    t = d[d["theme"] != "Dates and years"].copy()
    t["bucket"] = pd.cut(t["char_count"], BINS, labels=LABELS, right=False)

    rows = [row("length " + b, t[t["bucket"] == b]) for b in LABELS]
    rows.append(row("all topical documents", t))
    is_list = t["title"].str.startswith("List of")
    is_disamb = t["title"].str.contains("(disambiguation)", regex=False)
    rows.append(row("titles starting 'List of'", t[is_list]))
    rows.append(row("titles with '(disambiguation)'", t[is_disamb]))
    rows.append(row("all other titles", t[~is_list & ~is_disamb]))
    out = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    out.to_csv(REPORTS / "edge_case_analysis.csv", index=False)
    print(out.to_string(index=False))

    short = t["char_count"] < 800
    unc = t["theme"] == "Unclassified"
    table = [[int((unc & short).sum()), int((~unc & short).sum())],
             [int((unc & ~short).sum()), int((~unc & ~short).sum())]]
    chi2, p, _, _ = chi2_contingency(table)
    rs = table[0][0] / sum(table[0])
    rl = table[1][0] / sum(table[1])
    print("\nunder 800 chars: %.1f%% unclassified (n=%d) | 800+ chars: %.1f%% (n=%d) | chi-square p=%.3g"
          % (100 * rs, sum(table[0]), 100 * rl, sum(table[1]), p))
    print("median chars: unclassified %d | themed %d"
          % (t[unc]["char_count"].median(), t[~unc]["char_count"].median()))
    print("saved reports/edge_case_analysis.csv")


if __name__ == "__main__":
    main()
