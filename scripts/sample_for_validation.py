"""Draw a reproducible sample for manual validation of topic themes (Week 4)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(errors="replace")
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
N_THEMED, N_OUTLIER = 50, 10


def snippet(text, n=150):
    return " ".join(str(text).split())[:n]


def main():
    th = pd.read_parquet(PROC / "doc_themes.parquet")
    corpus = pd.read_parquet(PROC / "clean_corpus.parquet").set_index("id")
    th["text"] = th["id"].map(corpus["text"])
    rng = np.random.default_rng(42)

    themed = th[~th["theme"].isin(["Unclassified", "Dates and years"])]
    out = th[th["theme"] == "Unclassified"]
    a = themed.iloc[np.sort(rng.choice(len(themed), N_THEMED, replace=False))].copy()
    b = out.iloc[np.sort(rng.choice(len(out), N_OUTLIER, replace=False))].copy()
    a["group"] = "themed"
    b["group"] = "outlier"
    s = pd.concat([a, b], ignore_index=True)
    s.insert(0, "n", range(1, len(s) + 1))
    s["snippet"] = s["text"].map(snippet)
    REPORTS.mkdir(exist_ok=True)
    s[["n", "group", "id", "title", "topic", "theme", "snippet"]].to_csv(
        REPORTS / "theme_validation_sample.csv", index=False, encoding="utf-8-sig")

    for g, name in (("themed", "THEMED DOCUMENTS"), ("outlier", "UNCLASSIFIED OUTLIERS")):
        print("\n=== %s ===" % name)
        for _, r in s[s["group"] == g].iterrows():
            print("%d | %s | THEME: %s" % (r["n"], r["title"], r["theme"]))
            print("     %s" % r["snippet"])


if __name__ == "__main__":
    main()
