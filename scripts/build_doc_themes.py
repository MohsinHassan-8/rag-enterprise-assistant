"""Attach a broad theme to every document (Week 4).

Reads data/processed/doc_topics.parquet, writes data/processed/doc_themes.parquet
(git-ignored) and reports/theme_summary.csv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.topic_themes import DATES, UNCLASSIFIED, theme_for

sys.stdout.reconfigure(errors="replace")
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def main():
    dt = pd.read_parquet(PROC / "doc_topics.parquet")
    idx = pd.read_parquet(PROC / "doc_index.parquet")
    assert dt["id"].is_unique
    assert (dt["id"].to_numpy() == idx["id"].to_numpy()).all(), "doc order mismatch"

    dt["theme"] = dt["topic"].map(lambda t: theme_for(int(t)))
    dt.to_parquet(PROC / "doc_themes.parquet", index=False)

    summary = (dt.groupby("theme")
                 .agg(documents=("id", "count"), topics=("topic", "nunique"))
                 .sort_values("documents", ascending=False))
    assert int(summary["documents"].sum()) == len(dt)
    REPORTS.mkdir(exist_ok=True)
    summary.to_csv(REPORTS / "theme_summary.csv")
    print(summary.to_string())

    themed = dt[~dt["theme"].isin([UNCLASSIFIED, DATES])]
    print("\ndocuments:", len(dt))
    print("with a topic theme: %d (%.1f%%)" % (len(themed), 100.0 * len(themed) / len(dt)))
    print("unclassified outliers:", int((dt["theme"] == UNCLASSIFIED).sum()))
    print("date/year pages:", int((dt["theme"] == DATES).sum()))
    print("themes:", themed["theme"].nunique())
    print("saved doc_themes.parquet and reports/theme_summary.csv")


if __name__ == "__main__":
    main()
