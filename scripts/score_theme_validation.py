"""Record manual validation verdicts for topic themes (Week 4).

Verdicts come from reading each title and the first 150 characters of text.
Edit the lists below to change a verdict, then re-run.
"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(errors="replace")
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

WRONG = [20, 50]
BORDERLINE = [3, 5, 8, 34, 37, 39, 44]
OUTLIER_MISS = [53, 56, 58, 59, 60]
OUTLIER_BORDERLINE = [51, 52]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m


def main():
    assert all(1 <= n <= 50 for n in WRONG + BORDERLINE), "themed numbers must be 1-50"
    assert all(51 <= n <= 60 for n in OUTLIER_MISS + OUTLIER_BORDERLINE), "outlier numbers must be 51-60"
    assert not set(WRONG) & set(BORDERLINE)

    s = pd.read_csv(REPORTS / "theme_validation_sample.csv", encoding="utf-8-sig")

    def verdict(r):
        n = int(r["n"])
        if r["group"] == "themed":
            if n in WRONG:
                return "wrong"
            return "borderline" if n in BORDERLINE else "correct"
        if n in OUTLIER_MISS:
            return "miss"
        return "borderline" if n in OUTLIER_BORDERLINE else "genuine_outlier"

    s["verdict"] = s.apply(verdict, axis=1)
    s.to_csv(REPORTS / "theme_validation_labels.csv", index=False, encoding="utf-8-sig")

    t = s[s["group"] == "themed"]
    n = len(t)
    c = int((t["verdict"] == "correct").sum())
    b = int((t["verdict"] == "borderline").sum())
    w = int((t["verdict"] == "wrong").sum())
    print("THEMED (n=%d): correct %d, borderline %d, wrong %d" % (n, c, b, w))
    for name, k in (("lenient (borderline counted correct)", c + b), ("strict (borderline counted wrong)", c)):
        lo, hi = wilson(k, n)
        print("  %s: %d/%d = %.1f%%  95%% CI %.1f%% to %.1f%%" % (name, k, n, 100.0 * k / n, 100 * lo, 100 * hi))
    print("  wrong items:", t[t["verdict"] == "wrong"][["n", "title", "theme"]].values.tolist())

    o = s[s["group"] == "outlier"]
    print("\nOUTLIERS (n=%d): %s" % (len(o), o["verdict"].value_counts().to_dict()))
    print("saved reports/theme_validation_labels.csv")


if __name__ == "__main__":
    main()
