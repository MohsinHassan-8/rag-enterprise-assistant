"""Score spaCy NER on the manually checked sample (Week 4).

Strict scoring: an entity counts only if span and type are both right.
Verdicts were proposed from the sentence text alone; edit and re-run to change.
"""
import ast
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(errors="replace")
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
TYPES = ["PERSON", "ORG", "GPE", "LOC"]

# sentence number -> positions (1-based, printed order) of wrong predictions
WRONG = {
    5: [2], 6: [1], 7: [1, 2], 9: [2, 3, 5], 11: [2], 18: [1], 22: [1, 2],
    23: [1], 28: [2, 3], 30: [1], 33: [1], 36: [1], 37: [1, 2], 38: [2, 3],
    40: [1, 4],
}
# sentence number -> gold entities that were not predicted correctly
MISSED = {
    5: [("Delos", "LOC")],
    7: [("Milano", "PERSON")],
    9: [("Makkah", "GPE"), ("Makkah al-Mukarramah", "GPE")],
    16: [("Mourning Beloveth", "ORG")],
    18: [("U.K.", "GPE")],
    23: [("Hemera", "PERSON")],
    33: [("Gilbertville", "GPE"), ("Simcoe", "GPE")],
    37: [("Cumbria", "GPE"), ("Lancashire", "GPE")],
    38: [("Alpheus S. Packard", "PERSON"), ("Pierce", "PERSON"), ("Bowdoin", "ORG")],
    40: [("United Kingdom", "GPE"), ("Greater Manchester", "GPE"),
         ("Greater Manchester", "GPE"), ("Manchester United F.C.", "ORG")],
}


def main():
    s = pd.read_csv(REPORTS / "ner_validation_sample.csv", encoding="utf-8-sig")
    assert len(s) == 40
    rows = []
    for _, r in s.iterrows():
        n = int(r["n"])
        preds = ast.literal_eval(r["predicted"])
        bad = WRONG.get(n, [])
        assert all(1 <= p <= len(preds) for p in bad), "bad position in sentence %d" % n
        for i, (text, label) in enumerate(preds, 1):
            rows.append({"n": n, "entity": text.replace("\n", " "), "label": label,
                         "verdict": "wrong" if i in bad else "correct"})
        for text, label in MISSED.get(n, []):
            rows.append({"n": n, "entity": text, "label": label, "verdict": "missed"})
    d = pd.DataFrame(rows)
    d.to_csv(REPORTS / "ner_validation_labels.csv", index=False, encoding="utf-8-sig")

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        f = 2 * p * r / (p + r) if tp and p + r else float("nan")
        return p, r, f

    print("%-7s %5s %5s %5s %9s %7s %6s" % ("type", "TP", "FP", "FN", "precision", "recall", "F1"))
    tot = [0, 0, 0]
    for t in TYPES + ["ALL"]:
        x = d if t == "ALL" else d[d["label"] == t]
        tp = int((x["verdict"] == "correct").sum())
        fp = int((x["verdict"] == "wrong").sum())
        fn = int((x["verdict"] == "missed").sum())
        p, r, f = prf(tp, fp, fn)
        print("%-7s %5d %5d %5d %8.1f%% %6.1f%% %5.1f%%" % (t, tp, fp, fn, 100 * p, 100 * r, 100 * f))
    print("\nsentences: %d | predicted entities: %d" % (len(s), int((d["verdict"] != "missed").sum())))
    print("saved reports/ner_validation_labels.csv")


if __name__ == "__main__":
    main()
