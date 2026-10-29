"""Sample sentences and spaCy NER predictions for manual validation (Week 4)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

sys.stdout.reconfigure(errors="replace")
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
TYPES = {"PERSON", "ORG", "GPE", "LOC"}
N = 40


def main():
    corpus = pd.read_parquet(PROC / "clean_corpus.parquet")
    th = pd.read_parquet(PROC / "doc_themes.parquet")
    dates = set(th.loc[th["theme"] == "Dates and years", "id"])
    corpus = corpus[~corpus["id"].isin(dates)].reset_index(drop=True)

    nlp = spacy.load("en_core_web_sm")
    rng = np.random.default_rng(42)
    rows = []
    for i in rng.permutation(len(corpus)):
        r = corpus.iloc[int(i)]
        doc = nlp(r["text"][:1500])
        sents = [s for s in doc.sents
                 if 40 <= len(s.text) <= 250 and sum(c.isupper() for c in s.text) >= 2]
        if not sents:
            continue
        s = sents[int(rng.integers(len(sents)))]
        ents = [(e.text, e.label_) for e in s.ents if e.label_ in TYPES]
        rows.append({"n": len(rows) + 1, "id": r["id"], "title": r["title"],
                     "sentence": " ".join(s.text.split()), "predicted": repr(ents)})
        if len(rows) == N:
            break

    out = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    out.to_csv(REPORTS / "ner_validation_sample.csv", index=False, encoding="utf-8-sig")
    for _, r in out.iterrows():
        print("%d | %s | %s" % (r["n"], r["title"], r["sentence"]))
        print("     PRED: %s" % r["predicted"])
    print("\nsentences:", len(out), "| predicted entities:",
          sum(len(eval(p)) for p in out["predicted"]))


if __name__ == "__main__":
    main()
