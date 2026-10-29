"""Extract place entities (GPE, LOC) per document with spaCy (Week 4).

Only GPE and LOC are kept, because manual validation showed PERSON and ORG
are too unreliable to filter on (see reports/ner_validation_labels.csv).
Writes data/processed/doc_places.parquet (git-ignored).
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy

sys.stdout.reconfigure(errors="replace")
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
KEEP = {"GPE", "LOC"}
MAX_CHARS = 2000
TOP_N = 5


def clean_place(text):
    text = text.strip()
    if "\n" in text or len(text) > 40 or len(text) < 3:
        return None
    for prefix in ("the ", "The "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text or None


def main():
    corpus = pd.read_parquet(PROC / "clean_corpus.parquet")
    index = pd.read_parquet(PROC / "doc_index.parquet")
    corpus = corpus.set_index("id").loc[index["id"]].reset_index()

    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    texts = corpus["text"].str.slice(0, MAX_CHARS).tolist()

    places, overall = [], Counter()
    for doc in nlp.pipe(texts, batch_size=64):
        c = Counter()
        for e in doc.ents:
            if e.label_ in KEEP:
                p = clean_place(e.text)
                if p:
                    c[p] += 1
        top = [p for p, _ in c.most_common(TOP_N)]
        places.append("|".join(top))
        overall.update(top)

    out = pd.DataFrame({"id": corpus["id"], "title": corpus["title"], "places": places})
    out.to_parquet(PROC / "doc_places.parquet", index=False)

    has = (out["places"] != "").sum()
    print("documents:", len(out))
    print("with at least one place: %d (%.1f%%)" % (has, 100.0 * has / len(out)))
    print("distinct places:", len(overall))
    print("top 40 places (documents mentioning):")
    print(overall.most_common(40))
    print("saved doc_places.parquet")


if __name__ == "__main__":
    main()
