"""Build the clean corpus: data/raw/wiki_simple_raw.jsonl -> data/processed/clean_corpus.parquet."""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.preprocessing import clean_document, detect_language

RAW_PATH = ROOT / "data" / "raw" / "wiki_simple_raw.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "clean_corpus.parquet"
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-chars", type=int, default=200,
                        help="minimum length AFTER cleaning")
    parser.add_argument("--min-docs", type=int, default=5000)
    args = parser.parse_args()

    df = pd.read_json(RAW_PATH, lines=True, dtype={"id": str})
    n_raw = len(df)
    print(f"Raw documents: {n_raw}")

    # 1. Clean
    df["text"] = [clean_document(t) for t in tqdm(df["text"], desc="Cleaning")]

    # 2. Drop documents that are too short after cleaning
    short = df["text"].str.len() < args.min_chars
    n_short = int(short.sum())
    df = df[~short]

    # 3. Language filter
    langs = [detect_language(t) for t in tqdm(df["text"], desc="Detecting language")]
    df = df.assign(lang=langs)
    non_en = df["lang"] != "en"
    n_non_en = int(non_en.sum())
    lang_counts = Counter(df.loc[non_en, "lang"])
    df = df[~non_en]

    # 4. Deduplicate on cleaned text
    n_dupes = int(df["text"].duplicated().sum())
    df = df.drop_duplicates(subset="text")

    # 5. Final columns
    df = df.drop(columns=["lang"]).reset_index(drop=True)
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    # 6. Validate before saving
    assert len(df) >= args.min_docs, f"Only {len(df)} docs left; need {args.min_docs}"
    assert df["id"].is_unique, "duplicate ids"
    assert not df["text"].str.contains(TAG_RE).any(), "HTML tags remain"
    assert not df["text"].str.contains(r"\n{3,}", regex=True).any(), "blank-line runs remain"
    assert not df["text"].str.contains(r" {2,}", regex=True).any(), "double spaces remain"
    assert not df["text"].str.contains("\t", regex=False).any(), "tabs remain"
    assert (df["text"].str.len() >= args.min_chars).all(), "short docs remain"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("\n=== Corpus build summary ===")
    print(f"Raw documents:          {n_raw}")
    print(f"Dropped (too short):    {n_short}")
    print(f"Dropped (non-English):  {n_non_en}  {dict(lang_counts.most_common(5))}")
    print(f"Dropped (duplicates):   {n_dupes}")
    print(f"Final documents:        {len(df)}")
    print(f"Chars/doc: median {int(df['char_count'].median())}, mean {int(df['char_count'].mean())}")
    print(f"Words total:            {int(df['word_count'].sum()):,}")
    print(f"Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print("All validation checks passed")


if __name__ == "__main__":
    main()