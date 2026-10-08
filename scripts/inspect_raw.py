"""Inspect the raw dataset: counts, length stats, samples, and noise indicators."""
import re
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "wiki_simple_raw.jsonl"

df = pd.read_json(PATH, lines=True)
lengths = df["text"].str.len()

print(f"Rows: {len(df)}")
print(f"Unique ids: {df['id'].nunique()} | unique titles: {df['title'].nunique()}")
print(f"Exact duplicate texts: {df['text'].duplicated().sum()}")
print(f"Chars per doc -> min {lengths.min()}, median {int(lengths.median())}, "
      f"mean {int(lengths.mean())}, max {lengths.max()}")
print(f"Total chars: {lengths.sum():,}")

print("\nNoise indicators (docs affected):")
print(f"  HTML tags:           {df['text'].str.contains(r'<[^>]+>', regex=True).sum()}")
print(f"  HTML entities:       {df['text'].str.contains(r'&[a-z]+;|&#\d+;', regex=True).sum()}")
print(f"  Non-ASCII chars:     {df['text'].str.contains(r'[^\x00-\x7f]', regex=True).sum()}")
print(f"  3+ consecutive \\n:   {df['text'].str.contains(r'\n{3,}', regex=True).sum()}")
print(f"  Double spaces:       {df['text'].str.contains(r' {2,}', regex=True).sum()}")
print(f"  Tabs:                {df['text'].str.contains(r'\t', regex=True).sum()}")

for i in (0, 1, 2):
    row = df.iloc[i]
    print(f"\n--- Sample {i}: {row['title']} ({len(row['text'])} chars) ---")
    print(repr(row["text"][:400]))