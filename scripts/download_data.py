"""Download a Simple English Wikipedia sample to data/raw/wiki_simple_raw.jsonl."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "wiki_simple_raw.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7000, help="documents to save")
    parser.add_argument("--min-chars", type=int, default=300)
    args = parser.parse_args()

    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.simple", split="train", streaming=True
    )
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f, tqdm(total=args.n) as bar:
        for row in ds:
            text = row["text"]
            if len(text) < args.min_chars:
                continue
            record = {
                "id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "text": text,
                "source": "wikipedia_simple",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            bar.update(1)
            if saved >= args.n:
                break

    print(f"Saved {saved} documents to {OUT_PATH}")


if __name__ == "__main__":
    main()