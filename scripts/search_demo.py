"""Interactive semantic search demo.

Usage:
  python scripts/search_demo.py "why is the sky blue" --k 3
  python scripts/search_demo.py            # runs a few built-in queries
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from sentence_transformers import SentenceTransformer

from src.vectorstore import DEFAULT_COLLECTION, get_collection, search

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QUERIES = [
    "How do solar eclipses happen?",
    "What causes the greenhouse effect?",
    "Who was the first person to walk on the moon?",
    "How does photosynthesis work in plants?",
    "What is the capital city of Japan?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="search query (omit to run built-ins)")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    collection = get_collection(ROOT / "chroma_db", DEFAULT_COLLECTION)
    print(f"Collection '{DEFAULT_COLLECTION}': {collection.count()} chunks\n")

    queries = [args.query] if args.query else DEFAULT_QUERIES
    for q in queries:
        vec = model.encode(q, normalize_embeddings=True)
        print(f"=== {q} ===")
        for hit in search(collection, vec, n_results=args.k):
            meta = hit["metadata"]
            snippet = hit["text"][:220].replace("\n", " ")
            print(f"  {hit['rank']}. [{hit['score']:.3f}] {meta['title']} "
                  f"(chunk {meta['chunk_index']})")
            print(f"     {snippet}...")
        print()


if __name__ == "__main__":
    main()