"""Ask the RAG pipeline one or more questions, optionally filtered.

Usage: python scripts/ask.py "question one" "question two" [--k 5]
       [--theme "Music"] [--place "France"]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import RAGConfig, build_default_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--theme", default=None, help="only search chunks with this theme")
    parser.add_argument("--place", default=None, help="only search chunks mentioning this place")
    args = parser.parse_args()

    pipeline = build_default_pipeline(RAGConfig(top_k=args.k))
    for q in args.questions:
        r = pipeline.answer(q, theme=args.theme, place=args.place)
        top = f"{r.top_score:.3f}" if r.top_score is not None else "n/a"
        print(f"\n=== {r.question} ===")
        print(f"status: {r.status} | top score: {top} | sources used: {len(r.sources)}")
        if r.filters:
            print(f"filters: {r.filters}")
        print(f"answer: {r.answer}")
        if r.raw_answer and r.raw_answer != r.answer:
            print(f"raw model answer (withheld): {r.raw_answer}")
        for s in r.sources:
            print(f"  [{s['number']}] {s['title']} (chunk {s['chunk_index']}) score {s['score']:.3f}"
                  f" | {s.get('theme')} | {s.get('places')}")
        if r.warnings:
            print("warnings:", ", ".join(r.warnings))
        if r.grounding and r.grounding["unsupported"]:
            for s in r.grounding["unsupported"]:
                print(f"  unsupported: {s['text']!r} support={s['support']} "
                      f"numbers={s['unsupported_numbers']}")
        if r.error:
            print("error:", r.error)
        print("latency ms: " + " | ".join(
            f"{k.removesuffix('_ms')} {v:.0f}" for k, v in r.latency_ms.items()))
        if r.usage:
            print("usage:", r.usage)


if __name__ == "__main__":
    main()
