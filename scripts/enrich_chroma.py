"""Add NLP metadata (theme, topic id, places) to every chunk in ChromaDB (Week 4).

Updates metadata in place (vectors and text are untouched), verifies it, and
previews filtered-search latency. If the collection is rebuilt with
scripts/ingest_chroma.py --reset, run this script again.
Writes reports/metadata_log.json.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.place_utils import normalize_places
from src.vectorstore import (DEFAULT_COLLECTION, PLACE_FIELDS, build_where,
                             get_collection, search, update_metadata)

sys.stdout.reconfigure(errors="replace")
PROC = ROOT / "data" / "processed"
DB_PATH = ROOT / "chroma_db"
LOG_PATH = ROOT / "reports" / "metadata_log.json"
SPECIAL = {"Unclassified", "Dates and years"}


def median_ms(fn, reps=20):
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    chunks = pd.read_parquet(PROC / "chunks_embedded.parquet")
    emb = np.load(PROC / "embeddings.npy")
    themes = pd.read_parquet(PROC / "doc_themes.parquet")
    places = pd.read_parquet(PROC / "doc_places.parquet")
    assert len(chunks) == emb.shape[0], "chunks and embeddings misaligned"
    assert chunks["chunk_id"].is_unique
    assert set(chunks["doc_id"]) <= set(themes["id"]), "chunk doc_id without theme"
    assert set(chunks["doc_id"]) <= set(places["id"]), "chunk doc_id without places"

    theme_of = dict(zip(themes["id"], themes["theme"]))
    topic_of = dict(zip(themes["id"], themes["topic"]))
    places_of = {d: normalize_places(p) for d, p in zip(places["id"], places["places"])}

    ids = chunks["chunk_id"].tolist()
    metas = []
    for d, i, t, u in zip(chunks["doc_id"], chunks["chunk_index"], chunks["title"], chunks["url"]):
        pl = places_of[d]
        m = {"doc_id": str(d), "chunk_index": int(i), "title": str(t), "url": str(u),
             "theme": str(theme_of[d]), "topic_id": int(topic_of[d]), "places": "|".join(pl)}
        for k, f in enumerate(PLACE_FIELDS):
            m[f] = pl[k] if k < len(pl) else ""
        metas.append(m)

    col = get_collection(DB_PATH, DEFAULT_COLLECTION)
    assert col.count() == len(ids), "collection has %d items, expected %d" % (col.count(), len(ids))
    t0 = time.perf_counter()
    stats = update_metadata(col, ids, metas)
    secs = time.perf_counter() - t0
    print("updated %d chunks in %.1fs" % (stats["updated"], secs))

    # Verify: counts per theme in the database equal the expected counts
    got = {"ids": [], "metadatas": []}
    for off in range(0, len(ids), 2000):
        part = col.get(limit=2000, offset=off, include=["metadatas"])
        got["ids"] += part["ids"]
        got["metadatas"] += part["metadatas"]
    assert len(got["ids"]) == len(ids)
    stored = Counter(m["theme"] for m in got["metadatas"])
    expected = Counter(m["theme"] for m in metas)
    assert stored == expected, "theme counts differ after update"
    assert col.count() == len(ids), "update changed the chunk count"
    with_place = sum(1 for m in metas if m["place_1"])
    print("chunks with a place: %d (%.1f%%)" % (with_place, 100.0 * with_place / len(metas)))
    print("chunks per theme:", dict(stored.most_common()))

    # Verify filters return only matching chunks, and time them
    idx = next(i for i, m in enumerate(metas) if m["theme"] not in SPECIAL and m["place_1"])
    theme, place, q = metas[idx]["theme"], metas[idx]["place_1"], emb[idx]
    print("\nfilter test query: chunk %s | theme=%s | place=%s" % (ids[idx], theme, place))
    variants = {
        "unfiltered": None,
        "theme": build_where(theme=theme),
        "place": build_where(place=place),
        "theme+place": build_where(theme=theme, place=place),
    }
    log_lat = {}
    for name, where in variants.items():
        hits = search(col, q, 5, where=where)
        assert hits and hits[0]["id"] == ids[idx], "%s: query chunk not first" % name
        for h in hits:
            hm = h["metadata"]
            if name in ("theme", "theme+place"):
                assert hm["theme"] == theme, "%s: wrong theme %s" % (name, hm["theme"])
            if name in ("place", "theme+place"):
                assert place in [hm[f] for f in PLACE_FIELDS], "%s: place missing" % name
        ms = median_ms(lambda w=where: search(col, q, 5, where=w))
        log_lat[name] = round(ms, 2)
        print("  %-12s %d hits | search p50 %.2f ms" % (name, len(hits), ms))
    assert search(col, q, 5, where=build_where(theme="No such theme")) == []
    print("  no-match filter returns an empty list")

    log = {"chunks_updated": stats["updated"], "update_seconds": round(secs, 1),
           "chunks_with_place": with_place, "chunks_per_theme": dict(stored),
           "filter_test": {"theme": theme, "place": place},
           "search_p50_ms": log_lat}
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("\nsaved reports/metadata_log.json")
    print("All metadata checks passed")


if __name__ == "__main__":
    main()
