import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.vectorstore import (PLACE_FIELDS, build_where, get_collection, search,
                             update_metadata, upsert_chunks)

IDS = ["a", "b", "c", "d"]


def meta(doc_id, theme, topic, *places):
    m = {"doc_id": doc_id, "theme": theme, "topic_id": topic}
    for k, f in enumerate(PLACE_FIELDS):
        m[f] = places[k] if k < len(places) else ""
    return m


@pytest.fixture
def col(tmp_path):
    c = get_collection(tmp_path / "db", "test_filters", reset=True)
    emb = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    upsert_chunks(c, IDS, emb, ["doc a", "doc b", "doc c", "doc d"],
                  [{"doc_id": i} for i in IDS])
    update_metadata(c, IDS, [
        meta("a", "Music", 1, "France"),
        meta("b", "Music", 1, "Spain", "France"),
        meta("c", "Sports", 2, "France"),
        meta("d", "Music", 1),
    ])
    return c, emb


def test_build_where_none():
    assert build_where() is None


def test_build_where_theme_only():
    assert build_where(theme="Music") == {"theme": "Music"}


def test_build_where_place_only():
    w = build_where(place="France")
    assert list(w) == ["$or"] and len(w["$or"]) == len(PLACE_FIELDS)


def test_build_where_theme_and_place():
    w = build_where(theme="Music", place="France")
    assert list(w) == ["$and"] and len(w["$and"]) == 2


def test_build_where_topic_zero_is_kept():
    assert build_where(topic_id=0) == {"topic_id": 0}


def test_build_where_rejects_bad_types():
    for kwargs in ({"theme": ""}, {"theme": 5}, {"topic_id": True},
                   {"topic_id": "1"}, {"place": ""}):
        with pytest.raises(ValueError):
            build_where(**kwargs)


def test_update_metadata_length_mismatch(col):
    with pytest.raises(ValueError):
        update_metadata(col[0], ["a", "b"], [{"x": 1}])


def test_update_metadata_duplicate_ids(col):
    with pytest.raises(ValueError):
        update_metadata(col[0], ["a", "a"], [{"x": 1}, {"x": 2}])


def test_update_metadata_rejects_none(col):
    with pytest.raises(ValueError):
        update_metadata(col[0], ["a"], [{"x": None}])


def test_theme_filter(col):
    c, emb = col
    hits = search(c, emb[0], 4, where=build_where(theme="Music"))
    assert {h["id"] for h in hits} == {"a", "b", "d"}


def test_place_filter_matches_any_slot(col):
    c, emb = col
    hits = search(c, emb[0], 4, where=build_where(place="France"))
    assert {h["id"] for h in hits} == {"a", "b", "c"}


def test_combined_filter(col):
    c, emb = col
    hits = search(c, emb[0], 4, where=build_where(theme="Music", place="France"))
    assert {h["id"] for h in hits} == {"a", "b"}


def test_filter_matching_nothing(col):
    c, emb = col
    assert search(c, emb[0], 4, where=build_where(theme="No such theme")) == []


def test_update_keeps_vectors_and_text(col):
    c, emb = col
    got = c.get(ids=["a"], include=["embeddings", "documents"])
    assert np.allclose(np.asarray(got["embeddings"][0]), emb[0])
    assert got["documents"][0] == "doc a"
