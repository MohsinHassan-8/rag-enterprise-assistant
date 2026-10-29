import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from src.topic_themes import (DATES, THEME_TOPICS, TOPIC_TO_THEME, UNCLASSIFIED,
                              theme_for)


def test_every_topic_id_mapped_exactly_once():
    # The saved topic run has ids 0..108
    assert sorted(TOPIC_TO_THEME) == list(range(109))


def test_no_topic_id_in_two_themes():
    flat = [t for ids in THEME_TOPICS.values() for t in ids]
    assert len(flat) == len(set(flat))


def test_special_topics():
    assert theme_for(-1) == UNCLASSIFIED
    assert theme_for(-2) == DATES


def test_unknown_topic_raises():
    with pytest.raises(ValueError):
        theme_for(500)


def test_theme_names_are_clean():
    names = list(THEME_TOPICS)
    assert len(names) == len(set(names))
    assert all(n and n.isascii() for n in names)
    assert UNCLASSIFIED not in names and DATES not in names


def test_saved_topics_all_covered():
    path = ROOT / "data" / "processed" / "doc_topics.parquet"
    if not path.exists():
        pytest.skip("doc_topics.parquet not built")
    topics = pd.read_parquet(path)["topic"].unique()
    for t in topics:
        theme_for(int(t))
