import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.place_utils import normalize_places


def test_aliases_merged():
    assert normalize_places("U.S.|France|USA") == ["United States", "France"]


def test_stoplist_dropped():
    assert normalize_places("Earth|Spain|Middle Ages") == ["Spain"]


def test_dedup_keeps_order():
    assert normalize_places("Spain|France|Spain") == ["Spain", "France"]


def test_empty_input():
    assert normalize_places("") == []
    assert normalize_places(None) == []


def test_max_places():
    assert normalize_places("A1|B2|C3|D4|E5|F6", max_places=3) == ["A1", "B2", "C3"]
