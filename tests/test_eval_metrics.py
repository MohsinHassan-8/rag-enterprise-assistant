import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.eval_metrics import (hit_at_k, precision_at_k, recall_at_k,
                              reciprocal_rank, wilson)

R = ["a", "b", "c", "a", "d"]


def test_precision_counts_chunks_and_divides_by_k():
    assert precision_at_k(R, {"a"}, 5) == 2 / 5
    assert precision_at_k(R, {"a"}, 1) == 1.0
    assert precision_at_k(["a"], {"a"}, 5) == 1 / 5  # fewer results than k


def test_recall_counts_distinct_relevant_items():
    assert recall_at_k(R, {"a", "z"}, 5) == 0.5
    assert recall_at_k(R, {"a", "b"}, 2) == 1.0
    assert recall_at_k(R, {"z"}, 5) == 0.0


def test_hit_and_reciprocal_rank():
    assert hit_at_k(R, {"c"}, 2) is False and hit_at_k(R, {"c"}, 3) is True
    assert reciprocal_rank(R, {"c"}) == 1 / 3
    assert reciprocal_rank(R, {"z"}) == 0.0


def test_empty_results():
    assert precision_at_k([], {"a"}, 3) == 0.0
    assert recall_at_k([], {"a"}, 3) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        precision_at_k(R, {"a"}, 0)
    with pytest.raises(ValueError):
        recall_at_k(R, set(), 3)
    with pytest.raises(ValueError):
        hit_at_k(R, {"a"}, -1)
    with pytest.raises(ValueError):
        wilson(1, 0)


def test_wilson_bounds():
    lo, hi = wilson(9, 10)
    assert 0 < lo < 0.9 < hi < 1
    lo0, hi0 = wilson(0, 10)
    assert lo0 == pytest.approx(0, abs=1e-9) and hi0 > 0
