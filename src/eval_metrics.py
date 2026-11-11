"""Retrieval evaluation metrics (Week 5). Titles identify articles."""
import math
from collections.abc import Iterable, Sequence


def _rel(relevant: Iterable[str]) -> set[str]:
    """Return the relevant items as a set. An empty collection is rejected."""
    rel = set(relevant)
    if not rel:
        raise ValueError("relevant must not be empty")
    return rel


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the k result slots that hold a relevant item. Divides by k, not by len(retrieved)."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _rel(relevant)
    return sum(1 for t in retrieved[:k] if t in rel) / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant items that appear at least once in the top k."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _rel(relevant)
    return len({t for t in retrieved[:k] if t in rel}) / len(rel)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> bool:
    """True if at least one relevant item appears in the top k."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _rel(relevant)
    return any(t in rel for t in retrieved[:k])


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 divided by the rank of the first relevant item, or 0.0 if none was retrieved."""
    rel = _rel(relevant)
    for i, t in enumerate(retrieved, start=1):
        if t in rel:
            return 1.0 / i
    return 0.0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (low, high) for successes out of n. z=1.96 gives 95%."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m
