"""Retrieval evaluation metrics (Week 5). Titles identify articles."""
import math


def _rel(relevant):
    rel = set(relevant)
    if not rel:
        raise ValueError("relevant must not be empty")
    return rel


def precision_at_k(retrieved, relevant, k):
    """Share of the k result slots that hold a relevant item. Divides by k, not by len(retrieved)."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _rel(relevant)
    return sum(1 for t in retrieved[:k] if t in rel) / k


def recall_at_k(retrieved, relevant, k):
    """Share of the relevant items that appear at least once in the top k."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _rel(relevant)
    return len({t for t in retrieved[:k] if t in rel}) / len(rel)


def hit_at_k(retrieved, relevant, k):
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _rel(relevant)
    return any(t in rel for t in retrieved[:k])


def reciprocal_rank(retrieved, relevant):
    rel = _rel(relevant)
    for i, t in enumerate(retrieved, start=1):
        if t in rel:
            return 1.0 / i
    return 0.0


def wilson(successes, n, z=1.96):
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m
