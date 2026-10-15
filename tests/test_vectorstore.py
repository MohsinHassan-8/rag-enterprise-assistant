"""Unit tests for src/vectorstore.py (uses a temporary ChromaDB per test)."""
import numpy as np
import pytest

from src.vectorstore import get_collection, search, upsert_chunks

DIM = 4


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def col(tmp_path):
    return get_collection(tmp_path / "db", "test", reset=True)


def seed(col):
    ids = ["a", "b", "c"]
    embs = np.stack([unit(0), unit(1), unit(2)])
    docs = ["alpha text", "beta text", "gamma text"]
    metas = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
    return upsert_chunks(col, ids, embs, docs, metas)


class TestUpsert:
    def test_upsert_and_count(self, col):
        assert seed(col) == {"upserted": 3, "skipped_empty": 0}
        assert col.count() == 3

    def test_upsert_is_idempotent(self, col):
        seed(col)
        seed(col)
        assert col.count() == 3

    def test_upsert_updates_existing_chunk(self, col):
        seed(col)
        upsert_chunks(col, ["a"], np.stack([unit(0)]), ["new text"], [{"title": "A"}])
        assert col.count() == 3
        assert col.get(ids=["a"])["documents"] == ["new text"]

    def test_empty_documents_are_skipped(self, col):
        stats = upsert_chunks(
            col, ["a", "b", "c"], np.stack([unit(0), unit(1), unit(2)]), ["ok", "", "   "]
        )
        assert stats == {"upserted": 1, "skipped_empty": 2}
        assert col.count() == 1

    def test_mismatched_lengths_raise(self, col):
        with pytest.raises(ValueError, match="same length"):
            upsert_chunks(col, ["a", "b"], np.zeros((3, DIM)), ["x", "y"])

    def test_mismatched_metadatas_raise(self, col):
        with pytest.raises(ValueError, match="metadatas"):
            upsert_chunks(col, ["a", "b"], np.zeros((2, DIM)) + 1, ["x", "y"], [{"t": "1"}])

    def test_duplicate_ids_raise(self, col):
        with pytest.raises(ValueError, match="unique"):
            upsert_chunks(col, ["a", "a"], np.zeros((2, DIM)) + 1, ["x", "y"])
        assert col.count() == 0

    def test_nan_embeddings_raise(self, col):
        bad = np.array([[np.nan, 0, 0, 0]], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            upsert_chunks(col, ["a"], bad, ["x"])
        assert col.count() == 0

    def test_non_2d_embeddings_raise(self, col):
        flat = np.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="2-D"):
            upsert_chunks(col, ["a", "b", "c", "d"], flat, ["w", "x", "y", "z"])

    def test_invalid_batch_size_raises(self, col):
        with pytest.raises(ValueError, match="batch_size"):
            upsert_chunks(col, ["a"], np.stack([unit(0)]), ["x"], batch_size=0)

    def test_empty_input_is_a_noop(self, col):
        assert upsert_chunks(col, [], [], []) == {"upserted": 0, "skipped_empty": 0}
        assert col.count() == 0

    def test_batching_ingests_everything(self, col):
        rng = np.random.default_rng(0)
        n = 25
        embs = rng.random((n, DIM)) + 0.1
        upsert_chunks(
            col, [f"id{i}" for i in range(n)], embs, [f"doc {i}" for i in range(n)],
            batch_size=7,
        )
        assert col.count() == n

    def test_works_without_metadata(self, col):
        upsert_chunks(col, ["a"], np.stack([unit(0)]), ["text"])
        assert col.count() == 1
        assert search(col, unit(0), n_results=1)[0]["id"] == "a"


class TestSearch:
    def test_empty_collection_returns_empty_list(self, col):
        assert search(col, unit(0), n_results=5) == []

    def test_nearest_first_with_cosine_scores(self, col):
        seed(col)
        hits = search(col, unit(0), n_results=3)
        assert [h["rank"] for h in hits] == [1, 2, 3]
        assert hits[0]["id"] == "a"
        assert abs(hits[0]["score"] - 1.0) < 1e-4
        # Orthogonal vectors have cosine distance 1, so score 0
        assert abs(hits[1]["score"]) < 1e-4
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_n_results_larger_than_collection_is_clamped(self, col):
        seed(col)
        assert len(search(col, unit(0), n_results=10)) == 3

    @pytest.mark.parametrize("n", [0, -1])
    def test_invalid_n_results_raises(self, col, n):
        with pytest.raises(ValueError):
            search(col, unit(0), n_results=n)

    @pytest.mark.parametrize("vec", [[], [np.nan, 0.0, 0.0, 0.0]])
    def test_invalid_query_vector_raises(self, col, vec):
        seed(col)
        with pytest.raises(ValueError):
            search(col, vec, n_results=2)

    def test_metadata_filter(self, col):
        seed(col)
        hits = search(col, unit(0), n_results=3, where={"title": "B"})
        assert [h["id"] for h in hits] == ["b"]

    def test_metadata_filter_with_no_match_returns_empty(self, col):
        seed(col)
        assert search(col, unit(0), n_results=3, where={"title": "Z"}) == []


class TestCollection:
    def test_reset_clears_collection(self, col, tmp_path):
        seed(col)
        fresh = get_collection(tmp_path / "db", "test", reset=True)
        assert fresh.count() == 0

    def test_reopen_keeps_data(self, col, tmp_path):
        seed(col)
        again = get_collection(tmp_path / "db", "test")
        assert again.count() == 3