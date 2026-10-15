"""Unit tests for src/chunking.py."""
import pytest

from src.chunking import chunk_document, chunk_text


def numbered_sentences(n: int) -> str:
    return " ".join(f"Sentence number {i} is here." for i in range(n))


class TestChunkText:
    def test_empty_and_whitespace_return_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_short_text_is_single_chunk(self):
        assert chunk_text("Short text.", 800, 100) == ["Short text."]

    def test_chunks_never_exceed_chunk_size(self):
        chunks = chunk_text(numbered_sentences(200), 200, 50)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_no_sentence_is_lost(self):
        chunks = chunk_text(numbered_sentences(60), 200, 50)
        joined = " ".join(chunks)
        for i in range(60):
            assert f"Sentence number {i} is here." in joined

    def test_overlap_repeats_content(self):
        chunks = chunk_text(numbered_sentences(60), 200, 50)
        assert len(chunks) > 1
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.split(". ")[0] in prev

    def test_zero_overlap_has_no_repeats(self):
        chunks = chunk_text(numbered_sentences(60), 200, 0)
        for i in range(60):
            sentence = f"Sentence number {i} is here."
            assert sum(c.count(sentence) for c in chunks) == 1

    def test_prefers_paragraph_boundaries(self):
        p1 = "Alpha beta gamma delta epsilon zeta eta theta iota kappa."
        p2 = "Lambda mu nu xi omicron pi rho sigma tau upsilon phi chi."
        assert chunk_text(p1 + "\n\n" + p2, 100, 0) == [p1, p2]

    def test_hard_splits_text_without_separators(self):
        text = "x" * 2500
        chunks = chunk_text(text, 1000, 0)
        assert len(chunks) == 3
        assert all(len(c) <= 1000 for c in chunks)
        assert "".join(chunks) == text

    @pytest.mark.parametrize(
        "size, overlap", [(0, 0), (-5, 0), (100, 100), (100, -1)]
    )
    def test_invalid_parameters_raise(self, size, overlap):
        with pytest.raises(ValueError):
            chunk_text("some text", size, overlap)

    def test_deterministic(self):
        text = numbered_sentences(80)
        assert chunk_text(text, 200, 50) == chunk_text(text, 200, 50)


class TestChunkDocument:
    def test_ids_and_metadata(self):
        chunks = chunk_document(
            "42", numbered_sentences(60), title="T", url="http://x", chunk_size=200
        )
        assert len(chunks) > 1
        first = chunks[0]
        assert first["chunk_id"] == "42_0"
        assert first["doc_id"] == "42"
        assert first["chunk_index"] == 0
        assert first["title"] == "T"
        assert first["url"] == "http://x"
        assert first["text"]

    def test_chunk_ids_are_unique(self):
        chunks = chunk_document("7", numbered_sentences(100), chunk_size=200)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_text_gives_no_chunks(self):
        assert chunk_document("1", "   ") == []

    def test_kwargs_are_passed_through(self):
        chunks = chunk_document("1", numbered_sentences(60), chunk_size=100, chunk_overlap=20)
        assert all(len(c["text"]) <= 100 for c in chunks)