"""Unit tests for src/rag.py (fake LLM, temporary ChromaDB, no network or API key)."""
import json

import numpy as np
import pytest

from src.llm_client import LLMRateLimitError, LLMResult, LLMTokenLimitError
from src.prompts import REFUSAL_SENTENCE
from src.rag import (
    INVALID_QUERY_MESSAGE,
    LLM_ERROR_MESSAGE,
    OFF_TOPIC_MESSAGE,
    UNVERIFIED_MESSAGE,
    RAGConfig,
    RAGPipeline,
    check_grounding,
    extract_citations,
)
from src.vectorstore import get_collection, upsert_chunks

DIM = 4
MOON_TEXT = "Neil Armstrong was the first person to walk on the moon in 1969."
QUESTION = "Who walked on the moon first?"
GOOD = "Neil Armstrong was the first person to walk on the moon in 1969 [1]."


def unit(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def embed(question):
    return unit(3) if "kubernetes" in question.lower() else unit(0)


@pytest.fixture
def col(tmp_path):
    collection = get_collection(tmp_path / "db", "test", reset=True)
    upsert_chunks(
        collection,
        ["a", "b", "c"],
        np.stack([unit(0), unit(1), unit(2)]),
        [
            MOON_TEXT,
            "Mars is the fourth planet from the Sun and has two small moons.",
            "The Pacific Ocean is the largest ocean on Earth.",
        ],
        [
            {"title": "Moon", "chunk_index": 0},
            {"title": "Mars", "chunk_index": 0},
            {"title": "Sea", "chunk_index": 0},
        ],
    )
    return collection


def llm_result(text):
    return LLMResult(
        text=text, model="fake", finish_reason="stop", prompt_tokens=100,
        completion_tokens=20, reasoning_tokens=5, total_tokens=120,
        latency_ms=1.0, retry_wait_ms=0.0, attempts=1,
    )


class FakeLLM:
    def __init__(self, *items):
        self.items = list(items)
        self.calls = []

    def chat(self, messages, max_tokens=1500, temperature=0.0, reasoning_effort=None):
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
        )
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return llm_result(item) if isinstance(item, str) else item


def make(col, tmp_path, *items, config=None):
    llm = FakeLLM(*items)
    return RAGPipeline(embed, col, llm, config, tmp_path / "logs" / "rag.jsonl"), llm


class TestExtractCitations:
    @pytest.mark.parametrize(
        "answer, expected",
        [
            ("[1][2]", [1, 2]),
            ("[1, 2]", [1, 2]),
            ("see [3].", [3]),
            ("[1] and [10]", [1, 10]),
            ("no citations", []),
            ("", []),
        ],
    )
    def test_extract(self, answer, expected):
        assert extract_citations(answer) == expected

    def test_none_input(self):
        assert extract_citations(None) == []


class TestCheckGrounding:
    def test_supported_sentence_passes(self):
        r = check_grounding("Paris is the capital of France.", "Paris is the capital city of France.")
        assert r["ok"]
        assert r["sentences_checked"] == 1
        assert r["min_support"] == 1.0

    def test_invented_number_fails(self):
        r = check_grounding("The population is 5000.", "The population is large.")
        assert not r["ok"]
        assert r["unsupported"][0]["unsupported_numbers"] == ["5000"]

    def test_low_support_fails(self):
        r = check_grounding("Dragons breathe fire over castles.", "The moon orbits the Earth.")
        assert not r["ok"]
        assert r["min_support"] < 0.6

    def test_question_words_count_as_supported(self):
        answer, context = "Cuisine sushi.", "Tokyo is known for sushi."
        assert not check_grounding(answer, context, "")["ok"]
        assert check_grounding(answer, context, "Tell me about cuisine")["ok"]

    def test_citation_markers_are_ignored(self):
        assert check_grounding("Paris is the capital [12].", "Paris is the capital of France.")["ok"]

    def test_list_markers_are_stripped(self):
        r = check_grounding(
            "- Paris is the capital.\n- Paris is in France.", "Paris is the capital of France."
        )
        assert r["ok"]
        assert r["sentences_checked"] == 2

    def test_empty_answer_checks_nothing(self):
        r = check_grounding("", "some context")
        assert r["ok"]
        assert r["sentences_checked"] == 0

    def test_simple_plurals_match(self):
        assert check_grounding("The eclipses happen.", "A solar eclipse happens.")["ok"]


class TestConfig:
    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.top_k == 5
        assert cfg.min_score == 0.40
        assert cfg.reasoning_effort == "low"
        assert cfg.withhold_unverified is True

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"top_k": 0},
            {"min_score": 2.0},
            {"max_context_chars": 0},
            {"max_tokens": 0},
            {"min_sentence_support": 1.5},
        ],
    )
    def test_invalid_values_raise(self, kwargs):
        with pytest.raises(ValueError):
            RAGConfig(**kwargs)


class TestPipeline:
    def test_ok_answer(self, col, tmp_path):
        pipe, _ = make(col, tmp_path, GOOD)
        r = pipe.answer(QUESTION)
        assert r.status == "ok"
        assert r.answer == GOOD
        assert r.raw_answer == GOOD
        assert r.citations == [1]
        assert r.warnings == []
        assert len(r.sources) == 3
        assert r.sources[0]["title"] == "Moon"
        assert r.top_score == pytest.approx(1.0, abs=1e-3)
        assert r.grounding["ok"]
        assert r.usage["attempts"] == 1

    def test_llm_receives_sources_and_config(self, col, tmp_path):
        pipe, llm = make(col, tmp_path, GOOD)
        pipe.answer(QUESTION)
        call = llm.calls[0]
        user = call["messages"][1]["content"]
        assert call["messages"][0]["role"] == "system"
        assert "Neil Armstrong" in user
        assert f"QUESTION: {QUESTION}" in user
        assert call["max_tokens"] == 700
        assert call["reasoning_effort"] == "low"

    def test_refusal(self, col, tmp_path):
        pipe, _ = make(col, tmp_path, REFUSAL_SENTENCE)
        r = pipe.answer(QUESTION)
        assert r.status == "refused"
        assert r.answer == REFUSAL_SENTENCE
        assert r.warnings == []
        assert r.citations == []
        assert r.grounding is None
        assert len(r.sources) == 3

    def test_off_topic_never_calls_llm(self, col, tmp_path):
        pipe, llm = make(col, tmp_path)
        r = pipe.answer("kubernetes pod autoscaling")
        assert r.status == "off_topic"
        assert r.answer == OFF_TOPIC_MESSAGE
        assert r.warnings == ["below_relevance_threshold"]
        assert r.sources == []
        assert r.top_score < 0.40
        assert r.latency_ms["llm_ms"] == 0
        assert llm.calls == []

    def test_empty_collection_is_off_topic(self, tmp_path):
        empty = get_collection(tmp_path / "empty_db", "empty", reset=True)
        pipe, llm = make(empty, tmp_path)
        r = pipe.answer(QUESTION)
        assert r.status == "off_topic"
        assert r.warnings == ["no_results"]
        assert r.top_score is None
        assert llm.calls == []

    @pytest.mark.parametrize("question", ["", "   "])
    def test_invalid_query(self, col, tmp_path, question):
        pipe, llm = make(col, tmp_path)
        r = pipe.answer(question)
        assert r.status == "invalid_query"
        assert r.answer == INVALID_QUERY_MESSAGE
        assert llm.calls == []

    def test_invalid_citation_is_withheld(self, col, tmp_path):
        text = "Neil Armstrong walked on the moon [7]."
        pipe, _ = make(col, tmp_path, text)
        r = pipe.answer(QUESTION)
        assert r.status == "unverified"
        assert r.answer == UNVERIFIED_MESSAGE
        assert r.raw_answer == text
        assert r.warnings == ["invalid_citation:7", "no_citation"]

    def test_missing_citation_is_withheld(self, col, tmp_path):
        text = "Neil Armstrong was the first person to walk on the moon."
        pipe, _ = make(col, tmp_path, text)
        r = pipe.answer(QUESTION)
        assert r.status == "unverified"
        assert r.answer == UNVERIFIED_MESSAGE
        assert r.raw_answer == text
        assert r.warnings == ["no_citation"]

    def test_flagged_when_withholding_disabled(self, col, tmp_path):
        text = "Neil Armstrong was the first person to walk on the moon."
        pipe, _ = make(col, tmp_path, text, config=RAGConfig(withhold_unverified=False))
        r = pipe.answer(QUESTION)
        assert r.status == "flagged"
        assert r.answer == text
        assert r.warnings == ["no_citation"]

    def test_unsupported_number_is_withheld(self, col, tmp_path):
        pipe, _ = make(col, tmp_path, "Neil Armstrong walked on the moon in 1971 [1].")
        r = pipe.answer(QUESTION)
        assert r.status == "unverified"
        assert r.warnings == ["unsupported_content"]

    def test_fabricated_content_is_withheld(self, col, tmp_path):
        text = "Armstrong later founded a successful pizza restaurant chain in Italy [1]."
        pipe, _ = make(col, tmp_path, text)
        r = pipe.answer(QUESTION)
        assert r.status == "unverified"
        assert r.warnings == ["unsupported_content"]
        assert r.grounding["unsupported"]

    def test_llm_error_returns_sources(self, col, tmp_path):
        pipe, _ = make(col, tmp_path, LLMRateLimitError("limit"))
        r = pipe.answer(QUESTION)
        assert r.status == "llm_error"
        assert r.answer == LLM_ERROR_MESSAGE
        assert len(r.sources) == 3
        assert r.error.startswith("LLMRateLimitError")

    def test_token_limit_retries_with_larger_budget(self, col, tmp_path):
        pipe, llm = make(col, tmp_path, LLMTokenLimitError("too big"), GOOD)
        r = pipe.answer(QUESTION)
        assert r.status == "ok"
        assert [c["max_tokens"] for c in llm.calls] == [700, 1050]
        assert r.usage["token_limit_retry"] is True

    def test_token_limit_twice_is_llm_error(self, col, tmp_path):
        pipe, llm = make(col, tmp_path, LLMTokenLimitError("a"), LLMTokenLimitError("b"))
        r = pipe.answer(QUESTION)
        assert r.status == "llm_error"
        assert len(llm.calls) == 2
        assert r.error.startswith("LLMTokenLimitError")

    def test_request_is_logged(self, col, tmp_path):
        pipe, _ = make(col, tmp_path, GOOD)
        pipe.answer(QUESTION)
        lines = (tmp_path / "logs" / "rag.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["status"] == "ok"
        assert rec["question"] == QUESTION
        assert rec["n_sources"] == 3
        assert rec["latency_ms"]["total_ms"] > 0
        assert "answer" not in rec

    def test_log_failure_does_not_break_answering(self, col, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        pipe = RAGPipeline(embed, col, FakeLLM(GOOD), None, blocker / "rag.jsonl")
        assert pipe.answer(QUESTION).status == "ok"

    def test_logging_can_be_disabled(self, col):
        pipe = RAGPipeline(embed, col, FakeLLM(GOOD), None, None)
        assert pipe.answer(QUESTION).status == "ok"

    def test_latency_breakdown_keys(self, col, tmp_path):
        pipe, _ = make(col, tmp_path, GOOD)
        lat = pipe.answer(QUESTION).latency_ms
        assert set(lat) == {
            "embed_ms", "search_ms", "retrieval_ms", "llm_ms",
            "llm_retry_wait_ms", "check_ms", "total_ms",
        }
        assert lat["total_ms"] >= lat["retrieval_ms"] > 0