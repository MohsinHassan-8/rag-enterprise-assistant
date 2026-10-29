import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.rag import FILTER_EMPTY_MESSAGE, RAGPipeline
from src.vectorstore import PLACE_FIELDS, get_collection, upsert_chunks

IDS = ["a", "b", "c"]
TEXTS = ["Paris is the capital of France.", "Madrid is the capital of Spain.",
         "A guitar is a musical instrument."]
EMB = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=np.float32)
QUESTION = "What is the capital of France?"


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_tokens=None, reasoning_effort=None):
        self.calls += 1
        return SimpleNamespace(text="Paris is the capital of France [1].", retry_wait_ms=0.0,
                               prompt_tokens=10, completion_tokens=5, reasoning_tokens=0,
                               total_tokens=15, attempts=1)


def meta(i, theme, *places):
    m = {"doc_id": IDS[i], "chunk_index": 0, "title": "T" + IDS[i], "theme": theme,
         "topic_id": i, "places": "|".join(places)}
    for k, f in enumerate(PLACE_FIELDS):
        m[f] = places[k] if k < len(places) else ""
    return m


@pytest.fixture
def pipe(tmp_path):
    col = get_collection(tmp_path / "db", "rag_filters", reset=True)
    upsert_chunks(col, IDS, EMB, TEXTS, [meta(0, "Europe", "France"),
                                         meta(1, "Europe", "Spain"), meta(2, "Music")])
    llm = FakeLLM()
    p = RAGPipeline(lambda q: np.array([1, 0, 0], dtype=np.float32), col, llm, log_path=None)
    return p, llm


def test_unfiltered_unchanged(pipe):
    p, llm = pipe
    r = p.answer(QUESTION)
    assert r.filters == {} and len(r.sources) == 3 and llm.calls == 1


def test_theme_filter_limits_sources(pipe):
    p, llm = pipe
    r = p.answer(QUESTION, theme="Europe")
    assert r.filters == {"theme": "Europe"}
    assert {s["theme"] for s in r.sources} == {"Europe"} and len(r.sources) == 2


def test_place_filter_limits_sources(pipe):
    p, _ = pipe
    r = p.answer(QUESTION, place="Spain")
    assert [s["id"] for s in r.sources] == ["b"]
    assert r.sources[0]["places"] == "Spain"


def test_filter_matching_nothing_skips_llm(pipe):
    p, llm = pipe
    r = p.answer(QUESTION, theme="Sports")
    assert r.status == "off_topic" and r.answer == FILTER_EMPTY_MESSAGE
    assert "no_results" in r.warnings and llm.calls == 0


def test_invalid_filter_skips_llm(pipe):
    p, llm = pipe
    r = p.answer(QUESTION, theme="")
    assert r.status == "invalid_query" and llm.calls == 0
