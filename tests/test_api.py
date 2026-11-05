import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from src.api import KNOWN_THEMES, create_app
from src.rag import RAGResponse

SOURCE = {"number": 1, "id": "d1_0", "title": "Tokyo", "chunk_index": 0, "score": 0.71,
          "text": "Tokyo is the capital of Japan.", "theme": "Asia and Middle East",
          "places": "Japan"}


def make_response(status="ok", answer="Tokyo is the capital of Japan. [1]", **kw):
    base = dict(question="What is the capital of Japan?", status=status, answer=answer,
                sources=[SOURCE], citations=[1], top_score=0.71,
                latency_ms={"total_ms": 12.5}, usage={"total_tokens": 100})
    base.update(kw)
    return RAGResponse(**base)


class FakePipeline:
    def __init__(self, response=None, exc=None):
        self.response = response or make_response()
        self.exc = exc
        self.calls = []

    def answer(self, question, theme=None, place=None):
        self.calls.append((question, theme, place))
        if self.exc:
            raise self.exc
        return self.response


def make_client(tmp_path, pipeline=None):
    pipeline = pipeline or FakePipeline()
    log_path = tmp_path / "api.jsonl"
    return TestClient(create_app(pipeline=pipeline, log_path=log_path)), pipeline, log_path


def test_health(tmp_path):
    client, _, _ = make_client(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "pipeline_loaded": True}


def test_themes_lists_known_values(tmp_path):
    client, _, _ = make_client(tmp_path)
    themes = client.get("/themes").json()["themes"]
    assert themes == KNOWN_THEMES
    assert "Music" in themes and "Unclassified" in themes and "Dates and years" in themes


def test_ask_ok_shape_and_request_id(tmp_path):
    client, pipe, _ = make_client(tmp_path)
    r = client.post("/ask", json={"question": "What is the capital of Japan?"})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "ok" and body["citations"] == [1]
    assert body["sources"][0]["title"] == "Tokyo"
    assert body["request_id"] == r.headers["X-Request-ID"]
    assert pipe.calls == [("What is the capital of Japan?", None, None)]


def test_raw_answer_never_returned(tmp_path):
    resp = make_response(status="unverified", answer="withheld", raw_answer="SECRET RAW TEXT")
    client, _, _ = make_client(tmp_path, FakePipeline(resp))
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 200 and "SECRET RAW TEXT" not in r.text
    assert "raw_answer" not in r.json()


def test_filters_passed_to_pipeline(tmp_path):
    resp = make_response(filters={"theme": "Music", "place": "France"})
    client, pipe, _ = make_client(tmp_path, FakePipeline(resp))
    r = client.post("/ask", json={"question": "q", "theme": "Music", "place": "  France "})
    assert r.status_code == 200 and r.json()["filters"] == {"theme": "Music", "place": "France"}
    assert pipe.calls == [("q", "Music", "France")]


def test_off_topic_is_200(tmp_path):
    resp = make_response(status="off_topic", answer="not covered", sources=[], citations=[],
                         top_score=0.2)
    client, _, _ = make_client(tmp_path, FakePipeline(resp))
    r = client.post("/ask", json={"question": "kubernetes"})
    assert r.status_code == 200 and r.json()["status"] == "off_topic"


def test_missing_question_422(tmp_path):
    client, pipe, _ = make_client(tmp_path)
    r = client.post("/ask", json={})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert r.json()["detail"][0]["field"] == "question" and pipe.calls == []


def test_wrong_type_422(tmp_path):
    client, _, _ = make_client(tmp_path)
    assert client.post("/ask", json={"question": 123}).status_code == 422


def test_too_long_question_422(tmp_path):
    client, _, _ = make_client(tmp_path)
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422


def test_invalid_json_422(tmp_path):
    client, _, _ = make_client(tmp_path)
    r = client.post("/ask", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_blank_question_400(tmp_path):
    resp = RAGResponse(question="", status="invalid_query", answer="Please provide a non-empty question.")
    client, _, _ = make_client(tmp_path, FakePipeline(resp))
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    assert "request_id" in r.json()


def test_unknown_theme_400_skips_pipeline(tmp_path):
    client, pipe, _ = make_client(tmp_path)
    r = client.post("/ask", json={"question": "q", "theme": "No such theme"})
    assert r.status_code == 400 and "GET /themes" in r.json()["detail"] and pipe.calls == []


def test_blank_place_400(tmp_path):
    client, pipe, _ = make_client(tmp_path)
    r = client.post("/ask", json={"question": "q", "place": "   "})
    assert r.status_code == 400 and pipe.calls == []


def test_llm_error_502_with_sources(tmp_path):
    resp = make_response(status="llm_error", answer="The language model is unavailable.",
                         error="LLMAuthError: bad key")
    client, _, _ = make_client(tmp_path, FakePipeline(resp))
    r = client.post("/ask", json={"question": "q"})
    body = r.json()
    assert r.status_code == 502 and body["error"] == "upstream_error"
    assert body["sources"][0]["title"] == "Tokyo" and "bad key" not in r.text


def test_unexpected_error_500_hides_details(tmp_path):
    client, _, _ = make_client(tmp_path, FakePipeline(exc=RuntimeError("secret path C:/x")))
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 500 and r.json()["error"] == "internal_error"
    assert "secret" not in r.text and r.json()["request_id"] == r.headers["X-Request-ID"]


def test_unknown_path_404_json(tmp_path):
    client, _, _ = make_client(tmp_path)
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"] == "http_error"


def test_request_log_written(tmp_path):
    client, _, log_path = make_client(tmp_path)
    ok = client.post("/ask", json={"question": "q"})
    client.post("/ask", json={})
    lines = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines()]
    assert [x["status_code"] for x in lines] == [200, 422]
    assert lines[0]["request_id"] == ok.headers["X-Request-ID"]
    assert lines[0]["rag_status"] == "ok" and lines[1]["rag_status"] is None
    assert lines[0]["path"] == "/ask" and lines[0]["duration_ms"] >= 0
