"""FastAPI service around the RAG pipeline (Week 5).

Endpoints: GET /health, GET /themes, POST /ask. Interactive docs at /docs.
Every error is JSON and carries a request_id:
  400  request the schema cannot catch (blank question, unknown theme, blank place)
  422  request body fails validation (missing field, wrong type, too long)
  502  the language model failed after retries (retrieved sources are included)
  500  unexpected server error (details go to the log, never to the client)
Run:  uvicorn src.api:app --port 8000
"""
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.topic_themes import DATES, THEME_TOPICS, UNCLASSIFIED

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_LOG = ROOT / "logs" / "api_requests.jsonl"
KNOWN_THEMES = sorted(THEME_TOPICS) + [UNCLASSIFIED, DATES]
MAX_QUESTION_CHARS = 1000

log = logging.getLogger("rag.api")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS,
                          description="The question to answer from the knowledge base.")
    theme: str | None = Field(None, max_length=100,
                              description="Only search chunks with this theme (see GET /themes).")
    place: str | None = Field(None, max_length=100,
                              description="Only search chunks that mention this place.")


class Source(BaseModel):
    number: int
    id: str | None = None
    title: str | None = None
    chunk_index: int | None = None
    score: float
    text: str = ""
    theme: str | None = None
    places: str | None = None


class AskResponse(BaseModel):
    request_id: str
    question: str
    status: str
    answer: str
    sources: list[Source] = []
    citations: list[int] = []
    warnings: list[str] = []
    top_score: float | None = None
    filters: dict[str, str] = {}
    latency_ms: dict[str, float] = {}
    usage: dict = {}


def _error(status_code, error, detail, request_id, **extra):
    body = {"error": error, "detail": detail, "request_id": request_id}
    body.update(extra)
    return JSONResponse(status_code=status_code, content=body)


def _write_log(path, record):
    if path is None:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break answering


def _default_factory():
    from src.rag import build_default_pipeline
    return build_default_pipeline()


def create_app(pipeline=None, pipeline_factory=None, log_path=DEFAULT_API_LOG) -> FastAPI:
    """Build the app. Pass a pipeline (tests) or let the lifespan load the real one."""

    @asynccontextmanager
    async def lifespan(app):
        if app.state.pipeline is None:
            app.state.pipeline = (pipeline_factory or _default_factory)()
        yield

    app = FastAPI(
        title="RAG Enterprise Assistant API",
        version="0.5.0",
        description="Grounded question answering over a 7,000-article Simple English "
                    "Wikipedia sample, with optional theme and place filters.",
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline
    app.state.lock = threading.Lock()  # one pipeline call at a time

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        request.state.request_id = rid
        request.state.rag_status = None
        t0 = time.perf_counter()
        error = None
        try:
            response = await call_next(request)
        except Exception as exc:  # unexpected: log details, hide them from the client
            log.exception("unhandled error in request %s", rid)
            error = type(exc).__name__
            response = _error(500, "internal_error", "Internal server error.", rid)
        response.headers["X-Request-ID"] = rid
        _write_log(log_path, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            "rag_status": request.state.rag_status,
            "client": request.client.host if request.client else None,
            "error": error,
        })
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        problems = [{"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
                     "message": e["msg"]} for e in exc.errors()]
        return _error(422, "validation_error", problems, getattr(request.state, "request_id", None))

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "http_error", exc.detail, getattr(request.state, "request_id", None))

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "pipeline_loaded": app.state.pipeline is not None}

    @app.get("/themes", tags=["meta"])
    def themes():
        return {"themes": KNOWN_THEMES}

    @app.post("/ask", response_model=AskResponse, tags=["rag"])
    def ask(body: AskRequest, request: Request):
        rid = request.state.request_id
        theme = body.theme
        place = body.place.strip() if body.place is not None else None
        if theme is not None and theme not in KNOWN_THEMES:
            return _error(400, "bad_request",
                          "Unknown theme %r. Use GET /themes for valid values." % theme, rid)
        if place is not None and not place:
            return _error(400, "bad_request", "place must not be blank.", rid)

        with app.state.lock:
            resp = app.state.pipeline.answer(body.question, theme=theme, place=place)
        request.state.rag_status = resp.status

        if resp.status == "invalid_query":
            return _error(400, "bad_request", resp.answer, rid)
        if resp.status == "llm_error":
            return _error(502, "upstream_error", resp.answer, rid,
                          sources=[Source(**s).model_dump() for s in resp.sources])
        # raw_answer is deliberately not returned: withheld answers must not leak
        return AskResponse(
            request_id=rid, question=resp.question, status=resp.status, answer=resp.answer,
            sources=[Source(**s) for s in resp.sources], citations=resp.citations,
            warnings=resp.warnings, top_score=resp.top_score, filters=resp.filters,
            latency_ms=resp.latency_ms, usage=resp.usage,
        )

    return app


app = create_app()
