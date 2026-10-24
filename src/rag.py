"""RAG pipeline: retrieve -> relevance gate -> generate -> verify -> log latency.

Statuses
  ok             answer returned; citations valid and every sentence lexically supported
  flagged        answer returned despite failed checks (only if withhold_unverified=False)
  unverified     answer withheld because a check failed (default policy)
  refused        the model said the sources do not contain the answer
  off_topic      best retrieval score below min_score; the LLM was not called
  llm_error      the LLM call failed after retries; the retrieved sources are still returned
  invalid_query  empty question
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.llm_client import LLMError, LLMTokenLimitError
from src.prompts import build_messages, clean_question, is_refusal
from src.vectorstore import DEFAULT_COLLECTION, get_collection, search

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = ROOT / "logs" / "rag_requests.jsonl"

OFF_TOPIC_MESSAGE = (
    "This question does not appear to be covered by the knowledge base, "
    "so I did not attempt an answer."
)
UNVERIFIED_MESSAGE = (
    "I could not verify an answer against the retrieved sources, so I am not returning one."
)
LLM_ERROR_MESSAGE = (
    "The language model is unavailable right now, so I could not generate an answer. "
    "The retrieved sources are included."
)
INVALID_QUERY_MESSAGE = "Please provide a non-empty question."

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her
    his how i if in into is it its may more most of on or our she should so some such than that
    the their them then there these they this those to was we were what when where which who
    why will with would you your also about after all any because before between both each few
    other over very just not no""".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_CITE_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


def extract_citations(answer: str) -> list[int]:
    """Return every cited source number in order, e.g. '[1][2]' or '[1, 2]' -> [1, 2]."""
    numbers: list[int] = []
    for match in _CITE_RE.finditer(answer or ""):
        numbers += [int(n) for n in re.findall(r"\d+", match.group(1))]
    return numbers


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(_CITE_RE.sub(" ", (text or "").lower()))


def _is_content_word(token: str) -> bool:
    return not token.isdigit() and token not in _STOPWORDS and len(token) >= 3


def _stem(word: str) -> str:
    """Very crude suffix stripping so 'eclipses' matches 'eclipse'."""
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def _content_stems(text: str) -> set[str]:
    return {_stem(t) for t in _tokens(text) if _is_content_word(t)}


def check_grounding(
    answer: str, context: str, question: str = "", min_support: float = 0.6
) -> dict:
    """Lexical grounding check, sentence by sentence.

    A sentence passes if at least min_support of its content words occur in the
    context or the question, and every number in it occurs in the context.
    """
    supported = _content_stems(context) | _content_stems(question)
    context_numbers = {t for t in _tokens(context) if t.isdigit()}
    sentences = []
    for raw in _SENT_SPLIT_RE.split(answer or ""):
        sent = _LIST_MARKER_RE.sub("", raw)
        tokens = _tokens(sent)
        words = [_stem(t) for t in tokens if _is_content_word(t)]
        numbers = [t for t in tokens if t.isdigit()]
        if not words and not numbers:
            continue
        support = sum(1 for w in words if w in supported) / len(words) if words else 1.0
        bad_numbers = [n for n in numbers if n not in context_numbers]
        sentences.append(
            {
                "text": sent.strip()[:200],
                "support": round(support, 2),
                "unsupported_numbers": bad_numbers,
                "ok": support >= min_support and not bad_numbers,
            }
        )
    return {
        "ok": all(s["ok"] for s in sentences),
        "min_support": min((s["support"] for s in sentences), default=1.0),
        "sentences_checked": len(sentences),
        "unsupported": [s for s in sentences if not s["ok"]],
    }


@dataclass
class RAGConfig:
    top_k: int = 5
    min_score: float = 0.40  # starting value from Week 2's benchmark; re-checked in Week 3 eval
    max_context_chars: int = 4500
    max_tokens: int = 700
    reasoning_effort: str | None = "low"
    min_sentence_support: float = 0.6
    withhold_unverified: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")
        if self.max_context_chars <= 0 or self.max_tokens <= 0:
            raise ValueError("max_context_chars and max_tokens must be positive")
        if not 0.0 <= self.min_sentence_support <= 1.0:
            raise ValueError("min_sentence_support must be between 0 and 1")


@dataclass
class RAGResponse:
    question: str
    status: str
    answer: str
    raw_answer: str | None = None
    sources: list[dict] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounding: dict | None = None
    top_score: float | None = None
    latency_ms: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _source_dicts(hits: list[dict]) -> list[dict]:
    out = []
    for number, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        out.append(
            {
                "number": number,
                "id": hit.get("id"),
                "title": meta.get("title"),
                "chunk_index": meta.get("chunk_index"),
                "score": round(hit["score"], 4),
                "text": hit.get("text") or "",
            }
        )
    return out


class RAGPipeline:
    def __init__(
        self,
        embed_fn: Callable[[str], object],
        collection,
        llm,
        config: RAGConfig | None = None,
        log_path: Path | str | None = DEFAULT_LOG_PATH,
    ):
        self.embed_fn = embed_fn
        self.collection = collection
        self.llm = llm
        self.config = config or RAGConfig()
        self.log_path = Path(log_path) if log_path is not None else None

    def answer(self, question: str) -> RAGResponse:
        cfg = self.config
        t_start = time.perf_counter()
        lat = dict.fromkeys(
            ("embed_ms", "search_ms", "retrieval_ms", "llm_ms",
             "llm_retry_wait_ms", "check_ms", "total_ms"),
            0.0,
        )

        try:
            q = clean_question(question)
        except ValueError:
            resp = RAGResponse(
                question=str(question or ""), status="invalid_query", answer=INVALID_QUERY_MESSAGE
            )
            return self._finish(resp, lat, t_start)

        # 1. Retrieve
        t0 = time.perf_counter()
        vec = self.embed_fn(q)
        t1 = time.perf_counter()
        hits = search(self.collection, vec, n_results=cfg.top_k)
        t2 = time.perf_counter()
        lat["embed_ms"] = (t1 - t0) * 1000
        lat["search_ms"] = (t2 - t1) * 1000
        lat["retrieval_ms"] = (t2 - t0) * 1000
        hits = [h for h in hits if (h.get("text") or "").strip()]
        top_score = hits[0]["score"] if hits else None

        # 2. Relevance gate: off-topic questions never reach the LLM
        if not hits or top_score < cfg.min_score:
            resp = RAGResponse(
                question=q,
                status="off_topic",
                answer=OFF_TOPIC_MESSAGE,
                warnings=["no_results" if not hits else "below_relevance_threshold"],
                top_score=top_score,
            )
            return self._finish(resp, lat, t_start)

        # 3. Generate
        t3 = time.perf_counter()
        try:
            used, result, retried = self._generate(q, hits)
        except LLMError as exc:
            lat["llm_ms"] = (time.perf_counter() - t3) * 1000
            resp = RAGResponse(
                question=q,
                status="llm_error",
                answer=LLM_ERROR_MESSAGE,
                sources=_source_dicts(hits),
                top_score=top_score,
                error=f"{type(exc).__name__}: {exc}",
            )
            return self._finish(resp, lat, t_start)
        lat["llm_ms"] = (time.perf_counter() - t3) * 1000
        lat["llm_retry_wait_ms"] = result.retry_wait_ms
        usage = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "total_tokens": result.total_tokens,
            "attempts": result.attempts,
            "token_limit_retry": retried,
        }

        # 4. Verify
        t4 = time.perf_counter()
        raw = result.text
        citations: list[int] = []
        warnings: list[str] = []
        grounding = None
        if is_refusal(raw):
            status, answer = "refused", raw
        else:
            citations = extract_citations(raw)
            invalid = sorted({n for n in citations if not 1 <= n <= len(used)})
            if invalid:
                warnings.append("invalid_citation:" + ",".join(map(str, invalid)))
            if not any(1 <= n <= len(used) for n in citations):
                warnings.append("no_citation")
            context = " ".join(
                f"{(h.get('metadata') or {}).get('title') or ''} {h['text']}" for h in used
            )
            grounding = check_grounding(raw, context, q, cfg.min_sentence_support)
            if not grounding["ok"]:
                warnings.append("unsupported_content")
            if warnings and cfg.withhold_unverified:
                status, answer = "unverified", UNVERIFIED_MESSAGE
            else:
                status, answer = ("flagged" if warnings else "ok"), raw
        lat["check_ms"] = (time.perf_counter() - t4) * 1000

        resp = RAGResponse(
            question=q,
            status=status,
            answer=answer,
            raw_answer=raw,
            sources=_source_dicts(used),
            citations=citations,
            warnings=warnings,
            grounding=grounding,
            top_score=top_score,
            usage=usage,
        )
        return self._finish(resp, lat, t_start)

    def _generate(self, question: str, hits: list[dict]):
        """Call the LLM; on a token-limit error retry once with half the context."""
        cfg = self.config
        budgets = [
            (cfg.max_context_chars, cfg.max_tokens),
            (max(500, cfg.max_context_chars // 2), int(cfg.max_tokens * 1.5)),
        ]
        for i, (chars, max_tokens) in enumerate(budgets):
            messages, used = build_messages(question, hits, chars)
            try:
                result = self.llm.chat(
                    messages, max_tokens=max_tokens, reasoning_effort=cfg.reasoning_effort
                )
                return used, result, i > 0
            except LLMTokenLimitError:
                if i == len(budgets) - 1:
                    raise
        raise RuntimeError("unreachable")

    def _finish(self, resp: RAGResponse, lat: dict, t_start: float) -> RAGResponse:
        lat["total_ms"] = (time.perf_counter() - t_start) * 1000
        resp.latency_ms = {k: round(v, 2) for k, v in lat.items()}
        self._log(resp)
        return resp

    def _log(self, resp: RAGResponse) -> None:
        if self.log_path is None:
            return
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": resp.question,
            "status": resp.status,
            "top_score": resp.top_score,
            "n_sources": len(resp.sources),
            "sources": [
                {k: s[k] for k in ("number", "title", "chunk_index", "score")}
                for s in resp.sources
            ],
            "citations": resp.citations,
            "warnings": resp.warnings,
            "answer_chars": len(resp.answer),
            "latency_ms": resp.latency_ms,
            "usage": resp.usage,
            "error": resp.error,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # logging must never break answering


def build_default_pipeline(
    config: RAGConfig | None = None, log_path: Path | str | None = DEFAULT_LOG_PATH
) -> RAGPipeline:
    """Wire up the real embedder, ChromaDB collection and LLM client from .env."""
    import torch
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer

    from src.llm_client import LLMClient

    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    collection = get_collection(ROOT / "chroma_db", DEFAULT_COLLECTION)
    llm = LLMClient.from_env(os.environ)

    def embed(text: str):
        return model.encode(text, normalize_embeddings=True)

    return RAGPipeline(embed, collection, llm, config, log_path)
