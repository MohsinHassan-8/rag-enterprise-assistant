"""Minimal OpenAI-compatible chat client with timeouts, retries and typed errors.

Works with any provider that exposes POST {base_url}/chat/completions
(Groq, OpenRouter, ...). Configuration comes from the constructor or from
LLM_API_KEY / LLM_BASE_URL / LLM_MODEL via from_env().
"""
from __future__ import annotations

import random
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass

import httpx

PLACEHOLDER_PREFIXES = ("paste_", "your_")


class LLMError(Exception):
    """Base class for client errors. Unclassified HTTP errors use this type directly."""


class LLMConfigError(LLMError):
    """Missing or placeholder configuration."""


class LLMAuthError(LLMError):
    """401 or 403: key missing, invalid, or not allowed to use the model."""


class LLMModelNotFoundError(LLMError):
    """404: the model ID does not exist (provider model lists change often)."""


class LLMTokenLimitError(LLMError):
    """Prompt too large, or the completion budget ran out before any answer text."""


class LLMBadResponseError(LLMError):
    """HTTP 200, but the body was unusable."""


class LLMRateLimitError(LLMError):
    """429 that persisted through all retries, or needs a wait longer than max_retry_wait."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMServerError(LLMError):
    """5xx that persisted through all retries."""


class LLMTimeoutError(LLMError):
    """The request timed out on every attempt."""


class LLMConnectionError(LLMError):
    """Network failure on every attempt."""


_RETRYABLE = (LLMRateLimitError, LLMServerError, LLMTimeoutError, LLMConnectionError)

_TOKEN_LIMIT_MARKERS = (
    "request too large",
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "reduce your message size",
    "reduce the length",
)

_RETRY_HINT_RE = re.compile(
    r"try again in\s+(?:(\d+)m(?!s))?\s*(?:([\d.]+)(ms|s))?", re.IGNORECASE
)


def retry_hint_seconds(message: str) -> float | None:
    """Parse hints such as 'try again in 1m26.4s' or 'try again in 250ms'."""
    match = _RETRY_HINT_RE.search(message or "")
    if not match:
        return None
    minutes, value, unit = match.groups()
    total = 0.0
    try:
        if minutes:
            total += int(minutes) * 60
        if value:
            total += float(value) / (1000.0 if unit.lower() == "ms" else 1.0)
    except ValueError:
        return None
    return total or None


def _retry_after_header(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _error_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    except (ValueError, AttributeError):
        pass
    return resp.text[:300]


@dataclass
class LLMResult:
    text: str
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    latency_ms: float  # wall time of the whole call, including retry waits
    retry_wait_ms: float  # time spent sleeping between attempts
    attempts: int


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        max_retry_wait: float = 60.0,
        jitter: float = 0.1,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ):
        if not api_key or api_key.startswith(PLACEHOLDER_PREFIXES):
            raise LLMConfigError("LLM_API_KEY is missing or still a placeholder")
        if not base_url:
            raise LLMConfigError("LLM_BASE_URL is missing")
        if not model:
            raise LLMConfigError("LLM_MODEL is missing")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.max_retry_wait = max_retry_wait
        self.jitter = jitter
        self._sleep = sleep
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str], **kwargs) -> "LLMClient":
        """Build a client from a mapping such as os.environ (call load_dotenv() first)."""
        return cls(
            api_key=env.get("LLM_API_KEY", ""),
            base_url=env.get("LLM_BASE_URL", ""),
            model=env.get("LLM_MODEL", ""),
            **kwargs,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1500,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        if not messages:
            raise ValueError("messages must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        start = time.perf_counter()
        waited = 0.0
        last_error: LLMError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post(f"{self.base_url}/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(f"request timed out after {self.timeout:.0f}s")
                last_error.__cause__ = exc
            except httpx.TransportError as exc:
                last_error = LLMConnectionError(f"network error: {type(exc).__name__}")
                last_error.__cause__ = exc
            else:
                if resp.status_code == 200:
                    return self._parse(resp, start, attempt + 1, waited)
                err = self._error_from_response(resp)
                if not isinstance(err, _RETRYABLE):
                    raise err
                if (
                    isinstance(err, LLMRateLimitError)
                    and err.retry_after is not None
                    and err.retry_after > self.max_retry_wait
                ):
                    raise err  # e.g. daily quota exhausted: waiting is pointless
                last_error = err
            if attempt < self.max_retries:
                delay = self._delay(attempt, last_error)
                self._sleep(delay)
                waited += delay
        raise last_error

    def _delay(self, attempt: int, err: LLMError) -> float:
        retry_after = getattr(err, "retry_after", None)
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(self.backoff_max, self.backoff_base * (2**attempt))
        return delay * (1.0 + self.jitter * random.random())

    def _error_from_response(self, resp: httpx.Response) -> LLMError:
        status = resp.status_code
        message = _error_message(resp)
        low = message.lower()
        if status in (401, 403):
            return LLMAuthError(f"authentication failed (HTTP {status}): {message}")
        if status == 404:
            return LLMModelNotFoundError(f"model '{self.model}' not found: {message}")
        if status == 413 or (
            status in (400, 429) and any(m in low for m in _TOKEN_LIMIT_MARKERS)
        ):
            return LLMTokenLimitError(f"request exceeds a token limit (HTTP {status}): {message}")
        if status == 429:
            retry_after = _retry_after_header(resp.headers.get("retry-after"))
            if retry_after is None:
                retry_after = retry_hint_seconds(message)
            return LLMRateLimitError(f"rate limited (HTTP 429): {message}", retry_after)
        if status >= 500:
            return LLMServerError(f"provider error (HTTP {status}): {message}")
        return LLMError(f"HTTP {status}: {message}")

    def _parse(self, resp: httpx.Response, start: float, attempts: int, waited: float) -> LLMResult:
        try:
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMBadResponseError("response body is not a valid chat completion") from exc
        content = (message.get("content") or "").strip()
        finish = choice.get("finish_reason")
        if not content:
            if finish == "length":
                raise LLMTokenLimitError(
                    "completion budget was used up before any answer text "
                    "(reasoning models may need a larger max_tokens)"
                )
            raise LLMBadResponseError("response had no answer text")
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return LLMResult(
            text=content,
            model=data.get("model", self.model),
            finish_reason=finish,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_ms=(time.perf_counter() - start) * 1000,
            retry_wait_ms=waited * 1000,
            attempts=attempts,
        )