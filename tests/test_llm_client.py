"""Unit tests for src/llm_client.py (HTTP is mocked; no API key or network needed)."""
import json

import httpx
import pytest

from src.llm_client import (
    LLMAuthError,
    LLMBadResponseError,
    LLMClient,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMTokenLimitError,
    retry_hint_seconds,
)

MESSAGES = [{"role": "user", "content": "hi"}]


def ok_response(content="hello", finish="stop"):
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    )


def error_response(status, message="boom", headers=None):
    return httpx.Response(status, json={"error": {"message": message}}, headers=headers)


class Script:
    """Replays one scripted response (or exception) per request and records requests."""

    def __init__(self, *items):
        self.items = list(items)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(*items, **kwargs):
    script = Script(*items)
    sleeps = []
    client = LLMClient(
        api_key="k-test",
        base_url="https://example.test/v1/",
        model="test-model",
        transport=httpx.MockTransport(script),
        sleep=sleeps.append,
        jitter=0.0,
        **kwargs,
    )
    return client, script, sleeps


class TestSuccess:
    def test_success_parses_result(self):
        client, script, sleeps = make_client(ok_response("hello"))
        result = client.chat(MESSAGES, max_tokens=123, temperature=0.2)
        assert result.text == "hello"
        assert result.model == "test-model"
        assert result.finish_reason == "stop"
        assert (result.prompt_tokens, result.completion_tokens) == (10, 5)
        assert result.reasoning_tokens == 2
        assert result.total_tokens == 15
        assert result.attempts == 1
        assert result.retry_wait_ms == 0
        assert result.latency_ms >= 0
        assert sleeps == []
        request = script.requests[0]
        assert str(request.url) == "https://example.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer k-test"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 123
        assert body["temperature"] == 0.2
        assert body["messages"] == MESSAGES

    def test_reasoning_effort_only_sent_when_given(self):
        client, script, _ = make_client(ok_response(), ok_response())
        client.chat(MESSAGES)
        client.chat(MESSAGES, reasoning_effort="low")
        assert "reasoning_effort" not in json.loads(script.requests[0].content)
        assert json.loads(script.requests[1].content)["reasoning_effort"] == "low"

    def test_content_is_stripped(self):
        client, _, _ = make_client(ok_response("  hello \n"))
        assert client.chat(MESSAGES).text == "hello"


class TestRetries:
    def test_429_uses_retry_after_header(self):
        client, script, sleeps = make_client(
            error_response(429, "slow down", headers={"retry-after": "2"}), ok_response()
        )
        result = client.chat(MESSAGES)
        assert sleeps == [2.0]
        assert result.attempts == 2
        assert result.retry_wait_ms == 2000.0

    def test_429_uses_hint_from_message(self):
        msg = (
            "Rate limit reached on tokens per minute (TPM): Limit 8000, Used 7500, "
            "Requested 900. Please try again in 1m26.4s."
        )
        client, _, sleeps = make_client(
            error_response(429, msg), ok_response(), max_retry_wait=120.0
        )
        client.chat(MESSAGES)
        assert sleeps == [pytest.approx(86.4)]

    def test_429_without_hint_backs_off_exponentially(self):
        client, _, sleeps = make_client(
            error_response(429), error_response(429), error_response(429), ok_response()
        )
        result = client.chat(MESSAGES)
        assert sleeps == [1.0, 2.0, 4.0]
        assert result.attempts == 4

    def test_persistent_429_raises_after_retries(self):
        client, script, sleeps = make_client(
            *[error_response(429, "slow down") for _ in range(3)], max_retries=2
        )
        with pytest.raises(LLMRateLimitError):
            client.chat(MESSAGES)
        assert len(script.requests) == 3
        assert len(sleeps) == 2

    def test_very_long_retry_after_fails_fast(self):
        client, script, sleeps = make_client(
            error_response(429, "daily quota", headers={"retry-after": "3600"})
        )
        with pytest.raises(LLMRateLimitError) as exc:
            client.chat(MESSAGES)
        assert exc.value.retry_after == 3600.0
        assert len(script.requests) == 1
        assert sleeps == []

    def test_5xx_is_retried(self):
        client, _, sleeps = make_client(error_response(503, "overloaded"), ok_response())
        result = client.chat(MESSAGES)
        assert result.attempts == 2
        assert len(sleeps) == 1

    def test_persistent_5xx_raises(self):
        client, _, _ = make_client(
            error_response(500, "x"), error_response(502, "y"), max_retries=1
        )
        with pytest.raises(LLMServerError):
            client.chat(MESSAGES)

    def test_timeout_is_retried(self):
        client, _, sleeps = make_client(httpx.ReadTimeout("timed out"), ok_response())
        result = client.chat(MESSAGES)
        assert result.attempts == 2
        assert sleeps == [1.0]

    def test_persistent_timeout_raises(self):
        client, _, _ = make_client(
            httpx.ReadTimeout("timed out"), httpx.ReadTimeout("timed out"), max_retries=1
        )
        with pytest.raises(LLMTimeoutError):
            client.chat(MESSAGES)

    def test_connection_error_raises_after_retries(self):
        client, _, _ = make_client(
            httpx.ConnectError("refused"), httpx.ConnectError("refused"), max_retries=1
        )
        with pytest.raises(LLMConnectionError):
            client.chat(MESSAGES)


class TestNonRetryableErrors:
    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_errors_not_retried(self, status):
        client, script, sleeps = make_client(error_response(status, "Invalid API key"))
        with pytest.raises(LLMAuthError):
            client.chat(MESSAGES)
        assert len(script.requests) == 1
        assert sleeps == []

    def test_model_not_found_not_retried(self):
        client, script, _ = make_client(error_response(404, "The model does not exist"))
        with pytest.raises(LLMModelNotFoundError) as exc:
            client.chat(MESSAGES)
        assert "test-model" in str(exc.value)
        assert len(script.requests) == 1

    @pytest.mark.parametrize(
        "status, message",
        [
            (413, "Request too large"),
            (400, "This model's maximum context length is 8192 tokens"),
            (
                429,
                "Request too large for model on tokens per minute. "
                "Please reduce your message size and try again.",
            ),
        ],
    )
    def test_token_limit_errors_not_retried(self, status, message):
        client, script, sleeps = make_client(error_response(status, message))
        with pytest.raises(LLMTokenLimitError):
            client.chat(MESSAGES)
        assert len(script.requests) == 1
        assert sleeps == []

    def test_other_400_is_generic_error(self):
        client, script, _ = make_client(error_response(400, "invalid parameter foo"))
        with pytest.raises(LLMError) as exc:
            client.chat(MESSAGES)
        assert type(exc.value) is LLMError
        assert len(script.requests) == 1


class TestBadResponses:
    def test_empty_content_with_length_is_token_limit(self):
        client, script, _ = make_client(ok_response("", finish="length"))
        with pytest.raises(LLMTokenLimitError):
            client.chat(MESSAGES)
        assert len(script.requests) == 1

    def test_empty_content_with_stop_is_bad_response(self):
        client, _, _ = make_client(ok_response("", finish="stop"))
        with pytest.raises(LLMBadResponseError):
            client.chat(MESSAGES)

    def test_null_content_is_bad_response(self):
        body = {
            "choices": [
                {"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}
            ]
        }
        client, _, _ = make_client(httpx.Response(200, json=body))
        with pytest.raises(LLMBadResponseError):
            client.chat(MESSAGES)

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, text="not json"),
            httpx.Response(200, json={"choices": []}),
            httpx.Response(200, json={"foo": 1}),
            httpx.Response(200, json=[1, 2]),
        ],
    )
    def test_malformed_body_is_bad_response(self, response):
        client, _, _ = make_client(response)
        with pytest.raises(LLMBadResponseError):
            client.chat(MESSAGES)


class TestConfigAndArguments:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"api_key": ""},
            {"api_key": "paste_your_key_here"},
            {"api_key": "your_api_key_here"},
            {"base_url": ""},
            {"model": ""},
        ],
    )
    def test_bad_config_raises(self, overrides):
        kwargs = {"api_key": "k", "base_url": "https://x.test/v1", "model": "m", **overrides}
        with pytest.raises(LLMConfigError):
            LLMClient(**kwargs)

    def test_from_env_reads_mapping(self):
        env = {"LLM_API_KEY": "k", "LLM_BASE_URL": "https://x.test/v1/", "LLM_MODEL": "m"}
        with LLMClient.from_env(env) as client:
            assert client.model == "m"
            assert client.base_url == "https://x.test/v1"

    def test_from_env_missing_values_raises(self):
        with pytest.raises(LLMConfigError):
            LLMClient.from_env({})

    @pytest.mark.parametrize("messages, max_tokens", [([], 100), (MESSAGES, 0)])
    def test_invalid_arguments_raise(self, messages, max_tokens):
        client, script, _ = make_client()
        with pytest.raises(ValueError):
            client.chat(messages, max_tokens=max_tokens)
        assert script.requests == []


class TestRetryHint:
    @pytest.mark.parametrize(
        "message, expected",
        [
            ("Please try again in 1m26.4s.", 86.4),
            ("try again in 250ms", 0.25),
            ("Try again in 2.5s", 2.5),
            ("try again in 7m", 420.0),
            ("try again in a bit", None),
            ("no hint here", None),
            ("", None),
        ],
    )
    def test_retry_hint_seconds(self, message, expected):
        result = retry_hint_seconds(message)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)