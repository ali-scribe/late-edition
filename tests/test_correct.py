import pytest

import correct
from guard import validate_chunk


class _RateLimitError(Exception):
    code = 429


class _Response:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return _Response(result)


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


def _patch_client(monkeypatch, responses):
    client = _FakeClient(responses)
    monkeypatch.setattr(correct.genai, "Client", lambda **kwargs: client)
    return client


def test_429_is_retried_with_backoff_then_succeeds(monkeypatch):
    client = _patch_client(
        monkeypatch,
        [_RateLimitError("quota"), "The correct text."],
    )
    delays = []
    monkeypatch.setattr(correct.time, "sleep", delays.append)
    monkeypatch.setattr(correct.random, "uniform", lambda _low, _high: 0.1)
    monkeypatch.setattr(correct, "INITIAL_BACKOFF_SECONDS", 0.25)
    monkeypatch.setattr(correct, "BACKOFF_FACTOR", 2)

    result = correct.correct_chunk_with_gemini(
        "The corrcct text.",
        model_name="test-model",
        api_key="test-key",
        max_retries=2,
    )

    assert result == "The correct text."
    assert client.models.calls == 2
    assert delays == pytest.approx([0.35])


def test_retry_cap_returns_original_with_unavailable_status(monkeypatch):
    client = _patch_client(
        monkeypatch,
        [_RateLimitError("quota") for _ in range(3)],
    )
    monkeypatch.setattr(correct.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(correct.random, "uniform", lambda _low, _high: 0.1)
    original = "The original OCR text."

    result = correct.correct_chunk_with_gemini(
        original, model_name="test-model", api_key="test-key", max_retries=2
    )

    assert result == (original, "flagged: model unavailable")
    assert client.models.calls == 3


def test_repeated_chunk_uses_cache_without_another_api_call(monkeypatch):
    client = _patch_client(monkeypatch, ["The corrected text."])

    first = correct.correct_chunk_with_gemini(
        "The corrccted text.", model_name="test-model", api_key="test-key"
    )
    second = correct.correct_chunk_with_gemini(
        "The corrccted text.", model_name="test-model", api_key="test-key"
    )

    assert first == second == "The corrected text."
    assert client.models.calls == 1


def test_number_change_from_model_reply_is_reverted_by_guard(monkeypatch):
    _patch_client(monkeypatch, ["The notice is dated 11 June."])
    original = "The notice is dated 14 June."

    model_reply = correct.correct_chunk_with_gemini(
        original, model_name="test-model", api_key="test-key"
    )
    result = validate_chunk(original, model_reply)

    assert not result.accepted
    assert result.final_text == original
    assert any("digit_changed" in reason for reason in result.flag_reasons)
