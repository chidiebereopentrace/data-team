"""Tests for reasoning-model handling in llm_chat_complete (Claude 5.5 on OpenRouter).

Claude 5.5 thinks by default and bills thinking tokens against max_tokens, so a
budget sized for a non-reasoning model can come back with empty content and
finish_reason="length". These tests lock in the fix: a reasoning payload with
headroom tokens, no temperature field, and a diagnostic log instead of a silent
empty return.
"""
from __future__ import annotations

from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "HF_API_TOKEN",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "RAG_LLM_REASONING_EFFORT",
        "RAG_LLM_REASONING_EFFORT_GENERATE",
        "RAG_LLM_REASONING_HEADROOM_TOKENS",
        "RAG_LLM_REASONING_MODELS",
        "RAG_LLM_NO_TEMPERATURE_MODELS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-key")


def _fake_post_capturing(captured: dict, content: str = "ok", finish_reason: str = "stop"):
    def fake_post(url: str, **kwargs: object) -> mock.Mock:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        resp.raise_for_status = mock.Mock()
        return resp

    return fake_post


def test_reasoning_model_gets_reasoning_payload_and_headroom() -> None:
    captured: dict = {}
    with mock.patch("ml.rag.llm_chat.requests.post", side_effect=_fake_post_capturing(captured)):
        from ml.rag.llm_chat import llm_chat_complete

        out = llm_chat_complete(
            [{"role": "user", "content": "hi"}],
            model="anthropic/claude-haiku-5.5",
            max_tokens=512,
            purpose="generate",
        )

    assert out == "ok"
    payload = captured["json"]
    assert payload["reasoning"] == {"effort": "low", "exclude": True}
    assert payload["max_tokens"] == 512 + 2048
    assert "temperature" not in payload


def test_reasoning_model_respects_per_purpose_effort_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_REASONING_EFFORT_GENERATE", "none")
    captured: dict = {}
    with mock.patch("ml.rag.llm_chat.requests.post", side_effect=_fake_post_capturing(captured)):
        from ml.rag.llm_chat import llm_chat_complete

        llm_chat_complete(
            [{"role": "user", "content": "hi"}],
            model="anthropic/claude-haiku-5.5",
            max_tokens=512,
            purpose="generate",
        )

    assert captured["json"]["reasoning"] == {"enabled": False}


def test_non_reasoning_model_payload_unchanged() -> None:
    captured: dict = {}
    with mock.patch("ml.rag.llm_chat.requests.post", side_effect=_fake_post_capturing(captured)):
        from ml.rag.llm_chat import llm_chat_complete

        llm_chat_complete(
            [{"role": "user", "content": "hi"}],
            model="qwen/qwen3-30b-a3b-instruct-2507",
            max_tokens=512,
            temperature=0.7,
        )

    payload = captured["json"]
    assert "reasoning" not in payload
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 512


def test_empty_content_with_length_finish_reason_logs_and_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    captured: dict = {}
    fake_post = _fake_post_capturing(captured, content="", finish_reason="length")
    with mock.patch("ml.rag.llm_chat.requests.post", side_effect=fake_post):
        from ml.rag.llm_chat import answer_was_length_capped, llm_chat_complete

        with caplog.at_level("WARNING"):
            out = llm_chat_complete(
                [{"role": "user", "content": "hi"}],
                model="anthropic/claude-haiku-5.5",
                max_tokens=512,
                purpose="generate",
            )

    assert out == ""
    assert answer_was_length_capped() is True
    assert any("empty content" in rec.message for rec in caplog.records)


def test_model_is_reasoning_and_rejects_temperature_detection() -> None:
    from ml.rag.llm_chat import model_is_reasoning, model_rejects_temperature

    assert model_is_reasoning("anthropic/claude-haiku-5.5") is True
    assert model_rejects_temperature("anthropic/claude-haiku-5.5") is True
    assert model_is_reasoning("qwen/qwen3-30b-a3b-instruct-2507") is False
    assert model_rejects_temperature("qwen/qwen3-30b-a3b-instruct-2507") is False


def test_reasoning_models_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_REASONING_MODELS", "some/custom-model")
    from ml.rag.llm_chat import model_is_reasoning

    assert model_is_reasoning("some/custom-model") is True
