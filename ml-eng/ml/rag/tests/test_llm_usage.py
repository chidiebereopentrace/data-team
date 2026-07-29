"""Unit tests for per-request LLM token usage accumulation."""

from __future__ import annotations

from ml.rag.llm_chat import add_llm_usage, get_llm_usage, reset_llm_usage


def test_add_llm_usage_sums_across_calls() -> None:
    reset_llm_usage()
    add_llm_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    add_llm_usage({"input_tokens": 20, "output_tokens": 10})
    usage = get_llm_usage()
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 60
    assert usage.total_tokens == 180
    d = usage.to_dict()
    assert d["input_tokens"] == 120
    assert d["output_tokens"] == 60


def test_reset_llm_usage_clears_totals() -> None:
    add_llm_usage({"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700})
    reset_llm_usage()
    usage = get_llm_usage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_add_llm_usage_ignores_empty() -> None:
    reset_llm_usage()
    add_llm_usage(None)
    add_llm_usage({})
    assert get_llm_usage().total_tokens == 0
