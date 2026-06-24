"""Unit tests for the reranker (cross_encoder / llm / off) and its degradation
paths. Backend-agnostic: anything that would touch a model or the network is
mocked, so the suite runs offline and deterministically."""

from __future__ import annotations

from unittest import mock

import pytest

from ml.rag.chatbot import reranker as R


@pytest.fixture(autouse=True)
def _clear_reranker_env(monkeypatch):
    """Each test starts with a known, clean reranker config."""
    for key in (
        "RAG_RERANKER_MODE",
        "RAG_RERANKER_MODEL",
        "RAG_RERANKER_TOP_K",
        "RAG_RERANKER_MAX_TEXT_CHARS",
        "RAG_LLM_RERANK",
        "RAG_LLM_BASE_URL",
        "HF_API_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    R._ce_cache.clear()
    yield


def _sample_items() -> list[dict]:
    return [
        {
            "content": "Senegal rice production rose 12% in 2023 according to ANSD.",
            "_context_kind": "news",
            "metadata": {"title": "Senegal rice"},
        },
        {
            "content": "Long historical overview of West African staple crops.",
            "_context_kind": "academic",
            "metadata": {"authors": "X", "publication_year": "2005"},
        },
        {
            "content": "{country: 'Senegal', metric: 'rice_production_tonnes'}",
            "_context_kind": "bigquery",
            "metadata": {"sql": "SELECT 1"},
        },
    ]


# --- mode resolution ---------------------------------------------------------


def test_reranker_mode_default_is_cross_encoder() -> None:
    assert R._reranker_mode() == "cross_encoder"


def test_reranker_mode_explicit_wins(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RERANKER_MODE", "llm")
    monkeypatch.setenv("RAG_LLM_RERANK", "off")
    assert R._reranker_mode() == "llm"


def test_reranker_mode_legacy_on_maps_to_llm(monkeypatch) -> None:
    monkeypatch.setenv("RAG_LLM_RERANK", "on")
    assert R._reranker_mode() == "llm"


def test_reranker_mode_legacy_off_maps_to_off(monkeypatch) -> None:
    monkeypatch.setenv("RAG_LLM_RERANK", "off")
    assert R._reranker_mode() == "off"


def test_reranker_mode_invalid_explicit_falls_through_to_default(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RERANKER_MODE", "totally_invalid")
    assert R._reranker_mode() == "cross_encoder"


# --- cross_encoder happy path ------------------------------------------------


def test_cross_encoder_reorders_to_query_relevant_chunk(monkeypatch) -> None:
    """The cross-encoder must lift the most-relevant chunk above the others,
    even when the static source boost would otherwise prefer a different one."""
    items = _sample_items()
    fake_model = mock.Mock()
    fake_model.predict.return_value = [0.1, 0.95, 0.2]  # news, academic, bq
    monkeypatch.setattr(
        R, "_load_cross_encoder",
        lambda mid: ("sentence_transformers", fake_model),
    )

    out = R.rerank("rice production senegal", items, top_k=3)

    assert len(out) == 3
    assert out[0]["_context_kind"] == "academic"
    for entry in out:
        assert "_ce_score" in entry
        assert "_ce_score_raw" in entry
        assert "_source_boost" in entry
        assert "_rerank_score" in entry


def test_cross_encoder_normalises_scores_into_unit_range(monkeypatch) -> None:
    items = _sample_items()
    fake_model = mock.Mock()
    fake_model.predict.return_value = [-8.0, 5.0, -2.0]
    monkeypatch.setattr(
        R, "_load_cross_encoder",
        lambda mid: ("sentence_transformers", fake_model),
    )

    out = R.rerank("q", items, top_k=3)
    for entry in out:
        assert 0.0 <= entry["_ce_score"] <= 1.0


# --- cross_encoder degradation -----------------------------------------------


def test_cross_encoder_unavailable_falls_back_to_llm_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setattr(R, "_load_cross_encoder", lambda mid: None)
    monkeypatch.setattr(R, "_score_with_llama", lambda q, t: 0.5)

    out = R.rerank("q", _sample_items(), top_k=3)
    assert len(out) == 3
    assert all("_llm_score" in entry for entry in out)
    assert all("_ce_score" not in entry for entry in out)


def test_cross_encoder_unavailable_without_llm_passes_through(monkeypatch) -> None:
    monkeypatch.setattr(R, "_load_cross_encoder", lambda mid: None)
    out = R.rerank("q", _sample_items(), top_k=3)
    assert len(out) == 3
    assert out[0]["_context_kind"] == "bigquery"
    assert all(entry["_llm_score"] == -1.0 for entry in out)


# --- llm mode ----------------------------------------------------------------


def test_llm_mode_without_backend_degrades_to_off(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RERANKER_MODE", "llm")
    out = R.rerank("q", _sample_items(), top_k=3)
    assert len(out) == 3
    assert out[0]["_context_kind"] == "bigquery"


def test_llm_mode_scores_each_chunk(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RERANKER_MODE", "llm")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setattr(R, "_score_with_llama", lambda q, t: 0.42)

    out = R.rerank("q", _sample_items(), top_k=3)
    assert all(entry["_llm_score"] == 0.42 for entry in out)


# --- off mode ----------------------------------------------------------------


def test_off_mode_applies_source_boost_only(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RERANKER_MODE", "off")
    out = R.rerank("q", _sample_items(), top_k=3)
    assert [x["_context_kind"] for x in out] == ["bigquery", "academic", "news"]


# --- top_k clamping ----------------------------------------------------------


def test_env_top_k_clamps_below_caller_value(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RERANKER_MODE", "off")
    monkeypatch.setenv("RAG_RERANKER_TOP_K", "1")
    out = R.rerank("q", _sample_items(), top_k=5)
    assert len(out) == 1


def test_env_top_k_zero_means_use_caller_value(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RERANKER_MODE", "off")
    monkeypatch.setenv("RAG_RERANKER_TOP_K", "0")
    out = R.rerank("q", _sample_items(), top_k=2)
    assert len(out) == 2


# --- empty-input safety -------------------------------------------------------


def test_empty_context_returns_empty() -> None:
    assert R.rerank("q", [], top_k=5) == []
