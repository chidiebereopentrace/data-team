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
        "COHERE_API_KEY",
        "RAG_RERANKER_COHERE_API_KEY",
        "RAG_RERANKER_COHERE_MODEL",
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


def test_reranker_mode_explicit_cohere(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RERANKER_MODE", "cohere")
    assert R._reranker_mode() == "cohere"


def test_reranker_mode_auto_promotes_to_cohere_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-key-123")
    assert R._reranker_mode() == "cohere"


def test_reranker_mode_explicit_overrides_cohere_auto_promote(monkeypatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-key-123")
    monkeypatch.setenv("RAG_RERANKER_MODE", "cross_encoder")
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


# --- Cohere backend ----------------------------------------------------------


def test_cohere_mode_returns_results_sorted_by_relevance(monkeypatch) -> None:
    """Cohere scores are already [0,1]; the highest relevance chunk must rank first."""
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_RERANKER_MODE", "cohere")

    # _sample_items() order: [news=0, academic=1, bigquery=2]
    # Cohere returns: index=1 (academic) → 0.9, index=0 (news) → 0.1, index=2 (bigquery) → 0.5
    # After source boost:
    #   academic (idx=1): 0.9 + 0.06 = 0.96  → #1
    #   bigquery (idx=2): 0.5 + 0.12 = 0.62  → #2
    #   news     (idx=0): 0.1 + 0.04 = 0.14  → #3
    fake_hit_0 = mock.Mock(index=1, relevance_score=0.9)   # academic
    fake_hit_1 = mock.Mock(index=0, relevance_score=0.1)   # news
    fake_hit_2 = mock.Mock(index=2, relevance_score=0.5)   # bigquery
    fake_response = mock.Mock(results=[fake_hit_0, fake_hit_1, fake_hit_2])

    fake_co = mock.Mock()
    fake_co.rerank.return_value = fake_response
    fake_cohere_module = mock.Mock(ClientV2=mock.Mock(return_value=fake_co))

    with mock.patch.dict("sys.modules", {"cohere": fake_cohere_module}):
        out = R.rerank("q", _sample_items(), top_k=3)

    assert len(out) == 3
    assert out[0]["_context_kind"] == "academic"  # relevance 0.9+boost 0.06=0.96
    assert out[1]["_context_kind"] == "bigquery"  # relevance 0.5+boost 0.12=0.62
    assert out[2]["_context_kind"] == "news"      # relevance 0.1+boost 0.04=0.14
    for entry in out:
        assert "_cohere_score" in entry
        assert "_rerank_score" in entry


def test_cohere_mode_no_key_degrades_to_cross_encoder(monkeypatch) -> None:
    """No COHERE_API_KEY → _rerank_cohere returns None → falls through to cross_encoder."""
    monkeypatch.setenv("RAG_RERANKER_MODE", "cohere")
    # No COHERE_API_KEY set (cleared by autouse fixture).
    fake_model = mock.Mock()
    fake_model.predict.return_value = [0.5, 0.9, 0.1]
    monkeypatch.setattr(R, "_load_cross_encoder", lambda mid: ("sentence_transformers", fake_model))

    out = R.rerank("q", _sample_items(), top_k=3)
    assert len(out) == 3
    assert all("_ce_score" in entry for entry in out)  # cross_encoder ran


def test_cohere_mode_api_failure_degrades_to_cross_encoder(monkeypatch) -> None:
    """Cohere API error → degrade to cross_encoder, not crash."""
    monkeypatch.setenv("RAG_RERANKER_MODE", "cohere")
    monkeypatch.setenv("COHERE_API_KEY", "test-key")

    fake_co = mock.Mock()
    fake_co.rerank.side_effect = RuntimeError("API unreachable")
    fake_cohere_module = mock.Mock(ClientV2=mock.Mock(return_value=fake_co))

    fake_model = mock.Mock()
    fake_model.predict.return_value = [0.5, 0.9, 0.1]
    monkeypatch.setattr(R, "_load_cross_encoder", lambda mid: ("sentence_transformers", fake_model))

    with mock.patch.dict("sys.modules", {"cohere": fake_cohere_module}):
        out = R.rerank("q", _sample_items(), top_k=3)

    assert len(out) == 3
    assert all("_ce_score" in entry for entry in out)  # cross_encoder ran as fallback


# --- empty-input safety -------------------------------------------------------


def test_empty_context_returns_empty() -> None:
    assert R.rerank("q", [], top_k=5) == []
