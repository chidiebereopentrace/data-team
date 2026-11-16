"""Tests for model-first query decompose."""
from __future__ import annotations

from datetime import date
from unittest import mock

from ml.rag.chatbot.decompose_context import DecomposeContext
from ml.rag.chatbot.query_decomposer import (
    _call_llama_decompose,
    decompose_query,
    should_use_llm_decompose,
)


def test_price_question_preserves_original_this_year_over_english_rewrite() -> None:
    context = DecomposeContext(
        query="What is the current year's price of maize in Kenya?",
        original_query="whats the price of maize in kenya this year",
    )
    out = decompose_query(context.query, context=context, use_llm=False, reference_date=date(2026, 11, 16))
    assert out["time_start"] == "2026-01-01"
    assert out["time_end"] == "2026-11-16"
    assert out["time_phrase"] == "this year"


def test_should_skip_llm_legacy_gate_unchanged() -> None:
    assert should_use_llm_decompose("maize production in Kenya 2020") is False
    assert should_use_llm_decompose("compare maize yields Kenya vs Nigeria 2020") is True


def test_always_calls_llm_when_backend_configured() -> None:
    q = "maize production in Kenya 2020"
    with mock.patch(
        "ml.rag.chatbot.query_decomposer._decompose_backend_configured",
        return_value=True,
    ):
        with mock.patch(
            "ml.rag.chatbot.query_decomposer._call_llama_decompose",
            return_value={
                "intent": "descriptive",
                "job": "fact",
                "entities": ["maize"],
                "geography": ["Kenya"],
                "domains": ["production"],
                "geo_scope": "country",
                "primary_measure_hints": ["production"],
                "time_start": "2020-01-01",
                "time_end": "2020-12-31",
            },
        ) as call:
            out = decompose_query(q)
    call.assert_called_once()
    assert out["_decompose_llm_used"] is True
    assert out["_legacy_would_skip_llm"] is True
    assert out["_skipped_decompose_llm"] is False
    assert out["job"] == "fact"
    assert out["geo_scope"] == "country"
    assert out["primary_measure_hints"] == ["production"]
    assert out["geography"] == ["Kenya"]


def test_entity_alias_normalization_paddy_to_rice() -> None:
    q = "paddy production in Senegal 2019"
    fake_llm = {
        "intent": "descriptive",
        "job": "fact",
        "entities": ["rice"],
        "geography": ["Senegal"],
        "domains": ["production"],
        "time_start": "2019-01-01",
        "time_end": "2019-12-31",
    }
    with mock.patch(
        "ml.rag.chatbot.query_decomposer._decompose_backend_configured",
        return_value=True,
    ):
        with mock.patch(
            "ml.rag.chatbot.query_decomposer._call_llama_decompose",
            return_value=fake_llm,
        ):
            out = decompose_query(q)
    assert "rice" in [e.lower() for e in out.get("entities") or []]


def test_geo_hallucination_still_dropped() -> None:
    q = "kedu obodo kacha ako ji na mba africa"
    fake_llm = {
        "intent": "locate",
        "entities": ["Nigeria", "Africa"],
        "geography": ["Nigeria"],
        "domains": ["agriculture"],
        "time_start": "",
        "time_end": "",
    }
    with mock.patch(
        "ml.rag.chatbot.query_decomposer._decompose_backend_configured",
        return_value=True,
    ):
        with mock.patch(
            "ml.rag.chatbot.query_decomposer._call_llama_decompose",
            return_value=fake_llm,
        ):
            out = decompose_query(q)
    geo_l = [g.lower() for g in out.get("geography") or []]
    ent_l = [e.lower() for e in out.get("entities") or []]
    assert "nigeria" not in geo_l
    assert "nigeria" not in ent_l


def test_heuristic_time_wins_since_till_now() -> None:
    q = "whats the trend of rice production in nigeria since 2015 till now"
    fake_llm = {
        "intent": "descriptive",
        "entities": ["rice"],
        "geography": ["Nigeria"],
        "time_start": "2015-01-01",
        "time_end": "2015-12-31",
    }
    with mock.patch(
        "ml.rag.chatbot.query_decomposer._decompose_backend_configured",
        return_value=True,
    ):
        with mock.patch(
            "ml.rag.chatbot.query_decomposer._call_llama_decompose",
            return_value=fake_llm,
        ):
            out = decompose_query(q, reference_date=date(2026, 10, 14))
    assert out["time_start"] == "2015-01-01"
    assert out["time_end"] == "2026-10-14"


def test_call_llama_uses_system_user_messages() -> None:
    with mock.patch(
        "ml.rag.chatbot.query_decomposer._decompose_backend_configured",
        return_value=True,
    ):
        with mock.patch(
            "ml.rag.chatbot.query_decomposer.llm_chat_complete",
            return_value='{"intent":"descriptive","entities":[],"geography":[],"domains":[]}',
        ) as llm:
            _call_llama_decompose("maize in Kenya")
    messages = llm.call_args[0][0]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert "OpenTrace" in messages[0]["content"]
    assert "maize in Kenya" in messages[1]["content"]


def test_no_backend_skips_llm_and_uses_heuristics() -> None:
    with mock.patch(
        "ml.rag.chatbot.query_decomposer._decompose_backend_configured",
        return_value=False,
    ):
        with mock.patch(
            "ml.rag.chatbot.query_decomposer._call_llama_decompose",
        ) as call:
            out = decompose_query("maize production in Kenya 2020")
    call.assert_not_called()
    assert out["_skipped_decompose_llm"] is True
    assert out["_decompose_llm_used"] is False
    assert out["geography"] == ["Kenya"]


def test_retrieval_contract_uses_primary_measure_hints() -> None:
    from ml.rag.chatbot.retrieval_contract import build_corpus_routing_contract

    dec = {
        "entities": ["maize"],
        "geography": ["Kenya"],
        "primary_measure_hints": ["production"],
    }
    contract = build_corpus_routing_contract("maize output in Kenya", decomposition=dec)
    assert contract.primary_measures == ["production"]

