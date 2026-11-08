"""Unit tests for plan_type tier gates."""

from __future__ import annotations

from ml.rag.chatbot.bq_sql_reasoner import _reasoner_model
from ml.rag.chatbot.plan_policy import (
    allows_cross_country,
    apply_plan_decomposition_gates,
    instruction_for_category,
    is_valid_category,
    is_valid_plan_type,
    model_for_plan,
    plan_generation_addendum,
)
from ml.rag.llm_chat import DEFAULT_LLM_MODEL_ID, llm_model_id


def test_valid_enums() -> None:
    assert is_valid_plan_type("Free")
    assert is_valid_plan_type("Integrated")
    assert is_valid_category("NGOs")
    assert not is_valid_plan_type("enterprise")
    assert not is_valid_category("Entrepreneurs")


def test_cross_country_only_agribusiness_and_integrated() -> None:
    assert not allows_cross_country("Free")
    assert not allows_cross_country("Government")
    assert allows_cross_country("Agribusinesses")
    assert allows_cross_country("Integrated")


def test_single_country_policy_off_allows_every_plan() -> None:
    from ml.rag.chatbot.plan_policy import plan_allows_cross_country

    assert plan_allows_cross_country("Farmers")
    assert plan_allows_cross_country("Free")
    assert plan_allows_cross_country("Government")


def test_export_only_agribusiness_and_integrated() -> None:
    from ml.rag.chatbot.plan_policy import allows_export

    assert allows_export("Agribusinesses")
    assert allows_export("Integrated")
    assert not allows_export("NGOs")


def test_decomposition_keeps_query_countries_when_policy_off() -> None:
    dec = {
        "intent": "compare",
        "geography": ["Nigeria", "Ghana"],
        "time_start": "2010",
        "time_end": "2024",
    }
    out = apply_plan_decomposition_gates(dec, "Government", "Kenya")
    assert out["geography"] == ["Nigeria", "Ghana"]
    assert out["intent"] == "compare"


def test_decomposition_clamps_to_query_country_when_policy_on(monkeypatch) -> None:
    monkeypatch.setenv("ADZA_SINGLE_COUNTRY_POLICY", "true")
    dec = {
        "intent": "compare",
        "geography": ["Nigeria", "Ghana"],
        "time_start": "2010",
        "time_end": "2024",
    }
    out = apply_plan_decomposition_gates(dec, "Government", "Kenya")
    assert out["geography"] == ["Nigeria"]
    assert out["intent"] == "descriptive"


def test_decomposition_allows_multi_country_for_agribusiness() -> None:
    dec = {"intent": "compare", "geography": ["Nigeria", "Ghana"]}
    out = apply_plan_decomposition_gates(dec, "Agribusinesses", None)
    assert out["geography"] == ["Nigeria", "Ghana"]
    assert out["intent"] == "compare"


def test_free_keeps_query_countries_when_policy_off() -> None:
    dec = {"intent": "compare", "geography": ["Nigeria", "Ghana"]}
    out = apply_plan_decomposition_gates(dec, "Free", "Ghana")
    assert out["geography"] == ["Nigeria", "Ghana"]
    assert out["intent"] == "descriptive"


def test_free_flag_on_prefers_query_country_over_profile(monkeypatch) -> None:
    monkeypatch.setenv("ADZA_SINGLE_COUNTRY_POLICY", "true")
    dec = {"intent": "compare", "geography": ["Nigeria", "Ghana"]}
    out = apply_plan_decomposition_gates(dec, "Free", "Ghana")
    assert out["geography"] == ["Nigeria"]
    assert out["intent"] == "descriptive"


def test_category_instruction_nonempty() -> None:
    assert "government" in instruction_for_category("Government").lower()
    assert "plain" in instruction_for_category("Farmers").lower()


def test_plan_addendum_free_is_brief() -> None:
    text = plan_generation_addendum("Free")
    assert "concise" in text.lower() or "top-line" in text.lower()


def test_integrated_plan_addendum_requires_evidence_linked_synthesis() -> None:
    text = plan_generation_addendum("Integrated").lower()
    assert "synthesize how relevant factors interact" in text
    assert "citing each factual link" in text
    assert "not present correlation" in text


def test_all_plan_models_default_to_8b(monkeypatch) -> None:
    for key in (
        "RAG_LLM_MODEL_FREE",
        "RAG_LLM_MODEL_FARMERS",
        "RAG_LLM_MODEL_GOVERNMENT",
        "RAG_LLM_MODEL_NGOS",
        "RAG_LLM_MODEL_AGRIBUSINESSES",
        "RAG_LLM_MODEL_INTEGRATED",
        "RAG_BQ_REASONER_MODEL_ID",
        "RAG_BQ_REASONER_FALLBACK_MODEL_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("RAG_LLM_MODEL_ID", raising=False)
    expected = DEFAULT_LLM_MODEL_ID
    assert llm_model_id() == expected
    for plan in ("Free", "Farmers", "Government", "NGOs", "Agribusinesses", "Integrated"):
        assert model_for_plan(plan) == expected
        assert "70b" not in (model_for_plan(plan) or "").lower()
        assert _reasoner_model(plan) == expected
