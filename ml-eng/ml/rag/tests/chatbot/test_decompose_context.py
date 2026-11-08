"""Tests for decompose context and prompt assembly."""
from __future__ import annotations

from ml.rag.chatbot.agri_measure_ontology import decompose_measure_vocabulary
from ml.rag.chatbot.decompose_context import (
    DecomposeContext,
    format_decompose_system_prompt,
    format_decompose_user_prompt,
)
from ml.rag.chatbot.geo_regions import decompose_region_vocabulary
from ml.rag.chatbot.query_decomposer import build_provisional_decompose_hints


def test_decompose_region_vocabulary_bounded() -> None:
    text = decompose_region_vocabulary(max_zones=24)
    assert "west africa" in text.lower()
    assert text.count("\n") <= 24


def test_decompose_measure_vocabulary_non_empty() -> None:
    text = decompose_measure_vocabulary(max_measures=20)
    assert "production" in text
    assert text.count("\n") <= 20


def test_format_user_prompt_includes_plan_and_overrides() -> None:
    ctx = DecomposeContext(
        query="maize in Kenya",
        original_query="maize in Kenya",
        plan_type="Farmers",
        category="Farmers",
        profile_country="Nigeria",
        geo_override="Kenya",
        time_start_override="2020-01-01",
        time_end_override="2020-12-31",
        conversation_summary="User asked about maize trends.",
    )
    prov = build_provisional_decompose_hints(ctx.query)
    user = format_decompose_user_prompt(ctx, prov)
    assert "Plan tier: Farmers" in user
    assert "climate × nutrition" not in user
    assert "Plan tier: NGOs / development" not in user
    assert "Profile country" in user
    assert "geo=Kenya" in user
    assert "Conversation summary" in user
    assert "Provisional heuristic hints" in user
    assert "Kenya" in user


def test_format_system_prompt_includes_schema_keys() -> None:
    system = format_decompose_system_prompt()
    assert "primary_measure_hints" in system
    assert "must not invent measures" in system
    assert "geo_scope" in system
    assert "job must be" in system


def test_provisional_hints_extract_countries_and_years() -> None:
    prov = build_provisional_decompose_hints("maize production in Kenya 2020")
    assert prov["countries"] == ["Kenya"]
    assert prov["time_start"] == "2020-01-01"
    assert prov["time_end"] == "2020-12-31"


def test_from_graph_state_reads_profile_and_overrides() -> None:
    state = {
        "query": "and in Ghana?",
        "plan_type": "Government",
        "category": "Government",
        "geo_override": "Ghana",
        "time_start_override": "2019-01-01",
        "user_profile": {"country": "Kenya", "category": "Government"},
        "conversation_summary": "Prior turn about maize.",
    }
    enrich = {
        "user_query": "and in Ghana?",
        "context_rewrite": "and in Ghana? (carried from earlier turns: commodity: maize)",
        "context_applied": True,
    }
    ctx = DecomposeContext.from_graph_state(state, enrich)
    assert ctx.plan_type == "Government"
    assert ctx.profile_country == "Kenya"
    assert ctx.geo_override == "Ghana"
    assert ctx.query == "and in Ghana?"
    assert ctx.context_applied is True
    assert "carried from earlier turns" in ctx.context_rewrite
