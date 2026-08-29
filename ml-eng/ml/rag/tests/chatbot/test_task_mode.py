"""Tests for unified task_mode router."""
from __future__ import annotations

from ml.rag.chatbot.fact_bq_plan import build_fact_bq_plan
from ml.rag.chatbot.task_mode import (
    is_briefing_query,
    is_data_export_only_query,
    is_fact_lookup_query,
    needs_clarify,
    resolve_task_mode,
)


def test_clarify_missing_geo_and_crop() -> None:
    assert needs_clarify("What was production last year?", {"intent": "descriptive", "geography": []})
    assert resolve_task_mode("maize yields?", {"entities": ["maize"], "geography": []}) == "clarify"


def test_clarify_not_when_country_and_crop() -> None:
    dec = {"entities": ["maize"], "geography": ["Nigeria"], "intent": "descriptive"}
    assert not needs_clarify("What was maize production in Nigeria in 2020?", dec)
    assert resolve_task_mode("What was maize production in Nigeria in 2020?", dec) == "fact_lookup"


def test_profile_country_avoids_clarify() -> None:
    dec = {"entities": ["maize"], "geography": [], "intent": "descriptive"}
    assert (
        resolve_task_mode("maize production trends", dec, profile_country="Ghana")
        == "fact_lookup"
    )


def test_analytical_beats_export_and_fact() -> None:
    q = (
        "create a pdf file of the detailed analytical report of agricultural "
        "activities in west africa for the past 30yrs, give me comparative "
        "analytics of all the countries and their major agricultural products"
    )
    assert resolve_task_mode(q, {"intent": "compare", "geography": []}) == "analytical"


def test_data_export_only_csv() -> None:
    q = "Just give me the data as CSV for maize production in Nigeria 2015-2020, no essay"
    dec = {"entities": ["maize"], "geography": ["Nigeria"], "intent": "descriptive"}
    assert is_data_export_only_query(q, dec)
    assert resolve_task_mode(q, dec) == "data_export_only"


def test_fact_lookup_vs_analytical() -> None:
    fact_q = "What was rice production in Senegal in 2019?"
    fact_dec = {"entities": ["rice"], "geography": ["Senegal"], "intent": "descriptive"}
    assert is_fact_lookup_query(fact_q, fact_dec)
    assert resolve_task_mode(fact_q, fact_dec) == "fact_lookup"
    assert resolve_task_mode(
        "comparative analytics of all countries in west africa over 30 years",
        {"intent": "compare", "geography": []},
    ) == "analytical"


def test_briefing_mode() -> None:
    q = "Brief me on the latest maize market headlines in Kenya this week"
    dec = {"entities": ["maize"], "geography": ["Kenya"], "intent": "monitoring"}
    assert is_briefing_query(q, dec)
    assert resolve_task_mode(q, dec) == "briefing"


def test_chat_default() -> None:
    assert resolve_task_mode("How do cooperatives help smallholders?", {"intent": "descriptive"}) == "chat"


def test_ranking_africa_default_is_fact_not_clarify() -> None:
    q = "Which African country produces the most maize?"
    dec = {
        "entities": ["maize"],
        "geography": [],
        "africa_default": True,
        "intent": "descriptive",
    }
    assert resolve_task_mode(q, dec) == "fact_lookup"


def test_fact_bq_plan_forced() -> None:
    known = {"fct_production"}
    plan = build_fact_bq_plan(
        "maize production Nigeria 2020",
        decomposition={"geography": ["Nigeria"], "entities": ["maize"], "time_end": "2020-12-31"},
        known_tables=known,
        task_mode="fact_lookup",
    )
    assert plan is not None
    assert plan["skip_bq"] is False
    assert len(plan["query_intents"]) >= 1

    export_plan = build_fact_bq_plan(
        "csv of maize Nigeria",
        decomposition={"geography": ["Nigeria"], "entities": ["maize"]},
        known_tables=known,
        task_mode="data_export_only",
    )
    assert export_plan is not None
    assert len(export_plan["query_intents"]) >= 2
