"""Tests for analytical report mode: geo expand, intent, BQ plan, exports."""
from __future__ import annotations

from ml.rag.chatbot.analytical_bq_plan import build_analytical_bq_plan
from ml.rag.chatbot.analytical_intent import is_analytical_query
from ml.rag.chatbot.export_runner import sections_from_answer
from ml.rag.chatbot.geo_regions import (
    countries_for_regions,
    detect_regions_in_text,
    expand_regions_in_decomposition,
)


def test_detect_west_africa_region() -> None:
    assert "west africa" in detect_regions_in_text(
        "agricultural activities in west africa for the past 30yrs"
    )


def test_expand_west_africa_fills_countries() -> None:
    dec = {
        "intent": "compare",
        "entities": ["agricultural activities", "West Africa"],
        "geography": [],
        "time_start": "1990-01-01",
        "time_end": "2020-12-31",
    }
    out = expand_regions_in_decomposition(
        dec,
        "comparative analytics of all the countries in west africa",
    )
    geo = out.get("geography") or []
    assert "Nigeria" in geo
    assert "Ghana" in geo
    assert len(geo) >= 10
    assert "west africa" in (out.get("expanded_regions") or [])


def test_countries_for_ecowas() -> None:
    countries = countries_for_regions(["ecowas"])
    assert "Nigeria" in countries
    assert "Senegal" in countries


def test_analytical_intent_west_africa_report() -> None:
    q = (
        "create a pdf file of the detailed analytical report of agricultural "
        "activities in west africa for the past 30yrs, give me comparative "
        "analytics of all the countries and their major agricultural products"
    )
    assert is_analytical_query(q, {"intent": "compare", "geography": []})


def test_analytical_intent_rejects_simple_chat() -> None:
    assert not is_analytical_query("What is Ask ADZA?", {"intent": "descriptive"})


def test_analytical_bq_plan_never_skips() -> None:
    known = {"stg_faostat_production", "stg_faostat_trade"}
    plan = build_analytical_bq_plan(
        "West Africa agricultural production report",
        decomposition={
            "geography": ["Nigeria", "Ghana", "Senegal"],
            "time_start": "1990-01-01",
            "time_end": "2020-12-31",
            "entities": ["agriculture"],
        },
        known_tables=known,
    )
    assert plan is not None
    assert plan["skip_bq"] is False
    assert len(plan["query_intents"]) >= 3
    assert "stg_faostat_production" in plan["selected_tables"]


def test_sections_from_markdown_headings() -> None:
    answer = (
        "## Executive summary\nWest Africa grew slowly.\n\n"
        "## Country comparison\nNigeria leads in maize.[1]\n\n"
        "## Conclusion\nInvest in yields.\n"
    )
    sections = sections_from_answer("Q?", answer)
    headings = [s["heading"] for s in sections]
    assert "Executive summary" in headings
    assert "Country comparison" in headings
    assert "Question" in headings
    assert any("Nigeria" in s["body"] for s in sections)


def test_sections_fallback_without_headings() -> None:
    sections = sections_from_answer("My question", "Just a short answer.")
    assert sections[0]["heading"] == "Executive summary"
    assert sections[0]["body"] == "Just a short answer."
