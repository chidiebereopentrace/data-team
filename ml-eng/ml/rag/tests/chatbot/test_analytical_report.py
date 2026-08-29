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
    known = {"fct_production", "fct_trade"}
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
    assert len(plan["query_intents"]) >= 2
    assert "fct_production" in plan["selected_tables"]


def test_analytical_bq_plan_hdi_via_contract() -> None:
    known = {"fct_hdi", "fct_economics", "fct_employment", "fct_production"}
    plan = build_analytical_bq_plan(
        "HDI trends across West Africa",
        decomposition={
            "geography": ["Nigeria", "Ghana", "Senegal"],
            "time_start": "2020-01-01",
            "time_end": "2024-12-31",
            "entities": ["HDI", "human development"],
        },
        known_tables=known,
    )
    assert plan is not None
    assert plan["skip_bq"] is False
    assert "fct_hdi" in plan["selected_tables"]
    assert plan["rationale"] == "analytical_forced_contract"


def test_analytical_bq_plan_climate_via_contract() -> None:
    known = {"fct_climate", "fct_production"}
    plan = build_analytical_bq_plan(
        "Climate rainfall Nigeria 2020-2024",
        decomposition={
            "geography": ["Nigeria"],
            "time_start": "2020-01-01",
            "time_end": "2024-12-31",
            "entities": ["rainfall", "climate"],
        },
        known_tables=known,
    )
    assert plan is not None
    assert "fct_climate" in plan["selected_tables"]
    assert "fct_production" not in plan["selected_tables"]


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


def test_sections_from_inline_headings() -> None:
    answer = (
        "## Executive summary West Africa grew slowly. "
        "## Country comparison Nigeria leads in maize. "
        "## Conclusion Invest in yields."
    )
    sections = sections_from_answer("Q?", answer)
    headings = [s["heading"] for s in sections]
    assert "Executive summary" in headings
    assert "Country comparison" in headings
    assert any("Nigeria" in s["body"] for s in sections)


def test_report_topic_is_not_query_slug() -> None:
    from ml.rag.chatbot.exports.tabular import report_topic

    query = (
        "give me a docx file report of the trend of production and trade maize and rice "
        "across west africa in 2022"
    )
    title, slug = report_topic(
        query,
        decomposition={
            "entities": ["maize", "rice"],
            "time_start": "2022-01-01",
            "time_end": "2022-12-31",
        },
    )
    assert "give_me_a_docx" not in slug
    assert "west" in slug
    assert "2022" in slug
    assert "West Africa" in title
    assert "2022" in title
