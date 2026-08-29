"""Tests for Africa zone catalog, Sahel expand, and geo purity."""
from __future__ import annotations

from ml.rag.chatbot.analytical_intent import is_analytical_query
from ml.rag.chatbot.bq_sql_templates import (
    build_mart_food_security_sql,
    match_mart_food_security_snapshot,
    try_sql_template,
)
from ml.rag.chatbot.generator import (
    _drop_geo_conflicting,
    usable_context_after_geo_purity,
)
from ml.rag.chatbot.geo_regions import (
    REGION_COUNTRIES,
    countries_for_regions,
    detect_regions_in_text,
    expand_regions_in_decomposition,
    is_zone_label,
)
from ml.rag.chatbot.plan_policy import apply_plan_decomposition_gates
from ml.rag.chatbot.query_decomposer import _NON_COUNTRY_GEO
from ml.rag.chatbot.task_mode import is_fact_lookup_query, resolve_task_mode


def test_sahel_detect_and_expand() -> None:
    assert "sahel" in detect_regions_in_text("food security across the Sahel")
    countries = countries_for_regions(["sahel"])
    assert "Mali" in countries
    assert "Niger" in countries
    assert "Kenya" not in countries


def test_ssa_not_west_africa_only() -> None:
    keys = detect_regions_in_text("maize production in sub-Saharan Africa")
    assert any(k in ("sub-saharan africa", "sub_saharan_africa") for k in keys)
    countries = countries_for_regions(keys)
    assert "Kenya" in countries
    assert "Nigeria" in countries
    assert "South Africa" in countries


def test_horn_maghreb_guinean_sadc() -> None:
    assert any("horn" in k for k in detect_regions_in_text("Horn of Africa drought"))
    assert "maghreb" in detect_regions_in_text("wheat in the Maghreb")
    assert any("guinean" in k for k in detect_regions_in_text("cocoa in the Guinean zone"))
    assert "sadc" in detect_regions_in_text("SADC maize outlook")
    gz = [k for k in detect_regions_in_text("Guinean zone") if "guinean" in k]
    assert "Ghana" in countries_for_regions(gz)


def test_afcfta_strip_only_is_zone_label() -> None:
    assert is_zone_label("AfCFTA")
    assert is_zone_label("sahel")
    assert not is_zone_label("Kenya")
    assert "sahel" in _NON_COUNTRY_GEO
    assert "afcfta" in _NON_COUNTRY_GEO or "african continental free trade area" in _NON_COUNTRY_GEO


def test_farmers_plan_preserves_sahel_expansion() -> None:
    q = "Give me a proper assessment of food security risk across the Sahel"
    dec = {"geography": ["Sahel"], "entities": ["food security", "Sahel"], "intent": "diagnostic"}
    expanded = expand_regions_in_decomposition(dec, q)
    assert expanded.get("expanded_regions")
    assert "Mali" in (expanded.get("geography") or [])
    gated = apply_plan_decomposition_gates(expanded, "Farmers", None)
    assert "Mali" in (gated.get("geography") or [])
    assert "Sahel" not in (gated.get("geography") or [])


def test_geo_purity_keeps_sahel_members_drops_kenya() -> None:
    decomp = expand_regions_in_decomposition(
        {"geography": ["Sahel"]},
        "food security in the Sahel",
    )
    items = [
        {
            "source": "news",
            "content": "IPC crisis in Mali",
            "metadata": {"country": "Mali", "geo_country_primary": "Mali"},
        },
        {
            "source": "news",
            "content": "Kenya maize",
            "metadata": {"country": "Kenya", "geo_country_primary": "Kenya"},
        },
        {
            "source": "policy",
            "content": "regional brief",
            "metadata": {},
        },
    ]
    kept = _drop_geo_conflicting(items, decomp)
    countries = {
        str((it.get("metadata") or {}).get("country") or "")
        for it in kept
    }
    assert "Mali" in countries
    assert "Kenya" not in countries
    assert any(not (it.get("metadata") or {}) for it in kept)


def test_kenya_purity_still_drops_senegal() -> None:
    decomp = {"geography": ["Kenya"]}
    items = [
        {
            "source": "news",
            "content": "x",
            "metadata": {"country": "Kenya", "geo_country_primary": "Kenya"},
        },
        {
            "source": "news",
            "content": "y",
            "metadata": {"country": "Senegal", "geo_country_primary": "Senegal"},
        },
    ]
    kept = _drop_geo_conflicting(items, decomp)
    assert len(kept) == 1
    assert (kept[0].get("metadata") or {}).get("country") == "Kenya"


def test_usable_after_purity_not_empty_for_sahel_narrative() -> None:
    decomp = expand_regions_in_decomposition(
        {"geography": ["Sahel"]},
        "Sahel food security assessment",
    )
    items = [
        {
            "source": "news",
            "content": "Flooding impacts farms in Niger",
            "metadata": {"country": "Niger", "geo_country_primary": "Niger"},
            "score": 0.5,
        },
        {
            "source": "public_reports",
            "content": "WFP Sahel update",
            "metadata": {"country": "Burkina Faso", "geo_country_primary": "Burkina Faso"},
            "score": 0.4,
        },
    ]
    usable = usable_context_after_geo_purity(items, decomp)
    assert len(usable) >= 2


def test_sahel_assessment_not_fact_lookup() -> None:
    q = (
        "Give me a proper assessment of food security risk across the Sahel "
        "— production, prices, and hunger pressure."
    )
    assert is_analytical_query(q, {"geography": ["Sahel"], "intent": "diagnostic"})
    assert not is_fact_lookup_query(q, {"geography": ["Sahel"], "intent": "diagnostic"})
    mode = resolve_task_mode(q, {"geography": ["Sahel"], "intent": "diagnostic"})
    assert mode == "analytical"


def test_fews_latest_and_country_in() -> None:
    assert match_mart_food_security_snapshot(
        query="food security risk across the Sahel",
        selected_tables={"fct_food_security"},
    )
    sql = build_mart_food_security_sql(
        project_id="proj",
        dataset="mart_dev",
        year=None,
        table_id="fct_food_security",
        countries=["Mali", "Niger", "Sahel"],
        blob="food security population",
    )
    assert "MAX(" in sql
    assert "IN (" in sql or "= 'MLI'" in sql or "'Mali'" in sql
    assert "'Sahel'" not in sql

    hit = try_sql_template(
        query="IPC food security assessment in the Sahel",
        project_id="proj",
        dataset="mart_dev",
        selected_tables={"fct_food_security"},
        geo_countries=list(REGION_COUNTRIES["sahel"]),
    )
    assert hit is not None
    assert hit["template"] == "mart_food_security_snapshot"
    assert "stg_" not in hit["sql"]
