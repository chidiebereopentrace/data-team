"""System regression tests for BQ semantic compile → validate → enrich."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.agri_measure_ontology import entity_is_measure_noise, measure_bind_skip_tokens
from ml.rag.chatbot.bq_context_enrich import enrich_bq_results
from ml.rag.chatbot.bq_sql_templates import build_mart_point_fact_sql, try_sql_template
from ml.rag.chatbot.bq_sql_validate import (
    inject_missing_metric_filters,
    max_bytes_billed_for_source,
    validate_semantic_coherence,
)
from ml.rag.chatbot.bq_table_schema_yaml import (
    compile_measure_filters,
    measure_blob,
    product_blob,
)
from ml.rag.chatbot.ontology_context import sanitize_decomposition_for_bq
from ml.rag.chatbot.retrieval_contract import choose_agg_vs_fact


def test_measure_bind_skip_tokens_includes_ontology_aliases() -> None:
    skip = measure_bind_skip_tokens()
    for token in ("exports", "export", "trade", "ipc", "market_price", "yield", "production"):
        assert token in skip


def test_product_blob_strips_measure_alias_entities() -> None:
    blob = product_blob(
        "maize from Nigeria in 2020",
        entities=["maize", "exports"],
    )
    assert "maize" in blob.lower()
    assert "exports" not in blob.lower()


def test_product_blob_strips_multi_word_measure_entity() -> None:
    blob = product_blob(
        "rice production in Kenya 2016",
        entities=["rice", "food security"],
    )
    assert "rice" in blob.lower()
    assert "food security" not in blob.lower()


def test_entity_is_measure_noise_phrase_and_explicit_aliases() -> None:
    assert entity_is_measure_noise(
        "food security phase",
        primary_measures=["food_security_ipc"],
    )
    assert entity_is_measure_noise("exporting", primary_measures=["trade"])
    assert entity_is_measure_noise("priced", primary_measures=["market_price"])


def test_entity_is_measure_noise_fuzzy_inflection_gated(monkeypatch) -> None:
    monkeypatch.delenv("RAG_MEASURE_ENTITY_FUZZY", raising=False)
    assert not entity_is_measure_noise("pricing", query="maize pricing in Mali 2024")

    monkeypatch.setenv("RAG_MEASURE_ENTITY_FUZZY", "1")
    assert entity_is_measure_noise(
        "pricing",
        query="maize pricing in Mali 2024",
        primary_measures=["market_price"],
    )
    assert not entity_is_measure_noise("pricing")
    assert not entity_is_measure_noise("maize", query="maize pricing in Mali 2024")


def test_disambiguate_promotes_yield_over_production_for_yield_question() -> None:
    from ml.rag.chatbot.agri_measure_ontology import disambiguate_primary_measures

    query = "What sorghum yield can farmers expect in Tahoua, Niger this harvest season?"
    dec = {
        "query": query,
        "entities": ["sorghum", "Tahoua", "Niger"],
        "primary_measures": ["production"],
    }
    assert disambiguate_primary_measures(query, ["production"], dec)[0] == "yield"


def test_disambiguate_keeps_production_for_volume_phrasing() -> None:
    from ml.rag.chatbot.agri_measure_ontology import disambiguate_primary_measures

    query = "What was total production of maize in Nigeria in 2022?"
    dec = {
        "query": query,
        "entities": ["maize", "yield"],
        "primary_measures": ["production"],
    }
    assert disambiguate_primary_measures(query, ["production"], dec)[0] == "production"


def test_disambiguate_keeps_declared_trade() -> None:
    from ml.rag.chatbot.agri_measure_ontology import disambiguate_primary_measures

    query = "maize exports from Nigeria in 2020"
    dec = {
        "query": query,
        "entities": ["maize", "exports"],
        "primary_measures": ["trade"],
    }
    assert disambiguate_primary_measures(query, ["trade"], dec)[0] == "trade"


def test_sanitize_decomposition_strips_trade_conflicts() -> None:
    dec = sanitize_decomposition_for_bq(
        {
            "entities": ["maize", "exports", "production"],
            "primary_measures": ["trade"],
        },
        primary_measures=["trade"],
    )
    ents = [e.lower() for e in dec.get("entities") or []]
    assert "exports" not in ents
    assert "production" not in ents
    assert "maize" in ents
    assert "trade" in ents


def test_sanitize_decomposition_strips_price_for_market_price() -> None:
    dec = sanitize_decomposition_for_bq(
        {
            "entities": ["maize", "price"],
            "primary_measures": ["market_price"],
        },
        primary_measures=["market_price"],
    )
    ents = [e.lower() for e in dec.get("entities") or []]
    assert "price" not in ents
    assert "maize" in ents
    assert "market_price" in ents


def test_measure_blob_ignores_decomposer_yield_noise_for_production() -> None:
    mb = measure_blob(
        "What was maize production in Nigeria in 2022?",
        primary_measures=["production", "yield"],
    )
    filters = dict(
        compile_measure_filters(
            "fct_production",
            measure_blob_text=mb,
            primary_measures=["production", "yield"],
        )
    )
    assert filters.get("element") == "Production"
    assert filters.get("metric") == "production_production_physical"


def test_measure_compiler_explicit_yield_question() -> None:
    mb = measure_blob(
        "What was maize yield in Nigeria in 2022?",
        primary_measures=["yield"],
    )
    filters = dict(
        compile_measure_filters(
            "fct_yield",
            measure_blob_text=mb,
            primary_measures=["yield"],
        )
    )
    assert filters.get("metric") == "production_maize_yield"


def test_build_mart_point_fact_production_with_yield_entity_noise() -> None:
    sql = build_mart_point_fact_sql(
        project_id="proj",
        dataset="mart_dev",
        table_id="fct_production",
        country_labels=["Nigeria"],
        year=2022,
        blob=product_blob(
            "What was maize production in Nigeria in 2022?",
            entities=["maize", "yield"],
        ),
        query="What was maize production in Nigeria in 2022?",
        primary_measures=["production", "yield"],
        time_start="2022-01-01",
        time_end="2022-12-31",
    )
    assert "element = 'Production'" in sql
    assert "element = 'Yield'" not in sql
    assert "metric = 'production_production_physical'" in sql
    assert "NGA" in sql
    assert "product_name = 'Maize (corn)'" in sql


def test_sanitize_decomposition_strips_yield_climate_for_production() -> None:
    dec = sanitize_decomposition_for_bq(
        {
            "entities": ["maize", "yield", "climate"],
            "domains": ["production", "climate"],
            "primary_measures": ["production"],
        },
        primary_measures=["production"],
    )
    ents = [e.lower() for e in dec.get("entities") or []]
    assert "yield" not in ents
    assert "climate" not in ents
    assert "production" in ents
    domains = [str(d).lower() for d in dec.get("domains") or []]
    assert "climate" not in domains


def test_validate_semantic_coherence_accepts_agg_time_key() -> None:
    sql = (
        "SELECT country_name, product_name, time_key, total_production_qty "
        "FROM `proj.mart_dev.agg_production_annual` "
        "WHERE time_key = 2022 AND country_name = 'Nigeria' "
        "AND product_name = 'Maize' LIMIT 1"
    )
    err = validate_semantic_coherence(
        sql,
        query="maize production Nigeria 2022",
        primary_measures=["production"],
        geography=["Nigeria"],
        time_start="2022-01-01",
        time_end="2022-12-31",
        table_ids={"agg_production_annual"},
    )
    assert err is None


def test_validate_semantic_coherence_rejects_yield_for_production() -> None:
    sql = (
        "SELECT value FROM `proj.mart_dev.fct_production` "
        "WHERE year = 2022 AND country_iso3 = 'NGA' "
        "AND element = 'Yield' AND metric = 'yield_harvested_area' "
        "AND product_name = 'Maize' LIMIT 1"
    )
    err = validate_semantic_coherence(
        sql,
        query="maize production Nigeria 2022",
        primary_measures=["production"],
        geography=["Nigeria"],
        time_start="2022-01-01",
        time_end="2022-12-31",
        table_ids={"fct_production"},
    )
    assert err is not None
    assert "Yield" in err


def test_validate_semantic_coherence_requires_geo_filter() -> None:
    sql = (
        "SELECT value FROM `proj.mart_dev.fct_production` "
        "WHERE year = 2022 AND element = 'Production' "
        "AND metric = 'production_production_physical' "
        "AND product_name = 'Maize' LIMIT 1"
    )
    err = validate_semantic_coherence(
        sql,
        query="maize production Nigeria 2022",
        primary_measures=["production"],
        geography=["Nigeria"],
        time_start="2022-01-01",
        time_end="2022-12-31",
        table_ids={"fct_production"},
    )
    assert err is not None
    assert "geography" in err.lower()


def test_validate_semantic_coherence_partition_predicate_on_fct_production() -> None:
    sql = (
        "SELECT value FROM `proj.mart_dev.fct_production` "
        "WHERE country_iso3 = 'NGA' AND element = 'Production' "
        "AND metric = 'production_production_physical' "
        "AND product_name = 'Maize' LIMIT 1"
    )
    err = validate_semantic_coherence(
        sql,
        query="maize production Nigeria",
        primary_measures=["production"],
        geography=["Nigeria"],
        table_ids={"fct_production"},
    )
    assert err is not None
    assert "as_of_date" in err or "year" in err


def test_choose_agg_vs_fact_national_volume() -> None:
    routed = choose_agg_vs_fact(
        "fct_production",
        query="What was maize production in Nigeria in 2022?",
        multi_country=False,
        year_hint="2022",
        single_country=True,
    )
    assert routed == "agg_production_country_year"


def test_try_sql_template_routes_to_agg_when_available() -> None:
    hit = try_sql_template(
        query="What was maize production in Nigeria in 2022?",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["agg_production_annual", "fct_production"],
        geo_country="Nigeria",
        time_start="2022-01-01",
        time_end="2022-12-31",
        primary_measures=["production"],
    )
    assert hit is not None
    assert hit["table_id"] in ("agg_production_country_year", "agg_production_annual", "fct_production")
    assert hit["table_id"] in hit["sql"]


def test_inject_missing_metric_filters_uses_primary_measures() -> None:
    sql = (
        "SELECT value FROM `proj.mart_dev.fct_production` "
        "WHERE year = 2022 AND country_iso3 = 'NGA' AND product_name = 'Maize' "
        "LIMIT 1"
    )
    patched, notes = inject_missing_metric_filters(
        sql,
        {"fct_production"},
        query="maize production Nigeria 2022 yield",
        primary_measures=["production"],
    )
    assert "element = 'Production'" in patched
    assert notes
    assert any("Production" in n for n in notes)


def test_max_bytes_billed_source_defaults() -> None:
    assert max_bytes_billed_for_source("template") == 512 * 1024 * 1024
    assert max_bytes_billed_for_source("nl2sql") == 250 * 1024 * 1024


def test_enrich_bq_results_drops_yield_row_for_production_intent() -> None:
    item = {
        "content": str(
            {
                "country_iso3": "NGA",
                "country_name": "Nigeria",
                "product_name": "Maize",
                "year": 2022,
                "element": "Yield",
                "value": 3.5,
            }
        ),
        "source": "bigquery",
        "metadata": {
            "sql": (
                "SELECT * FROM `proj.mart_dev.fct_production` "
                "WHERE year = 2022 AND country_iso3 = 'NGA'"
            ),
        },
        "score": 1.0,
    }
    out = enrich_bq_results(
        [item],
        query="maize production Nigeria 2022",
        decomposition={
            "primary_measures": ["production"],
            "geography": ["Nigeria"],
            "time_start": "2022-01-01",
            "time_end": "2022-12-31",
        },
    )
    rejected = [
        it for it in out if (it.get("metadata") or {}).get("semantic_row_rejected")
    ]
    enriched = [
        it
        for it in out
        if is_usable_enriched(it)
    ]
    assert not rejected
    assert not enriched


def test_try_sql_template_tahoua_yield_routes_to_fct_yield() -> None:
    dec = sanitize_decomposition_for_bq(
        {
            "query": "What sorghum yield can farmers expect in Tahoua, Niger this harvest season?",
            "entities": ["sorghum", "Tahoua", "Niger"],
            "primary_measures": ["production"],
        },
        primary_measures=["production"],
    )
    assert dec["primary_measures"][0] == "yield"
    hit = try_sql_template(
        query=dec.get("query") or "",
        project_id="proj",
        dataset="mart_dev",
        selected_tables=["fct_yield", "fct_production"],
        entities=dec.get("entities"),
        geo_country="Niger",
        primary_measures=dec.get("primary_measures"),
        task_mode="fact_lookup",
    )
    assert hit is not None
    assert hit["table_id"] == "fct_yield"
    assert "fct_yield" in hit["sql"]
    assert "fct_production" not in hit["sql"]


def is_usable_enriched(item: dict) -> bool:
    meta = item.get("metadata") or {}
    return bool(meta.get("value_semantics")) and not meta.get("semantic_row_rejected")
