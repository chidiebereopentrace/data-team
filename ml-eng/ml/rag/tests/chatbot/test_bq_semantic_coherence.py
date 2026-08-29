"""System regression tests for BQ semantic compile → validate → enrich."""
from __future__ import annotations

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
from ml.rag.chatbot.retrieval_contract import choose_agg_vs_fact


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
    assert "product_name = 'Maize'" in sql


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
    assert routed == "agg_production_annual"


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
    assert hit["table_id"] == "agg_production_annual"
    assert "agg_production_annual" in hit["sql"]


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
    assert rejected
    assert not enriched


def is_usable_enriched(item: dict) -> bool:
    meta = item.get("metadata") or {}
    return bool(meta.get("value_semantics")) and not meta.get("semantic_row_rejected")
