"""Tests for NL2SQL-primary planned path and enriched NL2SQL prompts."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.rag.chatbot.bq_table_schema_yaml import intent_pattern_for_query
from ml.rag.retrievers.bq_retriever import BQRetriever, _format_query_constraints, _load_dotenv


def test_format_query_constraints_includes_bind_intents_and_crop_entities() -> None:
    bind = {
        "table_id": "agg_production_country_year",
        "nomenclature": (
            "TABLE: agg_production_country_year\n"
            "GEO: country_iso3 = 'KEN'\n"
            "TIME: year = 2016\n"
            "PRODUCT: product_name = 'Rice'"
        ),
        "required_filters_sql": "country_code = 'KEN' AND year = 2016",
        "product_literals": ["Rice"],
        "measure_column": "production_tons",
    }
    intents = [
        {
            "goal": "fact_lookup",
            "tables": ["agg_production_country_year"],
            "pattern": "custom",
            "metric": "production",
            "grain": "country_year",
            "filters": {"country": "Kenya", "year": 2016},
        }
    ]
    text = _format_query_constraints(
        geo_country="Kenya",
        time_start="2016-01-01",
        time_end="2016-12-31",
        entities=["production", "rice"],
        domains=None,
        query="what is the production of rice in kenya in 2016",
        bind_contracts={"agg_production_country_year": bind},
        query_intents=intents,
        primary_measures=["production"],
        task_mode="fact_lookup",
    )
    assert "country_iso3 = 'KEN'" in text
    assert "Rice" in text
    assert "fact_lookup" in text
    assert "rice" in text.lower()
    assert "Primary measures" in text


def test_credential_bootstrap_errors_are_not_silenced() -> None:
    with patch(
        "ml.rag.retrievers.bq_retriever.load_rag_dotenv",
        side_effect=RuntimeError("invalid BigQuery credentials"),
    ):
        with pytest.raises(RuntimeError, match="invalid BigQuery credentials"):
            _load_dotenv()


def test_planned_path_skips_template_and_pattern_calls_nl2sql() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    good_sql = (
        "SELECT production_qty FROM `proj.mart_dev.agg_production_country_year` "
        "WHERE country_iso3 = 'KEN' AND year = 2016 LIMIT 1"
    )
    bind = {
        "agg_production_country_year": {
            "table_id": "agg_production_country_year",
            "nomenclature": "TABLE: agg_production_country_year\nPRODUCT: product_name = 'Rice'",
            "required_filters_sql": "country_iso3 = 'KEN' AND year = 2016",
            "product_literals": ["Rice"],
        }
    }
    client = MagicMock()
    client.query.return_value.result.return_value = [{"production_tons": 100}]

    with patch.object(retriever, "_nl_to_sql_queries", return_value=[good_sql]) as nl2sql_fn:
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                with patch(
                    "ml.rag.retrievers.bq_retriever.try_sql_template",
                ) as template_fn:
                    with patch(
                        "ml.rag.retrievers.bq_retriever.try_sql_patterns",
                    ) as pattern_fn:
                        items = retriever.retrieve(
                            "what is the production of rice in kenya in 2016",
                            selected_tables=["agg_production_country_year"],
                            bind_contracts=bind,
                            query_intents=[{"goal": "fact_lookup", "pattern": "custom"}],
                            primary_measures=["production"],
                            entities=["production", "rice"],
                            task_mode="fact_lookup",
                            time_start="2016-01-01",
                            time_end="2016-12-31",
                        )

    template_fn.assert_not_called()
    pattern_fn.assert_not_called()
    nl2sql_fn.assert_called_once()
    assert nl2sql_fn.call_args.kwargs.get("query_intents") is not None
    assert any((it.get("metadata") or {}).get("sql_source") == "nl2sql" for it in items)
    assert "KEN" in good_sql
    assert "2016" in good_sql


def test_retrieve_mode_planned_without_bind_skips_templates() -> None:
    """Explicit retrieve_mode=planned prefers planned path even without bind map."""
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    good_sql = (
        "SELECT production_qty FROM `proj.mart_dev.agg_production_country_year` "
        "WHERE country_iso3 = 'KEN' LIMIT 1"
    )
    client = MagicMock()
    client.query.return_value.result.return_value = [{"production_tons": 1}]

    with patch.object(retriever, "_nl_to_sql_queries", return_value=[good_sql]) as nl2sql_fn:
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                with patch("ml.rag.retrievers.bq_retriever.try_sql_template") as template_fn:
                    with patch("ml.rag.retrievers.bq_retriever.try_sql_patterns") as pattern_fn:
                        retriever.retrieve(
                            "rice production kenya",
                            selected_tables=["agg_production_country_year"],
                            retrieve_mode="planned",
                            plan_source="class_engine",
                        )

    template_fn.assert_not_called()
    pattern_fn.assert_not_called()
    nl2sql_fn.assert_called_once()


def test_intent_pattern_point_fact_is_custom() -> None:
    pattern = intent_pattern_for_query(
        "what is the production of rice in kenya in 2016",
        "agg_production_country_year",
        multi_country=False,
        time_start="2016-01-01",
        time_end="2016-12-31",
    )
    assert pattern == "custom"


def test_prepare_sql_retry_passes_bind_contracts() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    bad_sql = (
        "SELECT production_qty FROM `proj.mart_dev.agg_production_country_year` LIMIT 1"
    )
    good_sql = (
        "SELECT production_qty FROM `proj.mart_dev.agg_production_country_year` "
        "WHERE country_iso3 = 'KEN' AND year = 2016 LIMIT 1"
    )
    bind = {
        "agg_production_country_year": {
            "table_id": "agg_production_country_year",
            "nomenclature": "TABLE: agg_production_country_year\nPRODUCT: product_name = 'Rice'",
            "required_filters_sql": "country_iso3 = 'KEN'",
            "product_literals": ["Rice"],
            "measure_columns": ["production_qty"],
        }
    }
    client = MagicMock()

    with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
        with patch("ml.rag.retrievers.bq_retriever.sql_retry_enabled", return_value=True):
            with patch(
                "ml.rag.retrievers.bq_retriever.validate_semantic_coherence",
                side_effect=["missing product filter", None],
            ):
                with patch.object(retriever, "_nl_to_sql_one", return_value=good_sql) as retry_fn:
                    validated, err = retriever._prepare_sql(
                        bad_sql,
                        question="what is the production of rice in kenya in 2016",
                        table_hints=["hint"],
                        selected_tables={"agg_production_country_year"},
                        allowed_datasets={"mart_dev"},
                        limit=1,
                        client=client,
                        bind_contracts=bind,
                        query_intents=[{"goal": "fact_lookup"}],
                        primary_measures=["production"],
                    )
    assert validated is not None
    assert err is None
    retry_fn.assert_called_once()
    assert retry_fn.call_args.kwargs.get("bind_contracts") == bind
    assert retry_fn.call_args.kwargs.get("query_intents") == [{"goal": "fact_lookup"}]


def test_bind_compiler_skips_nl2sql() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    bind = {
        "agg_production_country_year": {
            "table_id": "agg_production_country_year",
            "required_filters_sql": "country_iso3 = 'KEN' AND year = 2016",
            "measure_columns": ["production_qty"],
        }
    }
    client = MagicMock()
    client.query.return_value.result.return_value = [{"production_qty": 100}]

    with patch.object(retriever, "_nl_to_sql_queries") as nl2sql_fn:
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                items = retriever.retrieve(
                    "what is the production of rice in kenya in 2016",
                    selected_tables=["agg_production_country_year"],
                    bind_contracts=bind,
                    query_intents=[{"goal": "fact_lookup"}],
                    task_mode="fact_lookup",
                    plan_source="class_engine",
                )

    nl2sql_fn.assert_not_called()
    assert any((it.get("metadata") or {}).get("sql_source") == "bind_compiler" for it in items)


def test_price_template_keeps_other_requested_warehouse_indicators() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    production = "agg_production_country_year"
    client = MagicMock()
    client.query.return_value.result.return_value = [{"country_iso3": "KEN", "value": 100}]
    with (
        patch.object(retriever, "_get_client", return_value=client),
        patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None),
    ):
        items = retriever.retrieve(
            "What were the maize price and maize production in Kenya in 2023?",
            selected_tables=["fct_prices", production],
            bind_contracts={
                production: {
                    "table_id": production,
                    "required_filters_sql": (
                        "country_iso3 = 'KEN' AND year = 2023 AND product_key IN "
                        "(SELECT product_key FROM `proj.mart_dev.dim_product` WHERE product_name = 'Maize')"
                    ),
                    "measure_columns": ["production_qty"],
                },
            },
            query_intents=[{"goal": "price"}, {"goal": "production"}],
            primary_measures=["market_price", "production"],
            entities=["maize"],
            geo_country="Kenya",
            time_start="2023-01-01",
            time_end="2023-12-31",
            task_mode="analytical",
            plan_source="class_engine",
        )

    sources = {
        item["metadata"]["sql_source"] for item in items
        if item.get("metadata", {}).get("job_id")
    }
    assert {"template", "bind_compiler"}.issubset(sources)
