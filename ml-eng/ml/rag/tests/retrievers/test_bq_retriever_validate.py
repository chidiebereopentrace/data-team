"""Retriever integration tests for SQL allowlist, dry-run retry, and diagnostics."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from ml.rag.chatbot.graph import aggregate_bq_sql_debug
from ml.rag.chatbot.streamlit_inspector import INSPECTOR_JSON_KEYS, _bq_was_attempted
from ml.rag.retrievers.bq_retriever import BQRetriever


def test_prepare_sql_rejects_unselected_table() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    sql = "SELECT country_name FROM `proj.staging_dev.stg_africa_gdp_ppp` LIMIT 5"
    validated, err = retriever._prepare_sql(
        sql,
        question="highest production in Africa",
        table_hints=["hint"],
        selected_tables={"stg_faostat_production"},
        allowed_datasets={"staging_dev"},
        limit=5,
        client=MagicMock(),
    )
    assert validated is None
    assert err is not None
    assert "stg_africa_gdp_ppp" in err


def test_prepare_sql_dry_run_retry() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    bad_sql = "SELECT country FROM `proj.staging_dev.stg_faostat_production` LIMIT 5"
    good_sql = (
        "SELECT country_name FROM `proj.staging_dev.stg_faostat_production` LIMIT 5"
    )
    client = MagicMock()

    def dry_side_effect(_client, sql):
        if "country FROM" in sql and "country_name" not in sql:
            return "Unrecognized name: country"
        return None

    with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", side_effect=dry_side_effect):
        with patch("ml.rag.retrievers.bq_retriever.sql_retry_enabled", return_value=True):
            with patch.object(retriever, "_nl_to_sql_one", return_value=good_sql):
                validated, err = retriever._prepare_sql(
                    bad_sql,
                    question="production ranking",
                    table_hints=["hint"],
                    selected_tables={"stg_faostat_production"},
                    allowed_datasets={"staging_dev"},
                    limit=5,
                    client=client,
                )
    assert validated is not None
    assert "country_name" in validated
    assert err is None


def test_retrieve_no_project_returns_diagnostic() -> None:
    retriever = BQRetriever(project_id="", nl2sql_enabled=True)
    items = retriever.retrieve("highest production in Africa 2020")
    assert len(items) == 1
    meta = items[0]["metadata"]
    assert meta["status"] == "no_project"
    assert meta["validation_failed"] is True


def test_retrieve_no_valid_sql_returns_diagnostic() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    with patch.object(retriever, "_nl_to_sql_queries", return_value=[]):
        with patch.object(retriever, "_get_client", return_value=MagicMock()):
            items = retriever.retrieve(
                "highest production",
                table_hints=["Table: staging_dev.stg_faostat_production"],
                selected_tables=["stg_faostat_production"],
            )
    assert len(items) == 1
    meta = items[0]["metadata"]
    assert meta["status"] == "no_valid_sql"
    assert "0 SELECT" in meta["prep_error"]


def test_aggregate_bq_sql_debug_includes_failures() -> None:
    results = [
        {
            "content": "[BQ validation failed: bad col]",
            "source": "bigquery",
            "metadata": {
                "sql": "SELECT country FROM t",
                "status": "validation_failed",
                "validation_failed": True,
                "prep_error": "Unrecognized name: country",
                "sql_index": 1,
            },
        },
        {
            "content": "{'country_name': 'Nigeria'}",
            "source": "bigquery",
            "metadata": {"sql": "SELECT country_name FROM t", "sql_index": 2},
        },
        {
            "content": "{'country_name': 'Kenya'}",
            "source": "bigquery",
            "metadata": {"sql": "SELECT country_name FROM t", "sql_index": 2},
        },
    ]
    queries, debug = aggregate_bq_sql_debug(results)
    assert "SELECT country FROM t" in queries
    assert "SELECT country_name FROM t" in queries
    assert any(d["status"] == "validation_failed" for d in debug)
    assert any(d["status"] == "ok" for d in debug)
    assert len([d for d in debug if d["status"] == "ok"]) == 1


def test_inspector_keys_include_sql_plan_and_debug() -> None:
    assert "bq_sql_plan" in INSPECTOR_JSON_KEYS
    assert "bq_sql_debug" in INSPECTOR_JSON_KEYS


def test_bq_was_attempted_with_candidates() -> None:
    assert _bq_was_attempted({"bq_table_candidates": [{"table_name": "stg_x"}]})
    assert _bq_was_attempted({"bq_sql_debug": [{"status": "no_valid_sql", "sql": ""}]})
    assert not _bq_was_attempted({"bq_sql_plan": {"skip_bq": True, "selected_tables": []}})
