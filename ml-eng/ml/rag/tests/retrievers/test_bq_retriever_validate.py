"""Retriever integration tests for SQL allowlist and dry-run retry."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

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
