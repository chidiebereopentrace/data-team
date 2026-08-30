"""Retriever integration tests for SQL allowlist, dry-run retry, and diagnostics."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from ml.rag.chatbot.graph import aggregate_bq_sql_debug
from ml.rag.chatbot.streamlit_inspector import INSPECTOR_JSON_KEYS, _bq_was_attempted
from ml.rag.retrievers.bq_retriever import BQRetriever


def test_prepare_sql_rejects_unselected_table() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    sql = "SELECT country_name FROM `proj.staging_dev.stg_africa_gdp_ppp` LIMIT 5"
    with patch("ml.rag.retrievers.bq_retriever.sql_retry_enabled", return_value=False):
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


def test_prepare_sql_rejects_dim_geography() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 AND area_code_m49 IN ("
        "SELECT country_code FROM `proj.staging_dev.dim_geography` "
        "WHERE country_name = 'Africa') "
        "GROUP BY country_name ORDER BY total DESC LIMIT 5"
    )
    with patch("ml.rag.retrievers.bq_retriever.sql_retry_enabled", return_value=False):
        validated, err = retriever._prepare_sql(
            sql,
            question="highest production in Africa 2020",
            table_hints=["hint"],
            selected_tables={"stg_faostat_production"},
            allowed_datasets={"staging_dev"},
            limit=5,
            client=MagicMock(),
        )
    assert validated is None
    assert err is not None
    assert "dim_geography" in err


def test_prepare_sql_allowlist_triggers_retry() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    bad_sql = (
        "SELECT country_name FROM `proj.staging_dev.stg_faostat_production` "
        "JOIN `proj.staging_dev.dim_geography` d USING (area_code_m49) LIMIT 5"
    )
    good_sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2020 AND element = 'Production' "
        "GROUP BY country_name ORDER BY total DESC LIMIT 5"
    )
    with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
        with patch("ml.rag.retrievers.bq_retriever.sql_retry_enabled", return_value=True):
            with patch.object(retriever, "_nl_to_sql_one", return_value=good_sql) as retry_fn:
                validated, err = retriever._prepare_sql(
                    bad_sql,
                    question="highest production Africa 2020",
                    table_hints=["hint"],
                    selected_tables={"stg_faostat_production"},
                    allowed_datasets={"staging_dev"},
                    limit=5,
                    client=MagicMock(),
                )
    assert validated is not None
    assert "dim_geography" not in validated
    assert err is None
    retry_fn.assert_called_once()
    assert "dim_geography" in str(retry_fn.call_args)


def test_prepare_sql_dry_run_retry() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    bad_sql = "SELECT country FROM `proj.staging_dev.stg_faostat_production` LIMIT 5"
    good_sql = (
        "SELECT country_name FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE element = 'Production' LIMIT 5"
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


def test_prepare_sql_injects_missing_element_without_retry() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` "
        "WHERE year = 2022 AND country_name = 'Nigeria' AND product_name = 'Maize' "
        "GROUP BY country_name LIMIT 5"
    )
    with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
        with patch("ml.rag.retrievers.bq_retriever.sql_retry_enabled", return_value=False):
            with patch.object(retriever, "_nl_to_sql_one") as retry_fn:
                validated, err = retriever._prepare_sql(
                    sql,
                    question="maize production Nigeria 2022",
                    table_hints=["hint"],
                    selected_tables={"stg_faostat_production"},
                    allowed_datasets={"staging_dev"},
                    limit=5,
                    client=MagicMock(),
                )
    assert err is None
    assert validated is not None
    assert "element = 'Production'" in validated
    retry_fn.assert_not_called()


def test_retrieve_no_project_returns_diagnostic() -> None:
    retriever = BQRetriever(project_id="", nl2sql_enabled=True)
    items = retriever.retrieve("highest production in Africa 2020")
    assert len(items) == 1
    meta = items[0]["metadata"]
    assert meta["status"] == "no_project"
    assert meta["validation_failed"] is True


def test_retrieve_no_valid_sql_returns_diagnostic() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    retriever._last_nl2sql_raws = ["Here is an explanation without SELECT"]
    with patch.object(retriever, "_nl_to_sql_queries", return_value=[]):
        with patch.object(retriever, "_get_client", return_value=MagicMock()):
            with patch(
                "ml.rag.retrievers.bq_retriever.try_sql_template",
                return_value=None,
            ):
                items = retriever.retrieve(
                    "highest production",
                    table_hints=["Table: staging_dev.stg_faostat_production"],
                    selected_tables=["stg_faostat_production"],
                )
    assert len(items) == 1
    meta = items[0]["metadata"]
    assert meta["status"] == "no_valid_sql"
    assert "0 SELECT" in meta["prep_error"]
    assert "explanation" in (meta.get("nl2sql_raw") or "")


def test_retrieve_template_fallback_when_nl2sql_empty() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    client = MagicMock()
    client.query.return_value.result.return_value = [
        {"country_iso3": "NGA", "total": 100.0}
    ]

    with patch.object(retriever, "_nl_to_sql_queries", return_value=[]):
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                items = retriever.retrieve(
                    "which country in africa had the highest agricultural production in 2020",
                    selected_tables=["fct_production"],
                    time_start="2020-01-01",
                    time_end="2020-12-31",
                )
    assert any(
        (it.get("metadata") or {}).get("sql_source") == "template"
        or (it.get("metadata") or {}).get("template") == "mart_country_rank"
        or "NGA" in str(it.get("content"))
        for it in items
    )
    assert any(
        "fct_production" in str((it.get("metadata") or {}).get("sql") or "")
        for it in items
    )


def test_retrieve_fast_fact_skips_nl2sql_without_custom_intent() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    good_sql = (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `proj.mart_dev.fct_production` "
        "WHERE year = 2020 AND production_grain = 'physical' "
        "AND element = 'Production' AND metric = 'production_production_physical' "
        "GROUP BY country_iso3 ORDER BY total DESC LIMIT 10"
    )
    client = MagicMock()
    client.query.return_value.result.return_value = [
        {"country_iso3": "KEN", "total": 50.0}
    ]
    with patch.object(retriever, "_nl_to_sql_queries", return_value=[good_sql]) as nl:
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                with patch(
                    "ml.rag.retrievers.bq_retriever.try_sql_template",
                    return_value=None,
                ):
                    with patch(
                        "ml.rag.retrievers.bq_retriever.try_sql_patterns",
                        return_value=[],
                    ):
                        items = retriever.retrieve(
                            "highest agricultural production in Africa 2020",
                            selected_tables=["fct_production"],
                            time_start="2020-01-01",
                            task_mode="fact_lookup",
                        )
    nl.assert_not_called()
    assert len(items) == 1
    assert items[0]["metadata"]["status"] == "no_valid_sql"


_PRODUCTION_WHERE = (
    "production_grain = 'physical' "
    "AND element = 'Production' AND metric = 'production_production_physical'"
)


def test_retrieve_prefers_valid_nl2sql_with_custom_intent() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    good_sql = (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `proj.mart_dev.fct_production` "
        f"WHERE year = 2020 AND {_PRODUCTION_WHERE} "
        "GROUP BY country_iso3 ORDER BY total DESC LIMIT 10"
    )
    client = MagicMock()
    client.query.return_value.result.return_value = [
        {"country_iso3": "KEN", "total": 50.0}
    ]
    with patch.object(retriever, "_nl_to_sql_queries", return_value=[good_sql]) as nl:
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                with patch(
                    "ml.rag.retrievers.bq_retriever.try_sql_template",
                    return_value=None,
                ):
                    with patch(
                        "ml.rag.retrievers.bq_retriever.try_sql_patterns",
                        return_value=[],
                    ):
                        items = retriever.retrieve(
                            "highest agricultural production in Africa 2020",
                            selected_tables=["fct_production"],
                            time_start="2020-01-01",
                            query_intents=[
                                {"pattern": "custom", "tables": ["fct_production"]},
                            ],
                            task_mode="analytical",
                            crop_required=False,
                            geography_required=False,
                        )
    nl.assert_called_once()
    assert any("KEN" in str(it.get("content")) for it in items)
    assert any((it.get("metadata") or {}).get("sql_source") == "nl2sql" for it in items)


def test_retrieve_prefers_valid_nl2sql_over_template() -> None:
    test_retrieve_prefers_valid_nl2sql_with_custom_intent()


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


def test_retrieve_empty_nl2sql_rows_retries_template() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    tmpl_sql = (
        "SELECT country_iso3, value "
        "FROM `proj.mart_dev.fct_production` "
        f"WHERE year = 2022 AND {_PRODUCTION_WHERE} "
        "AND country_iso3 = 'NGA' LIMIT 5"
    )
    filled_job = MagicMock()
    filled_job.result.return_value = [{"country_iso3": "NGA", "value": 10.0}]
    client = MagicMock()
    client.query.return_value = filled_job
    with patch.object(retriever, "_nl_to_sql_queries") as nl:
        nl.return_value = []
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                with patch(
                    "ml.rag.retrievers.bq_retriever.try_sql_template",
                    return_value={"sql": tmpl_sql, "template": "mart_point_fact"},
                ) as tmpl:
                    with patch(
                        "ml.rag.retrievers.bq_retriever.try_sql_patterns",
                        return_value=[],
                    ):
                        items = retriever.retrieve(
                            "maize production west africa 2022",
                            selected_tables=["fct_production"],
                            time_start="2022-01-01",
                            time_end="2022-12-31",
                            crop_required=False,
                            geography_required=False,
                        )
    tmpl.assert_called()
    nl.assert_not_called()
    assert any("NGA" in str(it.get("content")) for it in items)
    assert any((it.get("metadata") or {}).get("sql_source") == "template" for it in items)


def test_retrieve_pattern_hits_still_nl2sql_leftover_custom() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    pattern_sql = (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `proj.mart_dev.fct_production` "
        f"WHERE year = 2022 AND {_PRODUCTION_WHERE} "
        "GROUP BY country_iso3 ORDER BY total DESC LIMIT 10"
    )
    leftover_sql = (
        "SELECT country_iso3 FROM `proj.mart_dev.fct_household` "
        "LIMIT 10"
    )
    client = MagicMock()
    client.query.return_value.result.return_value = [{"country_iso3": "NGA", "total": 1}]
    with patch(
        "ml.rag.retrievers.bq_retriever.try_sql_patterns",
        return_value=[
            {
                "sql": pattern_sql,
                "pattern": "rank_by_sum",
                "table_id": "fct_production",
                "intent_index": 0,
            }
        ],
    ):
        with patch(
            "ml.rag.retrievers.bq_retriever.try_sql_template",
            return_value=None,
        ):
            with patch.object(retriever, "_nl_to_sql_queries", return_value=[leftover_sql]) as nl:
                with patch.object(retriever, "_get_client", return_value=client):
                    with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                        items = retriever.retrieve(
                            "maize production and household food security 2022",
                            selected_tables=[
                                "fct_production",
                                "fct_household",
                            ],
                            query_intents=[
                                {
                                    "pattern": "rank_by_sum",
                                    "tables": ["fct_production"],
                                },
                                {
                                    "pattern": "custom",
                                    "tables": ["fct_household"],
                                },
                            ],
                            time_start="2022-01-01",
                        )
    nl.assert_called_once()
    leftover_tables = nl.call_args.kwargs.get("selected_tables")
    assert leftover_tables == ["fct_household"]
    sources = {(it.get("metadata") or {}).get("sql_source") for it in items}
    assert "pattern" in sources
    assert "nl2sql" in sources


def test_retrieve_zero_rows_labeled_empty_result() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    nl_sql = (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `proj.mart_dev.fct_production` "
        f"WHERE year = 2022 AND {_PRODUCTION_WHERE} "
        "GROUP BY country_iso3 LIMIT 10"
    )
    client = MagicMock()
    client.query.return_value.result.return_value = []
    with patch.object(retriever, "_nl_to_sql_queries", return_value=[nl_sql]):
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                with patch(
                    "ml.rag.retrievers.bq_retriever.try_sql_template",
                    return_value=None,
                ):
                    with patch(
                        "ml.rag.retrievers.bq_retriever.try_sql_patterns",
                        return_value=[],
                    ):
                        items = retriever.retrieve(
                            "maize production 2022",
                            selected_tables=["fct_production"],
                            time_start="2022-01-01",
                            query_intents=[
                                {"pattern": "custom", "tables": ["fct_production"]},
                            ],
                            task_mode="analytical",
                            crop_required=False,
                            geography_required=False,
                        )
    assert len(items) == 1
    meta = items[0]["metadata"]
    assert meta["status"] == "empty_result"
    assert "no rows" in (meta.get("prep_error") or "").lower()


def test_retrieve_broadens_empty_year_once() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    nl_sql = (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `proj.mart_dev.fct_production` "
        f"WHERE year = 2022 AND {_PRODUCTION_WHERE} "
        "GROUP BY country_iso3 LIMIT 10"
    )
    empty_job = MagicMock()
    empty_job.result.return_value = []
    filled_job = MagicMock()
    filled_job.result.return_value = [{"country_iso3": "NGA", "value": 10.0}]
    client = MagicMock()
    client.query.side_effect = [empty_job, filled_job]
    with patch.object(retriever, "_nl_to_sql_queries", return_value=[nl_sql]):
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                with patch(
                    "ml.rag.retrievers.bq_retriever.validate_dry_run_bytes",
                    return_value=None,
                ):
                    with patch(
                        "ml.rag.retrievers.bq_retriever.try_sql_template",
                        return_value=None,
                    ):
                        with patch(
                            "ml.rag.retrievers.bq_retriever.try_sql_patterns",
                            return_value=[],
                        ):
                            items = retriever.retrieve(
                                "maize production 2022",
                                selected_tables=["fct_production"],
                                time_start="2022-01-01",
                                query_intents=[
                                    {"pattern": "custom", "tables": ["fct_production"]},
                                ],
                                task_mode="analytical",
                                crop_required=False,
                                geography_required=False,
                            )
    assert any("NGA" in str(it.get("content")) for it in items)
    execute_calls = [
        c.args[0]
        for c in client.query.call_args_list
        if c.args and str(c.args[0]).upper().startswith("SELECT")
    ]
    assert len(execute_calls) >= 2
    assert "BETWEEN 2021 AND 2023" in execute_calls[-1]
    assert "dim_" not in execute_calls[-1]


def test_retrieve_invalid_nl2sql_rescued_by_pattern() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    bad_sql = (
        "SELECT country_iso3 FROM `proj.mart_dev.fct_production` "
        "JOIN `proj.mart_dev.dim_geography` d USING (geography_key) LIMIT 5"
    )
    pattern_sql = (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `proj.mart_dev.fct_production` "
        f"WHERE year = 2022 AND {_PRODUCTION_WHERE} "
        "GROUP BY country_iso3 LIMIT 10"
    )
    client = MagicMock()
    client.query.return_value.result.return_value = [{"country_iso3": "NGA", "total": 1}]
    with patch(
        "ml.rag.retrievers.bq_retriever.try_sql_patterns",
        side_effect=[
            [],
            [
                {
                    "sql": pattern_sql,
                    "pattern": "rank_by_sum",
                    "table_id": "fct_production",
                    "intent_index": 0,
                }
            ],
        ],
    ):
        with patch(
            "ml.rag.retrievers.bq_retriever.try_sql_template",
            return_value=None,
        ):
            with patch.object(retriever, "_nl_to_sql_queries", return_value=[bad_sql]):
                with patch.object(retriever, "_get_client", return_value=client):
                    with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                        with patch(
                            "ml.rag.retrievers.bq_retriever.sql_retry_enabled",
                            return_value=False,
                        ):
                            items = retriever.retrieve(
                                "maize production west africa 2022",
                                selected_tables=["fct_production"],
                                time_start="2022-01-01",
                                crop_required=False,
                                geography_required=False,
                            )
    assert any("NGA" in str(it.get("content")) for it in items)
    assert any((it.get("metadata") or {}).get("sql_source") == "pattern" for it in items)


def test_template_preferred_over_nl2sql() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    template_sql = (
        "SELECT country_iso3, value FROM `proj.mart_dev.fct_production` "
        "WHERE year = 2020 LIMIT 5"
    )
    client = MagicMock()
    client.query.return_value.result.return_value = [{"country_iso3": "KEN", "value": 100}]
    with patch("ml.rag.retrievers.bq_retriever.try_sql_template") as tmpl:
        with patch("ml.rag.retrievers.bq_retriever.try_sql_patterns") as patterns:
            with patch.object(retriever, "_nl_to_sql_queries") as nl2sql:
                tmpl.return_value = {"sql": template_sql, "template": "production_fact"}
                patterns.return_value = []
                nl2sql.return_value = []
                with patch.object(retriever, "_get_client", return_value=client):
                    with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                        items = retriever.retrieve(
                            "maize production Kenya 2020",
                            task_mode="fact_lookup",
                            selected_tables={"fct_production"},
                        )
    nl2sql.assert_not_called()
    patterns.assert_not_called()
    assert items
    assert any((it.get("metadata") or {}).get("sql_source") == "template" for it in items)


def test_prepare_sql_accepts_nigeria_maize_agg_time_key() -> None:
    """Regression: agg_production_annual uses time_key, not year."""
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    sql = (
        "SELECT country_name, product_name, time_key, total_production_qty "
        "FROM `proj.mart_dev.agg_production_annual` "
        "WHERE time_key = 2022 AND country_name = 'Nigeria' "
        "AND product_name = 'Maize' LIMIT 1"
    )
    with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
        with patch("ml.rag.retrievers.bq_retriever.sql_retry_enabled", return_value=False):
            validated, err = retriever._prepare_sql(
                sql,
                question="What was maize production in Nigeria in 2022?",
                table_hints=["hint"],
                selected_tables={"agg_production_annual"},
                allowed_datasets={"mart_dev"},
                limit=1,
                client=MagicMock(),
                time_start="2022-01-01",
                time_end="2022-12-31",
                primary_measures=["production"],
                geography=["Nigeria"],
                sql_source="template",
            )
    assert err is None, err
    assert validated is not None
    assert "time_key = 2022" in validated


def test_retrieve_market_price_uses_template_not_nl2sql() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    template_sql = (
        "SELECT price, as_of_date FROM `proj.mart_dev.fct_prices` "
        "WHERE country_iso3 = 'MLI' AND product_name = 'Maize' "
        "ORDER BY as_of_date DESC LIMIT 1"
    )
    client = MagicMock()
    client.query.return_value.result.return_value = [
        {"price": 320.0, "as_of_date": "2024-06-01"}
    ]
    with patch.object(retriever, "_nl_to_sql_queries") as nl:
        with patch.object(retriever, "_get_client", return_value=client):
            with patch("ml.rag.retrievers.bq_retriever.dry_run_sql", return_value=None):
                with patch(
                    "ml.rag.retrievers.bq_retriever.try_sql_template",
                    return_value={
                        "sql": template_sql,
                        "template": "mart_latest_price",
                        "table_id": "fct_prices",
                    },
                ):
                    with patch(
                        "ml.rag.retrievers.bq_retriever.try_sql_patterns",
                        return_value=[],
                    ):
                        items = retriever.retrieve(
                            "current retail price of maize in Bamako Mali",
                            selected_tables=["fct_prices"],
                            task_mode="fact_lookup",
                            primary_measures=["market_price"],
                            geo_country="Mali",
                        )
    nl.assert_not_called()
    assert len(items) == 1
    assert items[0]["metadata"]["sql_source"] == "template"


def test_retrieve_food_security_skips_nl2sql_when_template_misses() -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    with patch.object(retriever, "_nl_to_sql_queries") as nl:
        with patch.object(retriever, "_get_client", return_value=MagicMock()):
            with patch(
                "ml.rag.retrievers.bq_retriever.try_sql_template",
                return_value=None,
            ):
                with patch(
                    "ml.rag.retrievers.bq_retriever.try_sql_patterns",
                    return_value=[],
                ):
                    items = retriever.retrieve(
                        "How many people in Ethiopia are in IPC Phase 3 or higher right now?",
                        selected_tables=["fct_food_security"],
                        task_mode="fact_lookup",
                        primary_measures=["food_security_ipc"],
                        geo_country="Ethiopia",
                    )
    nl.assert_not_called()
    assert len(items) == 1
    assert items[0]["metadata"]["status"] == "no_valid_sql"

