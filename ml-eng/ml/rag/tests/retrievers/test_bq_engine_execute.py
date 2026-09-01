"""Engine execute-only BQ path, job telemetry, and debug reconciliation."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.rag.chatbot.bq_execute_state import bq_execute_flags
from ml.rag.chatbot.graph import reconcile_engine_bq_debug
from ml.rag.retrievers.bq_retriever import BQRetriever

_PRODUCTION_WHERE = (
    "production_grain = 'physical' "
    "AND element = 'Production' AND metric = 'production_production_physical'"
)


def _engine_sql(tag: str) -> str:
    return (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `proj.mart_dev.fct_production` "
        f"WHERE year = 2022 AND {_PRODUCTION_WHERE} "
        f"GROUP BY country_iso3 LIMIT 10 /* {tag} */"
    )


def _mock_bq_client(*, job_id: str = "job-abc") -> MagicMock:
    job = MagicMock()
    job.job_id = job_id
    job.total_bytes_processed = 4096
    job.total_bytes_billed = 4096
    job.result.return_value = [{"country_iso3": "NGA", "total": 1.0}]
    client = MagicMock()
    client.query.return_value = job
    return client


def test_engine_execute_only_skips_nl2sql(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=True)
    nl2sql = MagicMock(side_effect=AssertionError("nl2sql should not run"))
    monkeypatch.setattr(retriever, "_nl_to_sql_queries", nl2sql)
    monkeypatch.setattr(
        "ml.rag.retrievers.bq_retriever.try_sql_template",
        MagicMock(side_effect=AssertionError("template should not run")),
    )
    monkeypatch.setattr(
        "ml.rag.retrievers.bq_retriever.try_sql_patterns",
        MagicMock(side_effect=AssertionError("pattern should not run")),
    )
    client = _mock_bq_client()

    def _fast_prepare(raw_sql: str, **kwargs):
        limit = kwargs.get("limit") or 10
        return f"{raw_sql.rstrip(';')} LIMIT {limit}", None

    with patch.object(retriever, "_get_client", return_value=client):
        with patch.object(retriever, "_prepare_sql", side_effect=_fast_prepare):
            retriever.retrieve(
                "maize production 2022",
                sql=[_engine_sql("one")],
                engine_execute_only=True,
                selected_tables=["fct_production"],
                crop_required=False,
                geography_required=False,
            )

    nl2sql.assert_not_called()
    assert client.query.called


def test_engine_execute_records_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    client = _mock_bq_client(job_id="job-telemetry-1")

    def _fast_prepare(raw_sql: str, **kwargs):
        limit = kwargs.get("limit") or 10
        return f"{raw_sql.rstrip(';')} LIMIT {limit}", None

    with patch.object(retriever, "_get_client", return_value=client):
        with patch.object(retriever, "_prepare_sql", side_effect=_fast_prepare):
            items = retriever.retrieve(
                "maize production 2022",
                sql=[_engine_sql("telemetry")],
                engine_execute_only=True,
                selected_tables=["fct_production"],
                crop_required=False,
                geography_required=False,
            )

    assert items
    meta = items[0]["metadata"]
    assert meta.get("job_id") == "job-telemetry-1"
    assert meta.get("status") == "ok"
    assert meta.get("row_count") == 1
    assert meta.get("bq_ms") is not None
    assert meta.get("bytes_processed") == 4096
    assert meta.get("sql_source") == "engine"


def test_node_timeout_no_fourth_empty_row() -> None:
    sqls = [_engine_sql("a"), _engine_sql("b"), _engine_sql("c")]
    pre_debug = [{"sql": s, "status": "ready", "sql_source": "engine"} for s in sqls]
    queries, debug = reconcile_engine_bq_debug(
        execute_debug=[],
        pre_queries=sqls,
        pre_debug=pre_debug,
        node_timed_out=True,
        bq_timeout=25.0,
    )
    assert len(debug) == 3
    assert len(queries) == 3
    assert all(str(row.get("sql") or "").strip() for row in debug)
    assert all(row.get("status") == "timeout" for row in debug)
    assert not any(str(row.get("sql") or "").strip() == "" for row in debug)


def test_no_planned_in_final_debug() -> None:
    sqls = [_engine_sql("a"), _engine_sql("b")]
    execute_debug = [
        {
            "sql": sqls[0],
            "status": "ok",
            "job_id": "j1",
            "row_count": 2,
            "sql_source": "engine",
        },
        {
            "sql": sqls[1],
            "status": "ok",
            "job_id": "j2",
            "row_count": 1,
            "sql_source": "engine",
        },
    ]
    pre_debug = [{"sql": s, "status": "ready", "sql_source": "engine"} for s in sqls]
    _, debug = reconcile_engine_bq_debug(
        execute_debug=execute_debug,
        pre_queries=sqls,
        pre_debug=pre_debug,
        node_timed_out=False,
        bq_timeout=25.0,
    )
    statuses = {str(row.get("status") or "") for row in debug}
    assert "planned" not in statuses
    assert "ready" not in statuses
    assert statuses == {"ok"}


def test_bq_execute_flags_never_executed() -> None:
    flags = bq_execute_flags(
        [{"sql": "SELECT 1", "status": "timeout", "sql_source": "engine"}],
        pre_queries=["SELECT 1"],
        usable_bq=False,
    )
    assert flags["structured_bq_never_executed"] is True
    assert flags["structured_bq_timed_out"] is False


def test_bq_execute_flags_validation_failed() -> None:
    flags = bq_execute_flags(
        [{"sql": "SELECT 1", "status": "validation_failed", "prep_error": "bad"}],
        pre_queries=["SELECT 1"],
        usable_bq=False,
    )
    assert flags["structured_bq_validation_failed"] is True
    assert flags["structured_bq_never_executed"] is False


def test_bq_execute_flags_timed_out() -> None:
    flags = bq_execute_flags(
        [{"sql": "SELECT 1", "status": "timeout", "job_id": "j1", "sql_source": "engine"}],
        pre_queries=["SELECT 1"],
        usable_bq=False,
    )
    assert flags["structured_bq_timed_out"] is True
    assert flags["structured_bq_never_executed"] is False
