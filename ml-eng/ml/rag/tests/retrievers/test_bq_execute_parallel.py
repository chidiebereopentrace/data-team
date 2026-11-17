"""Parallel BigQuery execution and per-job timeout defaults."""
from __future__ import annotations

import time
from threading import Barrier, Event
from unittest.mock import MagicMock, patch

import pytest

from ml.rag.retrievers.bq_retriever import (
    BQRetriever,
    _bq_execute_parallel,
    _bq_job_timeout_s,
)

_PRODUCTION_WHERE = (
    "production_grain = 'physical' "
    "AND element = 'Production' AND metric = 'production_production_physical'"
)


def _explicit_sql(tag: str) -> str:
    return (
        "SELECT country_iso3, SUM(value) AS total "
        "FROM `proj.mart_dev.fct_production` "
        f"WHERE year = 2022 AND {_PRODUCTION_WHERE} "
        f"GROUP BY country_iso3 LIMIT 10 /* {tag} */"
    )


def _fast_prepare(raw_sql: str, **kwargs):
    limit = kwargs.get("limit") or 10
    validated = f"{raw_sql.rstrip(';')} LIMIT {limit}"
    return validated, None


def _mock_bq_client() -> MagicMock:
    def _slow_result(*_args, **_kwargs):
        time.sleep(0.15)
        return [{"country_iso3": "NGA", "total": 1.0}]

    job = MagicMock()
    job.result.side_effect = _slow_result
    client = MagicMock()
    client.query.return_value = job
    return client


def _retrieve_explicit_batch(
    retriever: BQRetriever,
    client: MagicMock,
    sqls: list[str],
    *, deadline: float | None = None,
) -> list[dict]:
    with patch.object(retriever, "_get_client", return_value=client):
        with patch.object(retriever, "_prepare_sql", side_effect=_fast_prepare):
            return retriever.retrieve(
                "maize production 2022",
                sql=sqls,
                selected_tables=["fct_production"],
                time_start="2022-01-01",
                crop_required=False,
                geography_required=False,
                deadline=deadline,
            )


def test_job_timeout_defaults_12(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_BQ_JOB_TIMEOUT_AGG_S", raising=False)
    monkeypatch.delenv("RAG_BQ_JOB_TIMEOUT_FACT_S", raising=False)
    assert _bq_job_timeout_s("SELECT 1 FROM agg_production_country_year LIMIT 1") == 12.0
    assert _bq_job_timeout_s("SELECT 1 FROM fct_production LIMIT 1") == 12.0


def test_bq_execute_parallel_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_BQ_EXECUTE_PARALLEL", raising=False)
    assert _bq_execute_parallel() is True


def test_execute_parallel_starts_three_jobs_together(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_BQ_EXECUTE_PARALLEL", "on")
    monkeypatch.setenv("RAG_BQ_EXECUTE_PARALLEL_WORKERS", "3")
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    sqls = [_explicit_sql(str(i)) for i in range(3)]
    client = _mock_bq_client()
    started = Barrier(3, timeout=3)

    def result(**_kwargs):
        started.wait()
        return [{"country_iso3": "NGA", "total": 1.0}]

    client.query.return_value.result.side_effect = result
    items = _retrieve_explicit_batch(retriever, client, sqls)

    assert client.query.call_count == 3
    assert len(items) == 3
    assert all(item["metadata"]["status"] == "ok" for item in items)


def test_one_stalled_job_preserves_completed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_BQ_EXECUTE_PARALLEL", "on")
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    client = MagicMock()
    release = Event()

    def submit(sql, **_kwargs):
        job = MagicMock()
        if "/* slow */" in sql:
            job.result.side_effect = lambda **_kwargs: release.wait(5) and []
        else:
            job.result.return_value = [{"country_iso3": "NGA", "total": 1.0}]
        return job

    client.query.side_effect = submit
    try:
        items = _retrieve_explicit_batch(
            retriever, client, [_explicit_sql("slow"), _explicit_sql("finished")],
            deadline=time.perf_counter() + 0.75,
        )
        assert {item["metadata"]["status"] for item in items} == {"ok", "timeout"}
    finally:
        release.set()


def test_execute_parallel_off_is_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_BQ_EXECUTE_PARALLEL", "off")
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    sqls = [_explicit_sql(str(i)) for i in range(3)]
    client = _mock_bq_client()

    started = time.perf_counter()
    with patch("ml.rag.retrievers.bq_retriever.ThreadPoolExecutor") as tp_cls:
        items = _retrieve_explicit_batch(retriever, client, sqls)
        tp_cls.assert_not_called()
    elapsed = time.perf_counter() - started

    assert client.query.call_count == 3
    assert elapsed >= 0.4
    assert len(items) == 3
