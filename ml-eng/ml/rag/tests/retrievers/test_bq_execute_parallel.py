"""Parallel BigQuery execution and per-job timeout defaults."""
from __future__ import annotations

import time
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


def _executor_that_runs_inline() -> MagicMock:
    pool = MagicMock()

    def submit(fn, *args, **kwargs):
        fut = MagicMock()
        fut.result.side_effect = lambda: fn(*args, **kwargs)
        return fut

    pool.submit.side_effect = submit
    pool.__enter__.return_value = pool
    return pool


def _retrieve_explicit_batch(
    retriever: BQRetriever,
    client: MagicMock,
    sqls: list[str],
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
            )


def test_job_timeout_defaults_12(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_BQ_JOB_TIMEOUT_AGG_S", raising=False)
    monkeypatch.delenv("RAG_BQ_JOB_TIMEOUT_FACT_S", raising=False)
    assert _bq_job_timeout_s("SELECT 1 FROM agg_production_country_year LIMIT 1") == 12.0
    assert _bq_job_timeout_s("SELECT 1 FROM fct_production LIMIT 1") == 12.0


def test_bq_execute_parallel_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_BQ_EXECUTE_PARALLEL", raising=False)
    assert _bq_execute_parallel() is True


def test_execute_parallel_uses_thread_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_BQ_EXECUTE_PARALLEL", "on")
    monkeypatch.setenv("RAG_BQ_EXECUTE_PARALLEL_WORKERS", "3")
    retriever = BQRetriever(project_id="proj", nl2sql_enabled=False)
    sqls = [_explicit_sql(str(i)) for i in range(3)]
    client = _mock_bq_client()
    pool = _executor_that_runs_inline()

    with patch("ml.rag.retrievers.bq_retriever.ThreadPoolExecutor", return_value=pool) as tp_cls:
        items = _retrieve_explicit_batch(retriever, client, sqls)

    tp_cls.assert_called_once()
    assert tp_cls.call_args.kwargs["max_workers"] == 3
    assert client.query.call_count == 3
    assert len(items) == 3


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
