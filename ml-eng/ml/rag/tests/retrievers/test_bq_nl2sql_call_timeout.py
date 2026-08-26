"""
Per-call NL2SQL timeout inside the parallel batch (RAG_BQ_NL2SQL_CALL_TIMEOUT_S).

Verifies that one artificially slow table-hint call cannot drag the whole batch past
the configured per-call budget, and that the fast hints' SQL is still returned.
"""
from __future__ import annotations

import time
from unittest import mock

import pytest

from ml.rag.retrievers.bq_retriever import BQRetriever, _nl2sql_call_timeout_s


def _mk_retriever() -> BQRetriever:
    return BQRetriever(project_id="test-project", nl2sql_enabled=True)


def test_nl2sql_call_timeout_default_and_env(monkeypatch) -> None:
    monkeypatch.delenv("RAG_BQ_NL2SQL_CALL_TIMEOUT_S", raising=False)
    assert _nl2sql_call_timeout_s() == 20.0

    monkeypatch.setenv("RAG_BQ_NL2SQL_CALL_TIMEOUT_S", "5")
    assert _nl2sql_call_timeout_s() == 5.0

    monkeypatch.setenv("RAG_BQ_NL2SQL_CALL_TIMEOUT_S", "0.1")
    # Clamped to a sane floor so a misconfigured value cannot make every call time out.
    assert _nl2sql_call_timeout_s() == 2.0

    monkeypatch.setenv("RAG_BQ_NL2SQL_CALL_TIMEOUT_S", "not-a-number")
    assert _nl2sql_call_timeout_s() == 20.0


def test_slow_hint_does_not_block_fast_hints(monkeypatch) -> None:
    """One hint sleeps past the budget; fast hints' SQL must still come back promptly."""
    monkeypatch.setenv("RAG_BQ_NL2SQL_PARALLEL", "on")
    monkeypatch.setenv("RAG_BQ_NL2SQL_PARALLEL_WORKERS", "3")
    monkeypatch.setenv("RAG_BQ_NL2SQL_CALL_TIMEOUT_S", "1")
    monkeypatch.setenv("RAG_BQ_NL2SQL_MODE", "per_hint")

    retriever = _mk_retriever()

    def fake_nl_to_sql_one(question, table_hints=None, **kwargs):
        hint = (table_hints or [""])[0]
        if hint == "slow_table":
            time.sleep(5)  # far beyond the 1s call budget
            return "SELECT 1 FROM slow_table"
        return f"SELECT 1 FROM {hint}"

    monkeypatch.setattr(retriever, "_nl_to_sql_one", fake_nl_to_sql_one)

    t0 = time.perf_counter()
    queries = retriever._nl_to_sql_queries(
        "test question",
        table_hints=["fast_a", "slow_table", "fast_b"],
        selected_tables=["fast_a", "slow_table", "fast_b"],
    )
    elapsed = time.perf_counter() - t0

    # Batch must return close to the 1s budget, not wait for the 5s sleeper.
    assert elapsed < 3.0, f"batch took {elapsed:.2f}s, expected it to respect the 1s call budget"

    # The two fast hints' SQL must still be present.
    joined = " ".join(queries)
    assert "fast_a" in joined
    assert "fast_b" in joined
    # The slow hint's SQL must NOT be present -- it was abandoned before finishing.
    assert "slow_table" not in joined


def test_all_hints_fast_within_budget_unaffected(monkeypatch) -> None:
    """When every call finishes well within budget, behavior is unchanged."""
    monkeypatch.setenv("RAG_BQ_NL2SQL_PARALLEL", "on")
    monkeypatch.setenv("RAG_BQ_NL2SQL_PARALLEL_WORKERS", "3")
    monkeypatch.setenv("RAG_BQ_NL2SQL_CALL_TIMEOUT_S", "20")
    monkeypatch.setenv("RAG_BQ_NL2SQL_MODE", "per_hint")

    retriever = _mk_retriever()

    def fake_nl_to_sql_one(question, table_hints=None, **kwargs):
        hint = (table_hints or [""])[0]
        return f"SELECT 1 FROM {hint}"

    monkeypatch.setattr(retriever, "_nl_to_sql_one", fake_nl_to_sql_one)

    queries = retriever._nl_to_sql_queries(
        "test question",
        table_hints=["a", "b", "c"],
        selected_tables=["a", "b", "c"],
    )
    joined = " ".join(queries)
    assert "a" in joined and "b" in joined and "c" in joined
