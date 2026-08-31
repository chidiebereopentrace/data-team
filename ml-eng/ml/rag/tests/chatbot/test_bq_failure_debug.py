"""Unit tests for synthetic BQ failure rows in the graph."""

from __future__ import annotations

from ml.rag.chatbot.graph import bq_failure_debug_row


class _FakeRetriever:
    _last_nl2sql_raws = ["(empty LLM response)", "SELECT bad"]
    last_sql_source = "nl2sql"


def test_bq_failure_debug_row_includes_nl2sql_raw_and_source() -> None:
    row = bq_failure_debug_row(
        _FakeRetriever(),  # type: ignore[arg-type]
        status="bq_timeout",
        prep_error="BQ retrieve exceeded 15s timeout",
    )
    assert row["status"] == "bq_timeout"
    assert row["sql"] == ""
    assert "timeout" in row["prep_error"]
    assert "empty LLM" in row["nl2sql_raw"]
    assert row["sql_source"] == "nl2sql"
