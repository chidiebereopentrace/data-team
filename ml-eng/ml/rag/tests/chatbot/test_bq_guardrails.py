"""Guardrails: typed warehouse gaps and forbid generic insufficient."""
from __future__ import annotations

from ml.rag.chatbot.bq_gap_messages import forbid_generic_insufficient, warehouse_was_attempted
from ml.rag.chatbot.bq_execute_state import bq_execute_flags
from ml.rag.chatbot.graph import _typed_bq_hard_return


def test_bq_execute_flags_validation_failed() -> None:
    flags = bq_execute_flags(
        [{"sql": "SELECT 1", "status": "validation_failed", "prep_error": "bad literal"}],
        pre_queries=["SELECT 1"],
        usable_bq=False,
    )
    assert flags["structured_bq_validation_failed"] is True
    assert flags["structured_bq_never_executed"] is False


def test_typed_hard_return_timeout() -> None:
    state = {
        "query": "West Africa production 2020",
        "decomposition": {"geography": ["West Africa"], "time_start": "2020-01-01"},
        "bq_sql_plan": {"bq_sql_queries": ["SELECT 1"]},
        "bq_sql_debug": [
            {"sql": "SELECT 1", "status": "timeout", "job_id": "j1", "sql_source": "engine"},
        ],
    }
    flags = bq_execute_flags(
        state["bq_sql_debug"],
        pre_queries=["SELECT 1"],
        usable_bq=False,
    )
    hard = _typed_bq_hard_return(state, flags, bq_debug=state["bq_sql_debug"])
    assert hard is not None
    assert "time limit" in hard["answer"].lower() or "timed out" in hard["answer"].lower()
    assert "don't have enough" not in hard["answer"].lower()


def test_forbid_generic_when_warehouse_attempted() -> None:
    assert forbid_generic_insufficient(
        {"bq_sql_plan": {"bq_sql_queries": ["SELECT 1"], "supervisor_plan": {"forbid_insufficient_without_attempt": True}}}
    )
    assert warehouse_was_attempted({"bq_sql_plan": {"selected_tables": ["fct_production"]}})
