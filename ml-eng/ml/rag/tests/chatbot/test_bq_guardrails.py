"""Guardrails: warehouse execution flags and the single gap exit."""
from __future__ import annotations

from ml.rag.chatbot.answer_outcome import decide_outcome, gap_answer
from ml.rag.chatbot.bq_execute_state import bq_execute_flags, warehouse_was_attempted
from ml.rag.chatbot.turn_contract import TurnContract


def test_bq_execute_flags_validation_failed() -> None:
    flags = bq_execute_flags(
        [{"sql": "SELECT 1", "status": "validation_failed", "prep_error": "bad literal"}],
        pre_queries=["SELECT 1"],
        usable_bq=False,
    )
    assert flags["structured_bq_validation_failed"] is True
    assert flags["structured_bq_never_executed"] is False


def test_warehouse_was_attempted_from_plan() -> None:
    assert warehouse_was_attempted({"bq_sql_plan": {"selected_tables": ["fct_production"]}})


def test_outcome_kinds() -> None:
    row = {"content": "[Structured data] 10 t", "_context_kind": "bigquery"}
    doc = {"content": "FEWS outlook lean season", "source": "public_report"}
    assert decide_outcome([row]).kind == "rows"
    assert decide_outcome([row, doc]).kind == "rows_and_documents"
    assert decide_outcome([doc]).kind == "documents_only"
    assert decide_outcome([]).is_gap


def test_gap_answer_names_place_indicator_and_period_without_internals() -> None:
    answer = gap_answer(
        "maize production in Kenya in 2031",
        {"geography": ["Kenya"], "time_start": "2031-01-01", "time_end": "2031-12-31"},
        TurnContract(measure_id="production"),
    )
    assert "Kenya" in answer
    assert "2031" in answer
    assert "production" in answer
    for internal in ("compiler", "validation", "SELECT", "mart_"):
        assert internal not in answer
