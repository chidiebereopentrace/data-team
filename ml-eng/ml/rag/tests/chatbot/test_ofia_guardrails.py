"""OFIA reasoner guardrails — acceptance tests for post Phase 0–7 hardening."""
from __future__ import annotations

from ml.rag.chatbot.acf_scoring import ACFResult, apply_bq_execute_ceiling
from ml.rag.chatbot.bq_execute_state import bq_execute_flags
from ml.rag.chatbot.bq_gap_messages import (
    has_usable_narrative_context,
    should_hard_return_bq_gap,
    warehouse_blocks_web,
)
from ml.rag.chatbot.class_engine_runner import run_class_engines
from ml.rag.chatbot.class_supervisor import compile_supervisor_plan
from ml.rag.chatbot.graph import _typed_bq_hard_return
from ml.rag.chatbot.schema_card import card_maturity, load_schema_card
from ml.rag.chatbot.sql_request import build_sql_request_from_facets
from ml.rag.chatbot.sql_compiler import sql_compiler_enabled


def test_dual_flag_compiler_wins_for_supervisor(monkeypatch) -> None:
    monkeypatch.setenv("RAG_SLOT_REASONER", "on")
    monkeypatch.setenv("RAG_SQL_COMPILER", "1")
    assert sql_compiler_enabled()
    q = "West Africa agricultural activities country by country 2020"
    dec = {"geography": ["West Africa"], "time_start": "2020-01-01", "time_end": "2020-12-31"}
    sp = compile_supervisor_plan(q, decomposition=dec)
    assert sp.classes
    assert "PROD" in sp.classes
    results = run_class_engines(q, supervisor_plan=sp, facets=dec)
    assert results
    assert any(r.class_code == "PROD" and r.status == "planned" and r.bind_contract for r in results)


def test_vectors_only_skips_hard_return() -> None:
    exec_flags = bq_execute_flags(
        [{"sql": "SELECT 1", "status": "ok", "job_id": "j1"}],
        pre_queries=["SELECT 1"],
        usable_bq=False,
    )
    context = [{"source": "public_report", "content": "FEWS outlook lean season"}]
    assert has_usable_narrative_context(context)
    assert not should_hard_return_bq_gap(
        exec_flags=exec_flags,
        pre_queries=["SELECT 1"],
        usable_bq=False,
        context_items=context,
        is_numeric_job=True,
    )


def test_outlook_empty_bq_skips_numeric_hard_return() -> None:
    exec_flags = bq_execute_flags(
        [{"sql": "SELECT 1", "status": "ok", "job_id": "j1"}],
        pre_queries=["SELECT 1"],
        usable_bq=False,
    )
    assert exec_flags["structured_bq_empty"]
    assert not should_hard_return_bq_gap(
        exec_flags=exec_flags,
        pre_queries=["SELECT 1"],
        usable_bq=False,
        context_items=[],
        is_numeric_job=False,
    )


def test_web_allowed_on_empty_with_job_id() -> None:
    state = {
        "bq_sql_plan": {"bq_sql_queries": ["SELECT ipc FROM fs"]},
        "bq_sql_debug": [{"sql": "SELECT ipc FROM fs", "status": "ok", "job_id": "job-1"}],
        "bq_results": [],
    }
    assert not warehouse_blocks_web(state)


def test_web_blocked_on_never_executed() -> None:
    state = {
        "bq_sql_plan": {"bq_sql_queries": ["SELECT 1"]},
        "bq_sql_debug": [{"sql": "SELECT 1", "status": "planned"}],
        "bq_results": [],
    }
    assert warehouse_blocks_web(state)


def test_web_blocked_on_timeout_without_job_id() -> None:
    state = {
        "bq_sql_plan": {"bq_sql_queries": ["SELECT 1"]},
        "bq_sql_debug": [{"sql": "SELECT 1", "status": "timeout"}],
        "bq_results": [],
    }
    assert warehouse_blocks_web(state)


def test_acf_partial_panel_not_no_evidence() -> None:
    base = ACFResult(
        band="moderate",
        band_label="Moderate confidence",
        score=65,
        explanation="cited rows",
        note="cited rows",
        components={"coverage": 0.8},
    )
    capped = apply_bq_execute_ceiling(
        base,
        {"structured_bq_timed_out": True},
        usable_bq=True,
        bq_sql_debug=[
            {"sql": "SELECT 1", "status": "ok", "job_id": "j1"},
            {"sql": "SELECT 2", "status": "timeout", "job_id": "j2"},
        ],
    )
    assert capped.applied_ceiling == "partial_panel"
    assert capped.score > 35
    assert capped.band != "no_evidence"


def test_acf_full_timeout_when_no_usable_bq() -> None:
    base = ACFResult(
        band="moderate",
        band_label="Moderate confidence",
        score=65,
        explanation="test",
        note="test",
    )
    capped = apply_bq_execute_ceiling(
        base,
        {"structured_bq_timed_out": True},
        usable_bq=False,
        bq_sql_debug=[{"sql": "SELECT 1", "status": "timeout", "job_id": "j1"}],
    )
    assert capped.applied_ceiling == "bq_timeout"
    assert capped.score <= 35


def test_schema_card_maturity_ready_and_stub() -> None:
    fs = load_schema_card("FS")
    assert card_maturity(fs)["status"] in ("ready", "partial")
    assert card_maturity(fs)["column_count"] >= 3
    stub = {"class": "X", "columns": {}}
    assert card_maturity(stub)["status"] == "stub"


def test_build_sql_request_from_facets_panel_shape() -> None:
    card = load_schema_card("PROD") or {}
    req = build_sql_request_from_facets(
        class_code="PROD",
        table_id="agg_production_country_year",
        query="West Africa agricultural activities country by country",
        facets={"geography": ["West Africa"], "time_start": "2015-01-01", "time_end": "2020-12-31"},
        card=card,
        iso_list=["GHA", "NGA", "MLI"],
    )
    assert req.shape == "panel"
    assert len(req.geos) == 3
    assert req.year_start == 2015


def test_typed_hard_return_still_returns_gap_message() -> None:
    state = {
        "query": "West Africa production 2020",
        "decomposition": {"geography": ["West Africa"], "time_start": "2020-01-01"},
        "bq_sql_plan": {"bq_sql_queries": ["SELECT 1"]},
        "bq_sql_debug": [
            {"sql": "SELECT 1", "status": "timeout", "job_id": "j1", "sql_source": "engine"},
        ],
    }
    flags = bq_execute_flags(state["bq_sql_debug"], pre_queries=["SELECT 1"], usable_bq=False)
    hard = _typed_bq_hard_return(state, flags, bq_debug=state["bq_sql_debug"])
    assert hard is not None
    assert "time limit" in hard["answer"].lower() or "timed out" in hard["answer"].lower()
