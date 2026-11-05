"""OFIA reasoner guardrails — acceptance tests for post Phase 0–7 hardening."""
from __future__ import annotations

from ml.rag.chatbot.acf_scoring import ACFResult, apply_bq_execute_ceiling
from ml.rag.chatbot.class_engine_runner import run_class_engines
from ml.rag.chatbot.class_supervisor import compile_supervisor_plan
from ml.rag.chatbot.schema_card import card_maturity, load_schema_card
from ml.rag.chatbot.sql_request import build_sql_request_from_facets
from ml.rag.chatbot.sql_compiler import sql_compiler_enabled
from ml.rag.retrievers.web_retriever import warehouse_blocks_web


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


def test_web_blocked_on_empty_when_policy_is_never() -> None:
    state = {
        "bq_sql_plan": {"bq_sql_queries": ["SELECT ipc FROM fs"]},
        "bq_sql_debug": [{"sql": "SELECT ipc FROM fs", "status": "ok", "job_id": "job-1"}],
        "bq_results": [],
    }
    assert warehouse_blocks_web(state)


def test_web_blocked_on_never_executed() -> None:
    state = {
        "bq_sql_plan": {"bq_sql_queries": ["SELECT 1"]},
        "bq_sql_debug": [{"sql": "SELECT 1", "status": "planned"}],
        "bq_results": [],
    }
    assert warehouse_blocks_web(state)


def test_web_blocked_when_planned_intent_never_started_a_job() -> None:
    state = {
        "bq_sql_plan": {
            "plan_source": "class_engine",
            "query_intents": [
                {"goal": "maize price", "tables": ["agg_prices_country_month"]},
            ],
        },
        "bq_results": [],
        "supervisor_plan": {"classes": ["PRC"]},
    }
    assert warehouse_blocks_web(state)


def test_web_allowed_when_unavailable_and_policy_is_not_never(monkeypatch) -> None:
    monkeypatch.setattr(
        "ml.rag.retrievers.web_retriever.resolve_web_policy",
        lambda _state: "if_empty_job",
    )
    state = {
        "bq_sql_plan": {
            "plan_source": "class_engine",
            "query_intents": [{"goal": "maize price"}],
        },
        "bq_results": [],
    }
    assert warehouse_blocks_web(state) is False


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
