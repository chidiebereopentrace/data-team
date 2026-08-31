"""POST /query include_trace BQ SQL debug fields."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ml.rag.app import api as api_mod


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("RAG_LLM_API_KEY", "sk-test")
    monkeypatch.delenv("RAG_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    return TestClient(api_mod.app)


def _mock_rag_result() -> dict[str, Any]:
    return {
        "answer": "Maize yields in Kenya were stable.",
        "citations": [],
        "decomposition": {"geography": ["Kenya"], "intent": "descriptive"},
        "bq_sql_queries": ["SELECT country_name FROM mart_dev.fct_production LIMIT 10"],
        "bq_sql_debug": [
            {
                "sql": "SELECT country_name FROM mart_dev.fct_production LIMIT 10",
                "status": "ok",
                "sql_source": "pattern",
            }
        ],
        "bq_sql_plan": {
            "selected_tables": ["fct_production"],
            "query_intents": [
                {
                    "goal": "maize yields Kenya",
                    "tables": ["fct_production"],
                    "pattern": "time_series",
                    "subquestion_id": "sq1",
                }
            ],
            "skip_bq": False,
            "rationale": "global_reasoner_slots",
            "slot_path": True,
            "reasoner_job": "trend",
            "extra_internal": "should_not_leak",
        },
        "sql_source": "pattern",
        "bq_cache_hit": False,
        "bq_nl2sql_ms": 120.5,
        "bq_execute_ms": 45.0,
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }


@contextmanager
def _fake_rag_trace(**_kwargs: Any):
    class _Handle:
        def update_output(self, _result: dict[str, Any]) -> None:
            return None

    yield _Handle()


def test_query_include_trace_exposes_bq_sql(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock = _mock_rag_result()

    monkeypatch.setattr("ml.rag.graph.run_rag", lambda query, **kwargs: mock)
    monkeypatch.setattr(api_mod, "rag_trace_context", _fake_rag_trace)
    monkeypatch.setattr(api_mod, "get_current_trace_id", lambda: "trace-sql-1")
    monkeypatch.setattr(api_mod, "flush_langfuse", lambda: None)

    resp = client.post(
        "/query/integrated",
        json={
            "query": "Maize yields in Kenya 2020",
            "include_trace": True,
            "user_profile": {
                "plan_type": "Integrated",
                "category": "Government",
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    trace = body.get("trace")
    assert isinstance(trace, dict)
    assert trace["bq_sql_queries"] == mock["bq_sql_queries"]
    assert trace["bq_sql_debug"][0]["status"] == "ok"
    assert trace["sql_source"] == "pattern"
    assert trace["bq_nl2sql_ms"] == 120.5
    plan = trace.get("bq_sql_plan")
    assert isinstance(plan, dict)
    assert plan.get("selected_tables") == ["fct_production"]
    assert plan.get("slot_path") is True
    assert "extra_internal" not in plan


def test_query_without_trace_omits_sql_fields(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ml.rag.graph.run_rag", lambda query, **kwargs: _mock_rag_result())
    monkeypatch.setattr(api_mod, "rag_trace_context", _fake_rag_trace)
    monkeypatch.setattr(api_mod, "get_current_trace_id", lambda: None)
    monkeypatch.setattr(api_mod, "flush_langfuse", lambda: None)

    resp = client.post(
        "/query/integrated",
        json={"query": "Who are you?", "include_trace": False},
    )
    assert resp.status_code == 200
    assert resp.json().get("trace") is None


def test_trace_bq_sql_debug_rows_caps_length() -> None:
    rows = [{"sql": f"SELECT {i}", "status": "ok"} for i in range(30)]
    result = {"bq_sql_debug": rows}
    capped = api_mod._trace_bq_sql_debug_rows(result)
    assert len(capped) == api_mod._TRACE_BQ_SQL_DEBUG_MAX
