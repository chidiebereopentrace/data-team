"""Avocado thread: a place-only follow-up keeps the commodity, measure and price type of earlier turns.

The fake pipeline runs the real ActiveRequest resolution, so the API session round-trip and the
Streamlit in-process loop are exercised end to end without LLM, BigQuery or Qdrant.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ml.rag import session_store
from ml.rag.app import api as api_mod
from ml.rag.chatbot.active_request import resolve_turn, resolved_request_text
from ml.rag.chatbot.qa_run_kwargs import build_run_kwargs

THREAD = [
    "What is the current price of avocado?",
    "I mean the retail price",
    "For kenya",
]


def _fake_run_rag(calls: list[dict[str, Any]]):
    def run(query: str, **kwargs: Any) -> dict[str, Any]:
        turn, persisted = resolve_turn(
            query,
            persisted=kwargs.get("active_request"),
            recent_turns=kwargs.get("recent_turns"),
        )
        calls.append({"query": query, "kwargs": kwargs, "turn": turn})
        return {
            "answer": f"answer to {query}",
            "citations": [],
            "resolved_request": resolved_request_text(query, turn),
            "active_request": persisted.to_dict(),
        }

    return run


@contextmanager
def _fake_trace(**_kwargs: Any):
    class _Handle:
        def update_output(self, _result: dict[str, Any], *, latency_ms: float | None = None) -> None:
            return None

    yield _Handle()


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RAG_API_AUTH", "off")
    monkeypatch.setattr(session_store, "_REDIS_URL", "")
    monkeypatch.setattr(session_store, "_redis_client", None)
    monkeypatch.setattr(api_mod, "rag_trace_context", _fake_trace)
    monkeypatch.setattr(api_mod, "get_current_trace_id", lambda: None)
    monkeypatch.setattr(api_mod, "flush_langfuse", lambda: None)
    monkeypatch.setattr(api_mod, "check_plan_rate_limit", lambda _slug, _req: None)
    session_store.clear_fallback_for_tests()
    return TestClient(api_mod.app)


def _ask(client: TestClient, query: str, sid: str, headers: dict[str, str] | None = None):
    return client.post("/query", json={"query": query, "session_id": sid}, headers=headers or {})


def test_place_only_follow_up_carries_avocado_retail_price(client, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("ml.rag.graph.run_rag", _fake_run_rag(calls))
    for query in THREAD:
        assert _ask(client, query, "avo").status_code == 200

    last = calls[-1]["turn"]
    assert last.value("geographies") == ["Kenya"]
    assert last.slots["geographies"].source == "question"
    assert last.value("commodity") == ["avocado"]
    assert last.slots["commodity"].source == "carried"
    assert last.value("indicator") == "price"
    assert last.value("price_type") == "retail"


def test_unrelated_turn_keeps_the_active_request(client, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("ml.rag.graph.run_rag", _fake_run_rag(calls))
    for query in [THREAD[0], "thanks!", "For kenya"]:
        assert _ask(client, query, "avo2").status_code == 200

    assert calls[1]["turn"].carried() == {}
    assert calls[2]["turn"].value("commodity") == ["avocado"]


def test_sessions_are_isolated_per_owner(client, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("ml.rag.graph.run_rag", _fake_run_rag(calls))
    assert _ask(client, THREAD[0], "shared-id").status_code == 200
    monkeypatch.setattr(api_mod, "_session_owner", lambda _req: "tenant:other")
    assert _ask(client, "For kenya", "shared-id").status_code == 200

    assert calls[1]["kwargs"].get("active_request") is None
    assert calls[1]["turn"].value("commodity") is None


def test_stale_turn_does_not_overwrite_newer_state(client, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("ml.rag.graph.run_rag", _fake_run_rag(calls))
    assert _ask(client, THREAD[0], "race").status_code == 200
    stale = session_store.get_session_blob("local", "race")

    assert _ask(client, "I mean the retail price", "race").status_code == 200
    monkeypatch.setattr(api_mod, "load_session", lambda _owner, _sid: (stale, True))
    assert _ask(client, "what about maize in Ghana", "race").status_code == 200

    saved = session_store.get_session_blob("local", "race")
    assert saved["active_request"]["slots"]["price_type"]["value"] == "retail"
    assert saved["active_request"]["slots"]["commodity"]["value"] == ["avocado"]


def test_idempotency_key_replays_without_rerunning(client, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("ml.rag.graph.run_rag", _fake_run_rag(calls))
    headers = {"Idempotency-Key": "turn-1"}
    first = _ask(client, THREAD[0], "idem", headers)
    second = _ask(client, THREAD[0], "idem", headers)

    assert first.json() == second.json()
    assert len(calls) == 1
    assert len(session_store.get_session_blob("local", "idem")["recent_turns"]) == 2


def test_version_reports_contract_identity_without_secrets(client, monkeypatch) -> None:
    monkeypatch.setenv("RAG_LLM_API_KEY", "sk-secret-value")
    body = client.get("/version").json()

    for key in ("sha", "execution_mode", "catalog_version", "registry_version", "ledger_snapshot", "settings_fingerprint"):
        assert key in body
    assert body["ledger_snapshot"]
    assert "sk-secret-value" not in json.dumps(body)


def test_api_and_streamlit_resolve_the_thread_identically(client, monkeypatch) -> None:
    api_calls: list[dict[str, Any]] = []
    monkeypatch.setattr("ml.rag.graph.run_rag", _fake_run_rag(api_calls))
    for query in THREAD:
        assert _ask(client, query, "parity").status_code == 200

    streamlit_calls: list[dict[str, Any]] = []
    run = _fake_run_rag(streamlit_calls)
    sess: dict[str, Any] = {"recent_turns": []}
    for query in THREAD:
        kwargs = build_run_kwargs(
            news_top_k=5,
            academic_top_k=5,
            bq_top_k=5,
            ota_top_k=5,
            rerank_top_k=5,
            plan_type="",
            category="",
            profile_country="",
            t_start="",
            t_end="",
            prior_summary="",
            prior_recent=sess["recent_turns"],
            active_request=sess.get("active_request"),
        )
        result = run(query, **kwargs)
        sess["active_request"] = result["active_request"]
        sess["recent_turns"] = [
            *sess["recent_turns"],
            {"role": "user", "content": query},
            {"role": "assistant", "content": result["answer"]},
        ]

    for api_call, st_call in zip(api_calls, streamlit_calls, strict=True):
        assert api_call["turn"].to_dict() == st_call["turn"].to_dict()
        assert "plan_type" not in api_call["kwargs"] and "plan_type" not in st_call["kwargs"]
