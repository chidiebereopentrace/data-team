"""Deployed RAG API access control: API keys, plan routes, trace gating, CORS, errors."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ml.rag import rate_limiter, session_store
from ml.rag.app import api as api_mod
from ml.rag.session_store import clear_fallback_for_tests
from ml.serving.enterprise.metering import get_usage
from ml.serving.enterprise.tenant_registry import hash_api_key, reload_tenants

WEB_KEY = "web-key"
INSPECTOR_KEY = "inspector-key"


@contextmanager
def _fake_rag_trace(**_kwargs: Any):
    class _Handle:
        def update_output(self, _result: dict[str, Any], *, latency_ms: float | None = None) -> None:
            return None

    yield _Handle()


@pytest.fixture()
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def _run_rag(query: str, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {
            "answer": "ok",
            "citations": [],
            "decomposition": {"geography": ["Kenya"]},
            "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }

    monkeypatch.setattr("ml.rag.graph.run_rag", _run_rag)
    monkeypatch.setattr(api_mod, "rag_trace_context", _fake_rag_trace)
    monkeypatch.setattr(api_mod, "get_current_trace_id", lambda: None)
    monkeypatch.setattr(api_mod, "flush_langfuse", lambda: None)
    return seen


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv(
        "ENTERPRISE_TENANTS_JSON",
        json.dumps(
            {
                "tenants": [
                    {
                        "tenant_id": "web",
                        "api_key_sha256": hash_api_key(WEB_KEY),
                        "plan_slug": "free",
                        "allow_trace": False,
                    },
                    {
                        "tenant_id": "inspector",
                        "api_key_sha256": hash_api_key(INSPECTOR_KEY),
                        "plan_slug": "integrated",
                        "allow_trace": True,
                    },
                ]
            }
        ),
    )
    monkeypatch.delenv("RAG_API_AUTH", raising=False)
    monkeypatch.delenv("RAG_DEBUG", raising=False)
    monkeypatch.setattr(session_store, "_REDIS_URL", "")
    monkeypatch.setattr(session_store, "_redis_client", None)
    monkeypatch.setenv("ENTERPRISE_METERING_ENABLED", "1")
    clear_fallback_for_tests()
    rate_limiter._windows.clear()
    reload_tenants()
    yield TestClient(api_mod.app)
    clear_fallback_for_tests()
    rate_limiter._windows.clear()
    monkeypatch.delenv("ENTERPRISE_TENANTS_JSON", raising=False)
    reload_tenants()


def test_query_without_key_is_401(client: TestClient, calls: list) -> None:
    resp = client.post("/query", json={"query": "maize prices"})
    assert resp.status_code == 401
    assert calls == []


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_docs_require_key(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 401


def test_session_delete_requires_key(client: TestClient) -> None:
    assert client.delete("/session/abc").status_code == 401


def test_other_plan_route_is_403(client: TestClient, calls: list) -> None:
    resp = client.post(
        "/query/integrated",
        json={"query": "maize prices"},
        headers={"X-API-Key": WEB_KEY},
    )
    assert resp.status_code == 403
    assert calls == []


def test_bare_query_uses_tenant_plan(client: TestClient, calls: list) -> None:
    resp = client.post("/query", json={"query": "maize prices"}, headers={"X-API-Key": WEB_KEY})
    assert resp.status_code == 200
    assert calls[0]["plan_type"] == "Free"
    assert calls[0]["export_enabled"] is False


def test_trace_stripped_for_tenant_without_allow_trace(client: TestClient, calls: list) -> None:
    resp = client.post(
        "/query/free",
        json={"query": "maize prices", "include_trace": True},
        headers={"X-API-Key": WEB_KEY},
    )
    assert resp.status_code == 200
    assert resp.json()["trace"] is None


def test_trace_returned_for_allow_trace_tenant(client: TestClient, calls: list) -> None:
    resp = client.post(
        "/query/integrated",
        json={"query": "maize prices", "include_trace": True},
        headers={"Authorization": f"Bearer {INSPECTOR_KEY}"},
    )
    assert resp.status_code == 200
    assert resp.json()["trace"]["decomposition"] == {"geography": ["Kenya"]}


def test_usage_is_metered_per_tenant(client: TestClient, calls: list) -> None:
    client.post("/query/free", json={"query": "maize prices"}, headers={"X-API-Key": WEB_KEY})
    usage = get_usage("web")
    assert usage.query_count == 1
    assert usage.total_tokens == 7


def test_cors_default_allows_no_origin(client: TestClient) -> None:
    resp = client.options(
        "/query",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in resp.headers


def test_500_hides_exception_text(
    client: TestClient, calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(query: str, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Access Denied: Table opentrace-prod-5ga4:mart_dev.fct_prices")

    monkeypatch.setattr("ml.rag.graph.run_rag", _boom)
    resp = client.post("/query/free", json={"query": "maize prices"}, headers={"X-API-Key": WEB_KEY})
    assert resp.status_code == 500
    assert "mart_dev" not in resp.json()["detail"]


def test_per_ip_rate_limit(client: TestClient, calls: list, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(rate_limiter._PLAN_LIMITS, "free", 2)
    headers = {"X-API-Key": WEB_KEY}
    for _ in range(2):
        assert client.post("/query/free", json={"query": "q"}, headers=headers).status_code == 200
    resp = client.post("/query/free", json={"query": "q"}, headers=headers)
    assert resp.status_code == 429
