"""Deployed RAG API access control: API keys, plan routes, trace gating, CORS, errors."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from threading import Event
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from ml.rag import session_store
from ml.rag.app import api as api_mod
from ml.rag.session_store import clear_fallback_for_tests
from ml.serving.enterprise.metering import get_usage
from ml.serving.enterprise.tenant_registry import hash_api_key, reload_tenants

WEB_KEY = "web-key"
WEB_APP_KEY = "web-app-key"
AGRI_KEY = "agri-key"
INSPECTOR_KEY = "inspector-key"
RESEARCH_KEY = "research-key"


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
                        "tenant_id": "web-frontend",
                        "api_key_sha256": hash_api_key(WEB_APP_KEY),
                        "plan_slug": "integrated",
                        "allow_trace": False,
                        "rate_limit_rpm": 0,
                    },
                    {
                        "tenant_id": "olam",
                        "api_key_sha256": hash_api_key(AGRI_KEY),
                        "plan_slug": "agribusinesses",
                        "allow_trace": False,
                    },
                    {
                        "tenant_id": "inspector",
                        "api_key_sha256": hash_api_key(INSPECTOR_KEY),
                        "plan_slug": "integrated",
                        "allow_trace": True,
                    },
                    {
                        "tenant_id": "research",
                        "api_key_sha256": hash_api_key(RESEARCH_KEY),
                        "plan_slug": "research",
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
    reload_tenants()
    yield TestClient(api_mod.app)
    clear_fallback_for_tests()
    monkeypatch.delenv("ENTERPRISE_TENANTS_JSON", raising=False)
    reload_tenants()


def test_query_without_key_is_401(client: TestClient, calls: list) -> None:
    resp = client.post("/query", json={"query": "maize prices"})
    assert resp.status_code == 401
    assert calls == []


@pytest.mark.parametrize(
    ("plan", "category", "country"),
    [
        ("Integrated", "", ""), ("", "Government", ""), ("", "", "Nigeria"),
        ("Integrated", "", "Nigeria"), ("", "Farmers", "Nigeria"),
        ("", "", ""), ("Integrated", "Government", "Nigeria"),
    ],
)
def test_inspector_optional_profile_choices_reach_authenticated_api(client, calls, monkeypatch, plan, category, country):
    from urllib.parse import urlsplit

    from ml.rag.chatbot.qa_run_kwargs import build_run_kwargs
    from ml.rag.chatbot.streamlit_inspector import query_via_http_api

    kwargs = build_run_kwargs(
        news_top_k=10, academic_top_k=10, bq_top_k=8, ota_top_k=10, rerank_top_k=14,
        plan_type=plan, category=category, profile_country=country, t_start="", t_end="",
        prior_summary="", prior_recent=[],
    )

    def post(url, *, json, headers, timeout):
        return client.post(urlsplit(url).path, json=json, headers=headers)

    monkeypatch.setenv("RAG_API_KEY", INSPECTOR_KEY)
    monkeypatch.setattr("ml.rag.chatbot.streamlit_inspector.requests.post", post)
    result = query_via_http_api("http://testserver", "hi", kwargs=kwargs)
    assert result["answer"] == "ok"
    assert calls[0]["plan_type"] == (plan or "Integrated")
    assert calls[0].get("category") == (category or None)


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_running_query_does_not_block_liveness(client: TestClient, calls: list, monkeypatch) -> None:
    entered = Event()
    release = Event()

    def slow_run(*_args, **_kwargs):
        entered.set()
        release.wait(3)
        return {"answer": "finished", "citations": []}

    monkeypatch.setattr("ml.rag.graph.run_rag", slow_run)

    async def probe() -> None:
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            query = asyncio.create_task(api.post("/query", headers={"X-API-Key": WEB_KEY}, json={"query": "maize yields"}))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                assert (await api.get("/health")).status_code == 200
                assert not query.done()
            finally:
                release.set()
            assert (await query).status_code == 200

    asyncio.run(probe())


def test_docs_are_public(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
        assert client.get(path).status_code == 200
    assert client.post("/query", json={"query": "hi"}).status_code == 401


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


def test_optional_profile_plan_cannot_escalate_access(client: TestClient, calls: list) -> None:
    resp = client.post(
        "/query/free",
        json={"query": "hi", "user_profile": {"plan_type": "Integrated", "category": None}},
        headers={"X-API-Key": WEB_KEY},
    )
    assert resp.status_code == 200
    assert calls[0]["plan_type"] == "Free"
    assert calls[0]["export_enabled"] is False


@pytest.mark.parametrize("field", ["category", "plan_type"])
def test_optional_profile_fields_still_reject_unknown_values(client: TestClient, calls: list, field: str) -> None:
    resp = client.post(
        "/query/integrated", json={"query": "hi", "user_profile": {field: "invalid-value"}},
        headers={"X-API-Key": INSPECTOR_KEY},
    )
    assert resp.status_code == 422
    assert calls == []


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


def test_per_user_rate_limit(client: TestClient, calls: list, monkeypatch: pytest.MonkeyPatch) -> None:
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
                        "rate_limit_rpm": 2,
                    }
                ]
            }
        ),
    )
    reload_tenants()
    clear_fallback_for_tests()
    headers = {"X-API-Key": WEB_KEY}
    for _ in range(2):
        assert client.post(
            "/query/free", json={"query": "q", "user_id": "alice"}, headers=headers
        ).status_code == 200
    limited = client.post("/query/free", json={"query": "q", "user_id": "alice"}, headers=headers)
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"
    assert limited.headers["x-ratelimit-limit"] == "2"
    assert client.post(
        "/query/free", json={"query": "q", "user_id": "bob"}, headers=headers
    ).status_code == 200
    for _ in range(2):
        assert client.post("/query/free", json={"query": "q"}, headers=headers).status_code == 200
    anonymous = client.post("/query/free", json={"query": "q"}, headers=headers)
    assert anonymous.status_code == 429


def _web_app_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": "maize in Kenya",
        "user_id": "acct-1",
        "session_id": "conv-1",
        "user_profile": {"plan_type": "Integrated", "category": "Farmers", "country": "NGA"},
    }
    body.update(overrides)
    return body


def test_web_frontend_rejects_incomplete_body(client: TestClient, calls: list) -> None:
    headers = {"X-API-Key": WEB_APP_KEY}
    cases = [
        _web_app_body(session_id=None),
        _web_app_body(user_id=None),
        _web_app_body(user_profile=None),
        _web_app_body(user_profile={"plan_type": "Integrated"}),
        _web_app_body(user_profile={"category": "Farmers"}),
        _web_app_body(chat_history=[{"role": "user", "content": "earlier"}]),
        _web_app_body(conversation_history=[{"role": "user", "content": "earlier"}]),
    ]
    for body in cases:
        resp = client.post("/query/integrated", json=body, headers=headers)
        assert resp.status_code == 422, body
        assert "detail" in resp.json()
    assert calls == []


def test_web_frontend_session_is_loaded_on_the_next_turn(client: TestClient, calls: list) -> None:
    headers = {"X-API-Key": WEB_APP_KEY}
    first = client.post("/query/integrated", json=_web_app_body(), headers=headers)
    assert first.status_code == 200, first.text
    assert first.json()["session_found"] is False
    assert first.json()["session_id"] == "conv-1"
    assert calls[0]["session_id"] == "conv-1"
    assert calls[0]["category"] == "Farmers"
    assert calls[0]["user_profile"]["plan_type"] == "Integrated"
    second = client.post(
        "/query/integrated",
        json=_web_app_body(query="and last year"),
        headers=headers,
    )
    assert second.status_code == 200, second.text
    assert second.json()["session_found"] is True
    assert second.json()["session_id"] == "conv-1"
    assert calls[1]["recent_turns"]


def test_agribusiness_tenant_may_omit_session_and_profile(client: TestClient, calls: list) -> None:
    resp = client.post(
        "/query/agribusinesses",
        json={"query": "cocoa in Ghana"},
        headers={"X-API-Key": AGRI_KEY},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["session_id"]
    assert calls[0]["plan_type"] == "Agribusinesses"


def test_explicit_export_intent_is_forwarded(client: TestClient, calls: list) -> None:
    resp = client.post(
        "/query/free",
        json={"query": "What are maize yields in Kenya?", "export_intent": "csv"},
        headers={"X-API-Key": WEB_KEY},
    )
    assert resp.status_code == 200, resp.text
    assert calls[0]["export_intent"] == "csv"


def test_unknown_export_intent_is_422(client: TestClient, calls: list) -> None:
    resp = client.post(
        "/query/free",
        json={"query": "What are maize yields in Kenya?", "export_intent": "spreadsheet"},
        headers={"X-API-Key": WEB_KEY},
    )
    assert resp.status_code == 422
    assert calls == []


@pytest.mark.parametrize("path", ["/query/research", "/query"])
def test_research_tenant_gets_paid_plan_and_metering(client, calls, path):
    response = client.post(path, headers={"X-API-Key": RESEARCH_KEY}, json={
        "query": "Compare yields and rainfall with methodological limitations",
        "user_profile": {"plan_type": "Integrated", "category": "Research"},
        "include_trace": True,
    })
    assert response.status_code == 200, response.text
    assert calls[0]["plan_type"] == "Research" and calls[0]["export_enabled"] is True
    assert calls[0]["category"] == "Research"
    assert response.json()["trace"] is not None and response.json()["session_id"]
    assert get_usage("research").query_count == 1


@pytest.mark.parametrize("key,route,expected", [
    (WEB_KEY, "/query/research", 403),
    (RESEARCH_KEY, "/query/integrated", 403),
    (INSPECTOR_KEY, "/query/research", 200),
])
def test_research_route_respects_tenant_entitlements(client, calls, key, route, expected):
    response = client.post(route, headers={"X-API-Key": key}, json={"query": "hi"})
    assert response.status_code == expected
    assert len(calls) == (1 if expected == 200 else 0)


@pytest.mark.parametrize("route,key,plan,exports", [
    ("free", WEB_KEY, "Free", False),
    ("integrated", INSPECTOR_KEY, "Integrated", True),
])
def test_research_category_does_not_escalate_plan(client, calls, route, key, plan, exports):
    response = client.post(f"/query/{route}", headers={"X-API-Key": key}, json={
        "query": "Evaluate agricultural evidence", "user_profile": {"plan_type": "Research", "category": "Research"},
    })
    assert response.status_code == 200, response.text
    assert calls[0]["category"] == "Research" and calls[0]["plan_type"] == plan
    assert calls[0]["export_enabled"] is exports


@pytest.mark.parametrize("formats,expected", [("CSV", {"csv"}), ("GeoJSON", {"geojson"}), ("CSV and GeoJSON", {"csv", "geojson"})])
def test_research_api_returns_real_export_artifacts(client, calls, monkeypatch, formats, expected):
    from ml.rag.chatbot.artifact_storage import mime_type_for_filename
    from ml.rag.chatbot.export_intent import detect_export_intent
    from ml.rag.chatbot.graph import node_export

    uploaded = {}

    def upload(data, filename):
        uploaded[filename] = data
        return {"id": "test-" + filename, "filename": filename, "mime_type": mime_type_for_filename(filename),
                "url": "https://exports.example/" + filename, "byte_size": len(data)}

    monkeypatch.setattr("ml.rag.chatbot.export_runner.upload_artifact", upload)

    def run(query, **kwargs):
        state = {
            **kwargs, "query": query, "answer": "Verified observations.",
            "export_intent": detect_export_intent(query),
            "bq_results": [{"source": "bigquery", "metadata": {"raw_row": {
                "country_name": "Kenya", "value": 2.5, "unit": "t/ha", "latitude": -1, "longitude": 36,
            }}}],
            "citations": [{"id": 1, "kind": "bigquery", "text": "Verified yield observation"}],
        }
        return {**state, **node_export(state)}

    monkeypatch.setattr("ml.rag.graph.run_rag", run)
    response = client.post("/query/research", headers={"X-API-Key": RESEARCH_KEY}, json={
        "query": f"Save soil observations as {formats}", "user_profile": {"plan_type": "Research", "category": "Research"},
    })
    assert response.status_code == 200, response.text
    artifacts = response.json()["artifacts"]
    assert {a["kind"] for a in artifacts} == expected
    for artifact in artifacts:
        data = uploaded[artifact["filename"]]
        assert artifact["citation_ids"] == [1] and artifact["byte_size"] == len(data)
        if artifact["kind"] == "geojson":
            assert artifact["mime_type"] == "application/geo+json"
            assert json.loads(data)["features"][0]["geometry"]["coordinates"] == [36, -1]
        else:
            assert b"t/ha" in data and b"2.5" in data


def test_research_catalog_is_in_openapi_and_root(client):
    assert client.get("/").json()["plan_routes"]["research"] == "POST /query/research"
    spec = client.get("/openapi.json", headers={"X-API-Key": RESEARCH_KEY}).json()
    assert "/query/research" in spec["paths"]
    profile = spec["components"]["schemas"]["UserProfile"]["properties"]
    for field in ("plan_type", "category"):
        variants = profile[field].get("anyOf", [profile[field]])
        assert any("Research" in variant.get("enum", []) for variant in variants)
    assert "geojson" in spec["components"]["schemas"]["ArtifactItem"]["properties"]["kind"]["enum"]
