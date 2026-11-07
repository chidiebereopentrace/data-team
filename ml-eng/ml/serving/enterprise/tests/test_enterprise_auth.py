"""Tests for the tenant registry and API key middleware."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ml.rag import session_store
from ml.rag.session_store import clear_fallback_for_tests
from ml.serving.enterprise.auth import EnterpriseAuthMiddleware, extract_api_key
from ml.serving.enterprise.tenant_registry import (
    enterprise_auth_mode,
    hash_api_key,
    load_tenants,
    reload_tenants,
    resolve_tenant,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    tenants_file = tmp_path / "tenants.json"
    tenants_file.write_text(
        json.dumps(
            {
                "tenants": [
                    {
                        "tenant_id": "test-tenant",
                        "name": "Test Tenant",
                        "api_key_sha256": hash_api_key("test-key-123"),
                        "plan_slug": "agribusinesses",
                        "environment": "sandbox",
                        "monthly_query_quota": 2,
                        "monthly_token_quota": 100,
                        "rate_limit_rpm": 10,
                        "active": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENTERPRISE_TENANTS_PATH", str(tenants_file))
    monkeypatch.delenv("ENTERPRISE_TENANTS_JSON", raising=False)
    monkeypatch.delenv("RAG_API_AUTH", raising=False)
    monkeypatch.setenv("ENTERPRISE_METERING_ENABLED", "1")
    monkeypatch.setattr(session_store, "_REDIS_URL", "")
    monkeypatch.setattr(session_store, "_redis_client", None)
    clear_fallback_for_tests()
    reload_tenants()
    yield
    clear_fallback_for_tests()
    reload_tenants()


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(EnterpriseAuthMiddleware)

    @app.post("/query/agribusinesses")
    async def agri():
        return {"ok": True}

    @app.post("/query/government")
    async def gov():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def test_auth_required_by_default():
    assert enterprise_auth_mode() == "required"


def test_resolve_tenant_by_api_key():
    tenant = resolve_tenant("test-key-123")
    assert tenant is not None
    assert tenant.tenant_id == "test-tenant"
    assert tenant.plan_slug == "agribusinesses"
    assert tenant.allow_trace is False
    assert resolve_tenant("wrong-key") is None


def test_registry_holds_only_key_digests():
    tenants = reload_tenants()
    assert "test-key-123" not in tenants
    assert hash_api_key("test-key-123") in tenants


def test_registry_rejects_non_digest_keys(monkeypatch):
    monkeypatch.setenv(
        "ENTERPRISE_TENANTS_JSON",
        json.dumps(
            {"tenants": [{"tenant_id": "t", "api_key_sha256": "plaintext", "plan_slug": "free"}]}
        ),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_tenants()
    monkeypatch.delenv("ENTERPRISE_TENANTS_JSON")


def test_tenant_plan_authorization():
    tenant = resolve_tenant("test-key-123")
    assert tenant is not None
    assert tenant.allows_plan_slug("agribusinesses")
    assert not tenant.allows_plan_slug("government")


def test_extract_api_key_from_headers():
    class _Req:
        headers = {"x-api-key": "abc"}

    assert extract_api_key(_Req()) == "abc"

    class _Bearer:
        headers = {"authorization": "Bearer xyz"}

    assert extract_api_key(_Bearer()) == "xyz"


def test_auth_middleware_blocks_missing_key():
    response = TestClient(_app()).post("/query/agribusinesses")
    assert response.status_code == 401


def test_auth_middleware_blocks_wrong_key():
    response = TestClient(_app()).post("/query/agribusinesses", headers={"X-API-Key": "nope"})
    assert response.status_code == 401


def test_auth_middleware_allows_own_plan():
    response = TestClient(_app()).post(
        "/query/agribusinesses", headers={"X-API-Key": "test-key-123"}
    )
    assert response.status_code == 200


def test_auth_middleware_forbids_other_plan():
    response = TestClient(_app()).post("/query/government", headers={"X-API-Key": "test-key-123"})
    assert response.status_code == 403


def test_auth_middleware_allows_public_health():
    response = TestClient(_app()).get("/health")
    assert response.status_code == 200


def test_auth_off_for_local_dev(monkeypatch):
    monkeypatch.setenv("RAG_API_AUTH", "off")
    response = TestClient(_app()).post("/query/government")
    assert response.status_code == 200
