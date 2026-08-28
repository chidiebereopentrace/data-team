"""Tests for enterprise tenant registry and auth."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ml.rag.session_store import clear_fallback_for_tests
from ml.serving.enterprise.auth import EnterpriseAuthMiddleware, extract_api_key
from ml.serving.enterprise.tenant_registry import reload_tenants, resolve_tenant


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
                        "api_key": "test-key-123",
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
    monkeypatch.setenv("CHATBOT_ENTERPRISE_AUTH", "required")
    monkeypatch.setenv("ENTERPRISE_METERING_ENABLED", "1")
    monkeypatch.delenv("RAG_REDIS_URL", raising=False)
    clear_fallback_for_tests()
    reload_tenants()
    yield
    clear_fallback_for_tests()
    reload_tenants()


def test_resolve_tenant_by_api_key():
    tenant = resolve_tenant("test-key-123")
    assert tenant is not None
    assert tenant.tenant_id == "test-tenant"
    assert tenant.plan_slug == "agribusinesses"


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
    app = FastAPI()
    app.add_middleware(EnterpriseAuthMiddleware)

    @app.get("/v1/chat/agribusinesses")
    async def chat():
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/v1/chat/agribusinesses")
    assert response.status_code == 401


def test_auth_middleware_allows_public_health():
    app = FastAPI()
    app.add_middleware(EnterpriseAuthMiddleware)

    @app.get("/v1/health")
    async def health():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 200
