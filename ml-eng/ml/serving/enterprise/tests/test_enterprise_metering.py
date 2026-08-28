"""Tests for enterprise usage metering."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from ml.rag.session_store import clear_fallback_for_tests
from ml.serving.enterprise.metering import (
    check_tenant_quotas,
    get_usage,
    record_usage,
)
from ml.serving.enterprise.tenant_registry import EnterpriseTenant, reload_tenants


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    tenants_file = tmp_path / "tenants.json"
    tenants_file.write_text(
        json.dumps(
            {
                "tenants": [
                    {
                        "tenant_id": "meter-tenant",
                        "name": "Meter Tenant",
                        "api_key": "meter-key",
                        "plan_slug": "agribusinesses",
                        "monthly_query_quota": 2,
                        "monthly_token_quota": 50,
                        "rate_limit_rpm": 0,
                        "active": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENTERPRISE_TENANTS_PATH", str(tenants_file))
    monkeypatch.setenv("ENTERPRISE_METERING_ENABLED", "1")
    monkeypatch.delenv("RAG_REDIS_URL", raising=False)
    clear_fallback_for_tests()
    reload_tenants()
    yield
    clear_fallback_for_tests()
    reload_tenants()


def _tenant() -> EnterpriseTenant:
    tenant = reload_tenants()["meter-key"]
    assert tenant is not None
    return tenant


def test_record_usage_increments_counters():
    tenant = _tenant()
    record_usage(tenant, input_tokens=10, output_tokens=5, total_tokens=15, export_count=1)
    usage = get_usage(tenant.tenant_id)
    assert usage.query_count == 1
    assert usage.total_tokens == 15
    assert usage.export_count == 1


def test_quota_enforcement():
    tenant = _tenant()
    record_usage(tenant, total_tokens=10)
    record_usage(tenant, total_tokens=10)
    with pytest.raises(HTTPException) as exc:
        check_tenant_quotas(tenant)
    assert exc.value.status_code == 429
