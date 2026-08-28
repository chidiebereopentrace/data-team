"""Per-tenant usage metering and quota enforcement."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from ml.rag.session_store import get_json, incr_json_fields, set_json
from ml.serving.enterprise.tenant_registry import EnterpriseTenant

logger = logging.getLogger(__name__)

_METER_PREFIX = "enterprise:meter"
_RATE_PREFIX = "enterprise:rate"
_lock = threading.Lock()
_rate_windows: dict[str, list[float]] = {}


@dataclass(frozen=True)
class TenantUsage:
    tenant_id: str
    period: str
    query_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    export_count: int


def _current_period() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _meter_key(tenant_id: str, period: str | None = None) -> str:
    return f"{_METER_PREFIX}:{tenant_id}:{period or _current_period()}"


def metering_enabled() -> bool:
    return os.environ.get("ENTERPRISE_METERING_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def get_usage(tenant_id: str, period: str | None = None) -> TenantUsage:
    key = _meter_key(tenant_id, period)
    raw = get_json(key) or {}
    return TenantUsage(
        tenant_id=tenant_id,
        period=period or _current_period(),
        query_count=int(raw.get("query_count", 0) or 0),
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        total_tokens=int(raw.get("total_tokens", 0) or 0),
        export_count=int(raw.get("export_count", 0) or 0),
    )


def check_tenant_quotas(tenant: EnterpriseTenant) -> None:
    if not metering_enabled():
        return
    usage = get_usage(tenant.tenant_id)
    if tenant.monthly_query_quota > 0 and usage.query_count >= tenant.monthly_query_quota:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Monthly query quota exceeded for tenant {tenant.tenant_id}: "
                f"{tenant.monthly_query_quota} queries allowed."
            ),
        )
    if tenant.monthly_token_quota > 0 and usage.total_tokens >= tenant.monthly_token_quota:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Monthly token quota exceeded for tenant {tenant.tenant_id}: "
                f"{tenant.monthly_token_quota} tokens allowed."
            ),
        )


def check_tenant_rate_limit(tenant: EnterpriseTenant) -> None:
    limit = tenant.rate_limit_rpm
    if limit <= 0:
        return

    now = time.time()
    window_start = now - 60.0
    key = f"{tenant.tenant_id}"

    with _lock:
        timestamps = _rate_windows.setdefault(key, [])
        timestamps[:] = [ts for ts in timestamps if ts >= window_start]
        if len(timestamps) >= limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded for tenant {tenant.tenant_id}: "
                    f"{limit} requests per minute allowed."
                ),
                headers={"Retry-After": "60", "X-RateLimit-Limit": str(limit)},
            )
        timestamps.append(now)


def record_usage(
    tenant: EnterpriseTenant,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    export_count: int = 0,
) -> TenantUsage:
    if not metering_enabled():
        return get_usage(tenant.tenant_id)

    key = _meter_key(tenant.tenant_id)
    increments = {
        "query_count": 1,
        "input_tokens": max(0, input_tokens),
        "output_tokens": max(0, output_tokens),
        "total_tokens": max(0, total_tokens),
        "export_count": max(0, export_count),
    }
    raw = incr_json_fields(key, increments) or {}
    return TenantUsage(
        tenant_id=tenant.tenant_id,
        period=_current_period(),
        query_count=int(raw.get("query_count", 0) or 0),
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        total_tokens=int(raw.get("total_tokens", 0) or 0),
        export_count=int(raw.get("export_count", 0) or 0),
    )


def usage_to_dict(usage: TenantUsage, tenant: EnterpriseTenant) -> dict[str, Any]:
    return {
        "tenant_id": usage.tenant_id,
        "tenant_name": tenant.name,
        "environment": tenant.environment,
        "period": usage.period,
        "query_count": usage.query_count,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "export_count": usage.export_count,
        "quotas": {
            "monthly_query_quota": tenant.monthly_query_quota,
            "monthly_token_quota": tenant.monthly_token_quota,
            "rate_limit_rpm": tenant.rate_limit_rpm,
        },
    }
