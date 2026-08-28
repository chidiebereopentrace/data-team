"""Enterprise B2B tenant registry for Ask ADZA Chatbot API."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VALID_PLAN_SLUGS = frozenset(
    {"free", "farmers", "government", "ngos", "agribusinesses", "integrated"}
)


@dataclass(frozen=True)
class EnterpriseTenant:
    tenant_id: str
    name: str
    api_key: str
    plan_slug: str
    environment: str
    monthly_query_quota: int
    monthly_token_quota: int
    rate_limit_rpm: int
    active: bool

    def allows_plan_slug(self, route_slug: str) -> bool:
        slug = route_slug.strip().lower()
        if self.plan_slug == "integrated":
            return slug in _VALID_PLAN_SLUGS
        return slug == self.plan_slug


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_tenants_path() -> Path:
    configured = os.environ.get("ENTERPRISE_TENANTS_PATH", "").strip()
    if configured:
        return Path(configured)
    return _repo_root() / "config" / "enterprise_tenants.json"


def _parse_tenant(raw: dict[str, Any]) -> EnterpriseTenant:
    plan_slug = str(raw.get("plan_slug", "")).strip().lower()
    if plan_slug not in _VALID_PLAN_SLUGS:
        raise ValueError(f"invalid plan_slug {plan_slug!r} for tenant {raw.get('tenant_id')!r}")

    return EnterpriseTenant(
        tenant_id=str(raw["tenant_id"]).strip(),
        name=str(raw.get("name", raw["tenant_id"])).strip(),
        api_key=str(raw["api_key"]).strip(),
        plan_slug=plan_slug,
        environment=str(raw.get("environment", "sandbox")).strip(),
        monthly_query_quota=int(raw.get("monthly_query_quota", 0) or 0),
        monthly_token_quota=int(raw.get("monthly_token_quota", 0) or 0),
        rate_limit_rpm=int(raw.get("rate_limit_rpm", 0) or 0),
        active=bool(raw.get("active", True)),
    )


def load_tenants(path: Path | None = None) -> dict[str, EnterpriseTenant]:
    """Load tenants keyed by api_key."""
    tenants_path = path or _default_tenants_path()
    inline = os.environ.get("ENTERPRISE_TENANTS_JSON", "").strip()
    if inline:
        payload = json.loads(inline)
    elif tenants_path.is_file():
        payload = json.loads(tenants_path.read_text(encoding="utf-8"))
    else:
        logger.info("enterprise: no tenant registry at %s", tenants_path)
        return {}

    raw_tenants = payload.get("tenants", payload if isinstance(payload, list) else [])
    if not isinstance(raw_tenants, list):
        raise ValueError("enterprise tenant registry must contain a 'tenants' array")

    by_key: dict[str, EnterpriseTenant] = {}
    for raw in raw_tenants:
        if not isinstance(raw, dict):
            continue
        tenant = _parse_tenant(raw)
        if not tenant.active:
            continue
        if not tenant.api_key:
            raise ValueError(f"tenant {tenant.tenant_id!r} has empty api_key")
        if tenant.api_key in by_key:
            raise ValueError(f"duplicate api_key for tenant {tenant.tenant_id!r}")
        by_key[tenant.api_key] = tenant
    return by_key


_TENANTS_BY_KEY: dict[str, EnterpriseTenant] | None = None


def get_tenants() -> dict[str, EnterpriseTenant]:
    global _TENANTS_BY_KEY
    if _TENANTS_BY_KEY is None:
        _TENANTS_BY_KEY = load_tenants()
    return _TENANTS_BY_KEY


def reload_tenants() -> dict[str, EnterpriseTenant]:
    """Reload tenant registry (for tests)."""
    global _TENANTS_BY_KEY
    _TENANTS_BY_KEY = load_tenants()
    return _TENANTS_BY_KEY


def resolve_tenant(api_key: str | None) -> EnterpriseTenant | None:
    if not api_key:
        return None
    return get_tenants().get(api_key.strip())


def enterprise_auth_mode() -> str:
    """off | optional | required"""
    mode = os.environ.get("CHATBOT_ENTERPRISE_AUTH", "off").strip().lower()
    if mode not in {"off", "optional", "required"}:
        return "off"
    return mode
