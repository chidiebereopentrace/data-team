"""API tenant registry for the Ask ADZA RAG API (partners and the web frontend)."""

from __future__ import annotations

import hashlib
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
    api_key_sha256: str
    plan_slug: str
    environment: str
    monthly_query_quota: int
    monthly_token_quota: int
    rate_limit_rpm: int
    allow_trace: bool
    active: bool

    def allows_plan_slug(self, route_slug: str) -> bool:
        slug = route_slug.strip().lower()
        if self.plan_slug == "integrated":
            return slug in _VALID_PLAN_SLUGS
        return slug == self.plan_slug


def hash_api_key(api_key: str) -> str:
    """Hex SHA-256 of an API key; the registry stores only this digest."""
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()


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
        api_key_sha256=str(raw["api_key_sha256"]).strip().lower(),
        plan_slug=plan_slug,
        environment=str(raw.get("environment", "sandbox")).strip(),
        monthly_query_quota=int(raw.get("monthly_query_quota", 0) or 0),
        monthly_token_quota=int(raw.get("monthly_token_quota", 0) or 0),
        rate_limit_rpm=int(raw.get("rate_limit_rpm", 0) or 0),
        allow_trace=bool(raw.get("allow_trace", False)),
        active=bool(raw.get("active", True)),
    )


def load_tenants(path: Path | None = None) -> dict[str, EnterpriseTenant]:
    """Load active tenants keyed by API key SHA-256 digest."""
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

    by_digest: dict[str, EnterpriseTenant] = {}
    for raw in raw_tenants:
        if not isinstance(raw, dict):
            continue
        tenant = _parse_tenant(raw)
        if not tenant.active:
            continue
        if len(tenant.api_key_sha256) != 64:
            raise ValueError(f"tenant {tenant.tenant_id!r} api_key_sha256 must be a hex SHA-256")
        if tenant.api_key_sha256 in by_digest:
            raise ValueError(f"duplicate api_key_sha256 for tenant {tenant.tenant_id!r}")
        by_digest[tenant.api_key_sha256] = tenant
    return by_digest


_TENANTS_BY_DIGEST: dict[str, EnterpriseTenant] | None = None


def get_tenants() -> dict[str, EnterpriseTenant]:
    global _TENANTS_BY_DIGEST
    if _TENANTS_BY_DIGEST is None:
        _TENANTS_BY_DIGEST = load_tenants()
    return _TENANTS_BY_DIGEST


def reload_tenants() -> dict[str, EnterpriseTenant]:
    """Reload tenant registry (for tests)."""
    global _TENANTS_BY_DIGEST
    _TENANTS_BY_DIGEST = load_tenants()
    return _TENANTS_BY_DIGEST


def resolve_tenant(api_key: str | None) -> EnterpriseTenant | None:
    if not api_key or not api_key.strip():
        return None
    return get_tenants().get(hash_api_key(api_key))


def enterprise_auth_mode() -> str:
    """``required`` (default) or ``off``. ``off`` is for local development only."""
    mode = os.environ.get("RAG_API_AUTH", "required").strip().lower()
    return "off" if mode == "off" else "required"
