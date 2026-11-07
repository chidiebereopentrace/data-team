"""API key authentication middleware for the RAG API."""

from __future__ import annotations

import re
from typing import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ml.serving.enterprise.metering import check_tenant_quotas, check_tenant_rate_limit
from ml.serving.enterprise.tenant_registry import (
    EnterpriseTenant,
    enterprise_auth_mode,
    resolve_tenant,
)

_PUBLIC_PATHS = frozenset({"/", "/health", "/ready"})
_PLAN_QUERY_RE = re.compile(r"^/query/(?P<slug>[a-z]+)$")


def extract_api_key(request: Request) -> str | None:
    header_key = request.headers.get("x-api-key", "").strip()
    if header_key:
        return header_key
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _route_plan_slug(path: str) -> str | None:
    match = _PLAN_QUERY_RE.match(path)
    if not match:
        return None
    return match.group("slug")


class EnterpriseAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if enterprise_auth_mode() == "off":
            return await call_next(request)

        path = request.url.path
        if request.method == "OPTIONS" or path in _PUBLIC_PATHS:
            return await call_next(request)

        tenant = resolve_tenant(extract_api_key(request))
        if tenant is None:
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key"})

        try:
            check_tenant_quotas(tenant)
            check_tenant_rate_limit(tenant)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers or {},
            )

        route_slug = _route_plan_slug(path)
        if route_slug and not tenant.allows_plan_slug(route_slug):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        f"Tenant {tenant.tenant_id!r} is not authorized for plan route "
                        f"/query/{route_slug}"
                    )
                },
            )

        request.state.enterprise_tenant = tenant
        return await call_next(request)


def get_request_tenant(request: Request) -> EnterpriseTenant | None:
    return getattr(request.state, "enterprise_tenant", None)
