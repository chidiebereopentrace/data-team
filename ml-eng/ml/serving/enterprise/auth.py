"""Enterprise API key authentication middleware."""

from __future__ import annotations

import re
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ml.serving.enterprise.metering import check_tenant_quotas, check_tenant_rate_limit
from ml.serving.enterprise.tenant_registry import (
    EnterpriseTenant,
    enterprise_auth_mode,
    resolve_tenant,
)

_PUBLIC_PATHS = frozenset({"/", "/docs", "/openapi.json", "/redoc", "/v1/health", "/v1/meta"})
_PLAN_CHAT_RE = re.compile(r"^/v1/chat/(?P<slug>[a-z]+)$")


def extract_api_key(request: Request) -> str | None:
    header_key = request.headers.get("x-api-key", "").strip()
    if header_key:
        return header_key
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _route_plan_slug(path: str) -> str | None:
    match = _PLAN_CHAT_RE.match(path)
    if not match:
        return None
    return match.group("slug")


class EnterpriseAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        mode = enterprise_auth_mode()
        if mode == "off":
            return await call_next(request)

        path = request.url.path
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = extract_api_key(request)
        tenant = resolve_tenant(api_key)

        if tenant is None:
            if mode == "required":
                return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key"})
            return await call_next(request)

        request.state.enterprise_tenant = tenant
        try:
            check_tenant_quotas(tenant)
            check_tenant_rate_limit(tenant)
        except Exception as exc:
            if hasattr(exc, "status_code") and hasattr(exc, "detail"):
                headers = getattr(exc, "headers", None) or {}
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers=headers,
                )
            raise

        route_slug = _route_plan_slug(path)
        if route_slug and not tenant.allows_plan_slug(route_slug):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        f"Tenant {tenant.tenant_id!r} is not authorized for plan route "
                        f"/v1/chat/{route_slug}"
                    )
                },
            )

        return await call_next(request)


def get_request_tenant(request: Request) -> EnterpriseTenant | None:
    return getattr(request.state, "enterprise_tenant", None)
