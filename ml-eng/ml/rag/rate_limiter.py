"""
Sliding-window rate limiter for plan-scoped API routes (ML-038).

Limits are per-plan per-client-IP, read from env vars at startup.
All limits default to 0 (disabled). Set e.g. FREE_RATE_LIMIT_RPM=10 to enable.

Thread-safe in-process implementation.
Redis-backed upgrade is straightforward when needed (swap _windows for a Redis ZSET).
"""
from __future__ import annotations

import collections
import os
import threading
import time

from fastapi import HTTPException, Request

# ---------------------------------------------------------------------------
# Per-plan RPM limits (0 = disabled) — read once at import time.
# Set via environment variables; all off by default for testing phase.
# ---------------------------------------------------------------------------
_PLAN_LIMITS: dict[str, int] = {
    "free": int(os.environ.get("FREE_RATE_LIMIT_RPM", "0") or 0),
    "farmers": int(os.environ.get("FARMERS_RATE_LIMIT_RPM", "0") or 0),
    "government": int(os.environ.get("GOVERNMENT_RATE_LIMIT_RPM", "0") or 0),
    "ngos": int(os.environ.get("NGOS_RATE_LIMIT_RPM", "0") or 0),
    "agribusinesses": int(os.environ.get("AGRIBUSINESSES_RATE_LIMIT_RPM", "0") or 0),
    "integrated": int(os.environ.get("INTEGRATED_RATE_LIMIT_RPM", "0") or 0),
}

# Sliding window storage: {window_key: deque[timestamp]}
_windows: dict[str, collections.deque] = {}
_lock = threading.Lock()

_WINDOW_SECONDS = 60.0  # 1-minute sliding window


def _client_ip(request: Request) -> str:
    """Extract the real client IP, respecting X-Forwarded-For (Railway proxy)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def check_plan_rate_limit(plan_slug: str, request: Request) -> None:
    """Raise HTTP 429 if this plan+IP has exceeded its per-minute request limit.

    No-ops when the plan limit is 0 (disabled).

    Parameters
    ----------
    plan_slug:
        URL slug for the plan (e.g. 'free', 'farmers'). Case-insensitive.
    request:
        The FastAPI Request object (used to extract the client IP).
    """
    limit = _PLAN_LIMITS.get(plan_slug.lower(), 0)
    if limit <= 0:
        return  # rate limiting disabled for this plan

    key = f"{plan_slug.lower()}:{_client_ip(request)}"
    now = time.time()
    window_start = now - _WINDOW_SECONDS

    with _lock:
        dq = _windows.setdefault(key, collections.deque())
        # Evict timestamps outside the current window
        while dq and dq[0] < window_start:
            dq.popleft()
        if len(dq) >= limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded for {plan_slug} plan: "
                    f"{limit} requests per minute allowed."
                ),
                headers={"Retry-After": "60", "X-RateLimit-Limit": str(limit)},
            )
        dq.append(now)


def get_rate_limit_status() -> dict[str, int]:
    """Return the configured limit per plan (for /meta or diagnostics)."""
    return dict(_PLAN_LIMITS)
