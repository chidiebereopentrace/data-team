"""
Redis-backed (or in-memory fallback) store for chat sessions and shared caches.

This enables the RAG FastAPI service to scale horizontally across workers/replicas
while keeping multi-turn conversation state durable and allowing cross-process reuse
of expensive lookups (BQ schema text, bronze catalog).

Design:
- Opaque JSON blobs keyed by namespaced strings (e.g. rag:session:<sid>).
- Callers own the shape of blobs and cache values.
- TTLs are explicit; 0 or None means no expiry.
- If RAG_REDIS_URL (or REDIS_URL) is absent or unreachable, falls back to a process-local
  dict with warning logs. Single-process dev / Streamlit continue to work.
- The pure chat memory compaction logic in chat_memory.py is untouched.

Intended use:
- api.py and chat_turn.py for sessions.
- BQRetriever and bronze_dataset_catalog for their caches.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

try:
    import redis
except ImportError:  # optional at dev time; required in production requirements
    redis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# --- Config (read at import; callers ensure load_rag_dotenv has run) ---
_REDIS_URL: str = (os.environ.get("RAG_REDIS_URL") or os.environ.get("REDIS_URL") or "").strip()
_REDIS_CONNECT_TIMEOUT_S: float = float(os.environ.get("RAG_REDIS_CONNECT_TIMEOUT_S", "2") or 2)
_SESSION_TTL_S: int = int(os.environ.get("RAG_SESSION_TTL_SECONDS", "86400") or 86400)
_CACHE_TTL_S: int = int(os.environ.get("RAG_CACHE_TTL_SECONDS", "3600") or 3600)

# Redis client (lazy, thread-safe init)
_redis_client: Any | None = None
_redis_lock = threading.Lock()

# In-memory fallback (thread-safe). Stores (value, expiry_ts or None)
_fallback: dict[str, tuple[Any, float | None]] = {}
_fallback_lock = threading.Lock()


def _get_client() -> Any | None:
    """Return a validated Redis client or None (triggers fallback)."""
    global _redis_client
    if redis is None:
        if _REDIS_URL:
            logger.warning(
                "session_store: redis package not installed; using in-process memory fallback"
            )
        return None
    if not _REDIS_URL:
        return None
    if _redis_client is not None:
        return _redis_client

    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            client = redis.Redis.from_url(
                _REDIS_URL,
                socket_connect_timeout=_REDIS_CONNECT_TIMEOUT_S,
                socket_timeout=_REDIS_CONNECT_TIMEOUT_S,
                decode_responses=True,
            )
            client.ping()
            _redis_client = client
            safe_url = _REDIS_URL.split("@")[-1] if "@" in _REDIS_URL else _REDIS_URL
            logger.info("session_store: connected to Redis (%s)", safe_url)
            return _redis_client
        except Exception as exc:
            logger.warning(
                "session_store: Redis unavailable (%s); falling back to in-process memory. "
                "Multi-worker deployments will not share sessions/caches. "
                "Set RAG_REDIS_URL (or REDIS_URL) for production.",
                exc,
            )
            return None


def _is_redis_available() -> bool:
    return _get_client() is not None


def _make_key(ns: str, ident: str) -> str:
    """Stable key with namespace prefix to avoid collisions."""
    return f"rag:{ns}:{ident}"


def _now() -> float:
    return time.time()


def get_json(key: str) -> Any | None:
    """
    Retrieve and JSON-decode a value, or None if missing/expired/unavailable.
    Works for both Redis and the fallback store.
    """
    client = _get_client()
    if client:
        try:
            raw = client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("session_store: redis get failed for %s: %s (falling back)", key, exc)
            # fall through to local fallback for this read

    # Fallback path (or after redis error)
    with _fallback_lock:
        if key not in _fallback:
            return None
        value, exp = _fallback[key]
        if exp is not None and _now() > exp:
            del _fallback[key]
            return None
        # Return a deep copy for safety (callers mutate)
        try:
            return json.loads(json.dumps(value))
        except Exception:
            return value


def set_json(key: str, value: Any, *, ttl_s: int | None = None) -> None:
    """
    JSON-encode and store value with optional TTL (seconds).
    ttl_s=None means use default for the category (session vs cache).
    ttl_s=0 means no expiry.
    """
    # Serialize up front so we fail fast on non-serializable
    try:
        payload = json.dumps(value)
    except (TypeError, ValueError) as exc:
        logger.error("session_store: cannot JSON-serialize value for %s: %s", key, exc)
        return

    eff_ttl: int | None = ttl_s
    if eff_ttl is None:
        # Heuristic: keys containing ":session:" get the session TTL; others the cache TTL
        eff_ttl = _SESSION_TTL_S if ":session:" in key else _CACHE_TTL_S

    client = _get_client()
    if client:
        try:
            if eff_ttl and eff_ttl > 0:
                client.set(key, payload, ex=eff_ttl)
            else:
                client.set(key, payload)
            return
        except Exception as exc:
            logger.warning("session_store: redis set failed for %s: %s (writing to fallback)", key, exc)

    # Fallback
    exp: float | None = None
    if eff_ttl and eff_ttl > 0:
        exp = _now() + eff_ttl
    with _fallback_lock:
        _fallback[key] = (value, exp)


def delete_key(key: str) -> None:
    """Delete a key from whichever backend is active."""
    client = _get_client()
    if client:
        try:
            client.delete(key)
            return
        except Exception as exc:
            logger.warning("session_store: redis delete failed for %s: %s", key, exc)

    with _fallback_lock:
        _fallback.pop(key, None)


# --- High-level session helpers (used by api.py and chat_turn.py) ---

def get_session_blob(session_id: str) -> dict[str, Any] | None:
    """Return the stored session blob or None."""
    if not session_id:
        return None
    return get_json(_make_key("session", session_id))


def save_session_blob(session_id: str, blob: dict[str, Any], ttl_s: int | None = None) -> None:
    """Persist (or update) a session blob. Defaults to RAG_SESSION_TTL_SECONDS."""
    if not session_id:
        return
    eff = ttl_s if ttl_s is not None else _SESSION_TTL_S
    set_json(_make_key("session", session_id), blob, ttl_s=eff)


def delete_session(session_id: str) -> None:
    """Remove a session entirely."""
    if not session_id:
        return
    delete_key(_make_key("session", session_id))


# --- Convenience cache helpers for known RAG caches ---

def get_bq_schema_cache(cache_key: str) -> str | None:
    """Retrieve cached BQ schema text (string)."""
    val = get_json(_make_key("bq", f"schema:{cache_key}"))
    return val if isinstance(val, str) else None


def set_bq_schema_cache(cache_key: str, schema_text: str, ttl_s: int | None = None) -> None:
    """Store BQ schema text. Use long TTL (hours) or versioned key for invalidation."""
    eff = ttl_s if ttl_s is not None else _CACHE_TTL_S
    set_json(_make_key("bq", f"schema:{cache_key}"), schema_text, ttl_s=eff)


def get_bronze_catalog_cache(cache_key: str) -> dict[str, str] | None:
    """Retrieve cached bronze table->columns mapping."""
    val = get_json(_make_key("catalog", f"bronze:{cache_key}"))
    return val if isinstance(val, dict) else None


def set_bronze_catalog_cache(cache_key: str, mapping: dict[str, str], ttl_s: int | None = None) -> None:
    eff = ttl_s if ttl_s is not None else _CACHE_TTL_S
    set_json(_make_key("catalog", f"bronze:{cache_key}"), mapping, ttl_s=eff)


# --- Diagnostics for /ready and ops ---

def redis_status() -> dict[str, Any]:
    """Lightweight status for health/readiness endpoints."""
    client = _get_client()
    if client:
        try:
            client.ping()
            return {"backend": "redis", "url": _REDIS_URL.split("@")[-1] if "@" in _REDIS_URL else _REDIS_URL, "connected": True}
        except Exception as exc:
            return {"backend": "redis", "connected": False, "error": str(exc)}
    return {"backend": "memory-fallback", "connected": False, "reason": "RAG_REDIS_URL not set or connect failed"}


def clear_fallback_for_tests() -> None:
    """Test helper only. Do not call in production."""
    with _fallback_lock:
        _fallback.clear()
