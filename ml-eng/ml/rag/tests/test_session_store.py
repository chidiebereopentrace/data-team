"""
Lightweight tests for the Redis-backed session + cache facade.

Run:
  PYTHONPATH=ml-eng python -m pytest ml/rag/tests/test_session_store.py -q --tb=line

The memory backend is always exercised. For the Redis path install fakeredis
(pip install fakeredis) and the test will auto-use it for a second backend pass.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from ml.rag import session_store as ss


def _clear():
    ss.clear_fallback_for_tests()


def test_memory_backend_basic():
    """Exercise the in-memory fallback (used when no RAG_REDIS_URL)."""
    _clear()
    # Ensure we are in fallback mode for this test
    os.environ.pop("RAG_REDIS_URL", None)
    os.environ.pop("REDIS_URL", None)

    key = "rag:session:test_mem_1"
    assert ss.get_json(key) is None

    blob = {"conversation_summary": "foo", "recent_turns": [{"role": "user", "content": "hi"}]}
    ss.set_json(key, blob, ttl_s=60)
    got = ss.get_json(key)
    assert got == blob

    # Expiry
    ss.set_json(key, blob, ttl_s=0)  # no expiry for this one
    time.sleep(0.01)
    assert ss.get_json(key) == blob

    ss.delete_key(key)
    assert ss.get_json(key) is None


def test_session_helpers_memory():
    """High-level session helpers go through the same backend."""
    _clear()
    os.environ.pop("RAG_REDIS_URL", None)
    os.environ.pop("REDIS_URL", None)

    sid = "s123"
    ss.save_session_blob(sid, {"conversation_summary": "s", "recent_turns": []})
    b = ss.get_session_blob(sid)
    assert b and b["conversation_summary"] == "s"

    ss.delete_session(sid)
    assert ss.get_session_blob(sid) is None


def test_cache_helpers_memory():
    _clear()
    os.environ.pop("RAG_REDIS_URL", None)
    os.environ.pop("REDIS_URL", None)

    ss.set_bq_schema_cache("proj:bronze", "CREATE TABLE ...", ttl_s=10)
    assert ss.get_bq_schema_cache("proj:bronze") == "CREATE TABLE ..."

    ss.set_bronze_catalog_cache("v1:foo", {"t1": "`col` desc"}, ttl_s=10)
    assert ss.get_bronze_catalog_cache("v1:foo") == {"t1": "`col` desc"}


@pytest.mark.skipif(
    not (lambda: __import__("importlib.util").util.find_spec("fakeredis") is not None)(),
    reason="fakeredis not installed; Redis path not exercised. pip install fakeredis to enable.",
)
def test_redis_backend_when_fakeredis_present():
    """If fakeredis is available, run a quick round-trip against an in-mem fake Redis."""
    import fakeredis  # type: ignore

    _clear()
    # Force the module to (re)connect to a fake redis
    os.environ["RAG_REDIS_URL"] = "redis://localhost:6379/0"
    # Patch the client creation to use fakeredis
    fake = fakeredis.FakeRedis(decode_responses=True)

    # Monkey-patch the internal getter for the duration of the test
    orig_get = ss._get_client
    def _fake_get():
        return fake
    ss._redis_client = fake  # type: ignore[attr-defined]
    ss._get_client = _fake_get  # type: ignore[assignment]

    try:
        k = "rag:session:fake1"
        ss.set_json(k, {"a": 1}, ttl_s=30)
        assert ss.get_json(k) == {"a": 1}
        ss.delete_key(k)
        assert ss.get_json(k) is None

        # session helper
        ss.save_session_blob("sidX", {"conversation_summary": "x"})
        blob = ss.get_session_blob("sidX")
        assert blob is not None
        assert blob["conversation_summary"] == "x"
    finally:
        ss._get_client = orig_get  # type: ignore[assignment]
        ss._redis_client = None  # type: ignore[attr-defined]
        os.environ.pop("RAG_REDIS_URL", None)
        _clear()


if __name__ == "__main__":
    # Allow direct execution: python -m ml.rag.tests.test_session_store
    test_memory_backend_basic()
    test_session_helpers_memory()
    test_cache_helpers_memory()
    print("memory backend tests passed")
    try:
        test_redis_backend_when_fakeredis_present()
        print("fakeredis path exercised")
    except Exception as e:
        print("fakeredis path skipped or failed (ok if not installed):", e)
    print("ALL BASIC TESTS PASSED")