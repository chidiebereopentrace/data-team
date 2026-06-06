"""
Optional Langfuse observability integration for the RAG pipeline.

- Gracefully no-ops when LANGFUSE_* keys are absent (fail-open for HF Space deploys).
- Provides a singleton Langfuse client and a ready-to-use CallbackHandler for LangGraph.
- Use get_langfuse() for manual traces/spans; get_langfuse_callback() for graph.invoke(config=...).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
except Exception:  # pragma: no cover - langfuse not installed or import error
    Langfuse = None  # type: ignore[assignment]
    LangfuseCallbackHandler = None  # type: ignore[assignment]


@lru_cache(maxsize=1)
def get_langfuse() -> Any | None:
    """Return a cached Langfuse client if keys are configured, else None."""
    if Langfuse is None:
        return None
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        return None
    host = os.environ.get("LANGFUSE_HOST", "").strip() or None
    try:
        return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except Exception:
        # Never break the app if Langfuse init fails
        return None


def get_langfuse_callback() -> Any | None:
    """Return a Langfuse callback handler configured for the current client (or None)."""
    if LangfuseCallbackHandler is None:
        return None
    lf = get_langfuse()
    if lf is None:
        return None
    try:
        return LangfuseCallbackHandler()
    except Exception:
        return None


def create_trace(name: str, **metadata: Any) -> Any | None:
    """Create a root trace if Langfuse is enabled; returns the trace or None."""
    lf = get_langfuse()
    if lf is None:
        return None
    try:
        return lf.trace(name=name, metadata=metadata or None)
    except Exception:
        return None
