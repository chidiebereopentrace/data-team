"""Shared time window helpers for BQ and Qdrant retrieval from TurnContract."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.turn_contract import TurnContract


def time_kwargs_from_contract(contract: TurnContract | None) -> dict[str, Any]:
    """Return published_at bounds and hard-filter flag for vector retrieval."""
    if contract is None:
        return {"hard_filter": False, "time_role": "either"}
    ts = contract.time_spec
    out: dict[str, Any] = {
        "hard_filter": bool(ts.hard_filter),
        "time_role": ts.time_role,
    }
    if ts.start:
        out["published_at_from"] = ts.start[:10]
    if ts.end:
        out["published_at_to"] = ts.end[:10]
    return out


def sync_decomposition_time(decomposition: dict[str, Any], contract: TurnContract | None) -> dict[str, Any]:
    """Copy contract time bounds into decomposition when decomposition lacks them."""
    if contract is None:
        return decomposition
    out = dict(decomposition or {})
    ts = contract.time_spec
    if ts.start and not str(out.get("time_start") or "").strip():
        out["time_start"] = ts.start[:10]
    if ts.end and not str(out.get("time_end") or "").strip():
        out["time_end"] = ts.end[:10]
    return out


def time_fallback_enabled(*, hard_filter: bool, env_var: str, default: bool = True) -> bool:
    """Disable Qdrant time cascade when the user gave an explicit time window."""
    if hard_filter:
        return False
    import os

    raw = os.environ.get(env_var, "on" if default else "off").strip().lower()
    return raw not in ("0", "false", "off", "no")
