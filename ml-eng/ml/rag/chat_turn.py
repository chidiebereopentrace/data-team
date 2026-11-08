"""
Durable server-side session memory (summary + recent turns + profile + active request) backed by
Redis when configured.

The facade in session_store.py provides the shared persistence; this module owns memory folding
and plan/category/country continuity on top of the blobs.
"""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.chat_memory import append_turn_and_compact
from ml.rag.chatbot.plan_policy import is_valid_plan_type
from ml.rag.chatbot.stakeholder_prompts import is_valid_category
from ml.rag.session_store import get_session_blob, save_session_blob


def empty_session_blob() -> dict[str, Any]:
    return {
        "conversation_summary": "",
        "recent_turns": [],
        "category": None,
        "plan_type": None,
        "country": None,
        "active_request": None,
        "last_bq_facts": None,
        "state_version": 0,
    }


def load_session(owner: str, session_id: str) -> tuple[dict[str, Any], bool]:
    """(blob, found). A missing session is an empty blob at version 0."""
    blob = get_session_blob(owner, session_id)
    return (blob if isinstance(blob, dict) else empty_session_blob()), blob is not None


def persist_session_turn(
    owner: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    *,
    loaded: dict[str, Any],
    result: dict[str, Any],
    category: str | None = None,
    plan_type: str | None = None,
    country: str | None = None,
) -> bool:
    """Append the turn on top of the state this request loaded. False when a newer turn won."""
    summary, recent = append_turn_and_compact(
        str(loaded.get("conversation_summary") or ""),
        loaded.get("recent_turns"),
        user_msg,
        assistant_msg,
    )
    new_blob: dict[str, Any] = {
        "conversation_summary": summary,
        "recent_turns": recent,
        "category": category.strip() if category and is_valid_category(category) else loaded.get("category"),
        "plan_type": plan_type.strip() if plan_type and is_valid_plan_type(plan_type) else loaded.get("plan_type"),
        "country": str(country).strip() if country and str(country).strip() else loaded.get("country"),
        "active_request": result.get("active_request") or loaded.get("active_request"),
        "last_bq_facts": result.get("last_bq_facts") or loaded.get("last_bq_facts"),
    }
    return save_session_blob(
        owner,
        session_id,
        new_blob,
        expected_version=int(loaded.get("state_version") or 0),
    )
