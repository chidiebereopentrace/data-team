"""
Durable server-side session memory (summary + recent turns + profile) backed by Redis
when configured.

The facade in session_store.py provides the shared persistence; this module owns
memory folding and plan/category/country continuity on top of the blobs.
"""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.chat_memory import append_turn_and_compact
from ml.rag.chatbot.plan_policy import is_valid_plan_type
from ml.rag.chatbot.stakeholder_prompts import is_valid_category
from ml.rag.session_store import get_session_blob, save_session_blob


def _empty_session_blob() -> dict[str, Any]:
    """Default shape for a fresh session blob (internal)."""
    return {
        "conversation_summary": "",
        "recent_turns": [],
        "category": None,
        "plan_type": None,
        "country": None,
        "last_structured_ranking": None,
        "last_bq_facts": None,
    }


def persist_session_turn(
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    *,
    category: str | None = None,
    plan_type: str | None = None,
    country: str | None = None,
) -> None:
    blob = get_session_blob(session_id) or _empty_session_blob()
    summary, recent = append_turn_and_compact(
        str(blob.get("conversation_summary") or ""),
        blob.get("recent_turns"),
        user_msg,
        assistant_msg,
    )
    new_cat = blob.get("category")
    if category and is_valid_category(category):
        new_cat = category.strip()
    new_plan = blob.get("plan_type")
    if plan_type and is_valid_plan_type(plan_type):
        new_plan = plan_type.strip()
    new_country = blob.get("country")
    if country and str(country).strip():
        new_country = str(country).strip()

    new_blob: dict[str, Any] = {
        "conversation_summary": summary,
        "recent_turns": recent,
        "category": new_cat,
        "plan_type": new_plan,
        "country": new_country,
    }
    if blob.get("last_structured_ranking") is not None:
        new_blob["last_structured_ranking"] = blob["last_structured_ranking"]
    if blob.get("last_bq_facts") is not None:
        new_blob["last_bq_facts"] = blob["last_bq_facts"]
    save_session_blob(session_id, new_blob)
