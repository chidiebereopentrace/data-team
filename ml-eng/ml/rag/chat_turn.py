"""
Shared chat execution entrypoint (execute_chat_turn) with durable server-side session
memory (summary + recent turns + profile) backed by Redis when configured.

The facade in session_store.py provides the shared persistence; this module owns
memory folding and plan/category/country continuity on top of the blobs.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from ml.rag.chatbot.chat_history import normalize_messages
from ml.rag.chatbot.chat_memory import append_turn_and_compact, flat_messages_to_memory
from ml.rag.chatbot.plan_policy import is_valid_plan_type
from ml.rag.chatbot.stakeholder_prompts import is_valid_category
from ml.rag.observability import flush_langfuse, get_current_trace_id, rag_trace_context
from ml.rag.session_store import get_session_blob, save_session_blob


@dataclass
class ChatTurnResult:
    answer: str
    session_id: str
    citations: list[dict[str, Any]] | None = None
    artifacts: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    pipeline_error: str | None = None
    raw_result: dict[str, Any] | None = None
    langfuse_trace_id: str | None = None


def _empty_session_blob() -> dict[str, Any]:
    """Default shape for a fresh session blob (internal)."""
    return {
        "conversation_summary": "",
        "recent_turns": [],
        "category": None,
        "plan_type": None,
        "country": None,
    }


def create_session(category: str, *, plan_type: str | None = None, country: str | None = None) -> str:
    if not is_valid_category(category):
        raise ValueError("invalid category")
    sid = uuid.uuid4().hex
    blob = _empty_session_blob()
    blob["category"] = category.strip()
    if plan_type and is_valid_plan_type(plan_type):
        blob["plan_type"] = plan_type.strip()
    if country and str(country).strip():
        blob["country"] = str(country).strip()
    save_session_blob(sid, blob)
    return sid


def _blob_str(blob: dict[str, Any], key: str) -> str | None:
    raw = blob.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _resolve_prior_and_profile(
    session_id: str | None,
    conversation_history: list[dict[str, str]] | None,
    explicit_category: str | None,
    explicit_plan_type: str | None,
    explicit_country: str | None,
) -> tuple[str, str, list[dict[str, str]], str | None, str | None, str | None]:
    """
    Returns (session_id, summary, recent_turns, category, plan_type, country).
    Explicit request fields win; otherwise fall back to the session blob.
    """
    if conversation_history is not None:
        sid = (session_id or "").strip() or uuid.uuid4().hex
        raw_msgs = list(conversation_history)
        prior = normalize_messages(raw_msgs)
        summary, recent = flat_messages_to_memory(prior)
        sid_key = (session_id or "").strip()
        blob = (get_session_blob(sid_key) or _empty_session_blob()) if sid_key else _empty_session_blob()
    else:
        sid = (session_id or "").strip() or uuid.uuid4().hex
        blob = get_session_blob(sid) or _empty_session_blob()
        summary = str(blob.get("conversation_summary") or "")
        recent = normalize_messages(blob.get("recent_turns"))

    cat: str | None = None
    if explicit_category and is_valid_category(explicit_category):
        cat = explicit_category.strip()
    else:
        blob_cat = _blob_str(blob, "category")
        if blob_cat and is_valid_category(blob_cat):
            cat = blob_cat

    plan: str | None = None
    if explicit_plan_type and is_valid_plan_type(explicit_plan_type):
        plan = explicit_plan_type.strip()
    else:
        blob_plan = _blob_str(blob, "plan_type")
        if blob_plan and is_valid_plan_type(blob_plan):
            plan = blob_plan

    country: str | None = None
    if explicit_country and str(explicit_country).strip():
        country = str(explicit_country).strip()
    else:
        country = _blob_str(blob, "country")

    return sid, summary, recent, cat, plan, country


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
    save_session_blob(session_id, new_blob)


def execute_chat_turn(
    query: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    plan_type: str | None = None,
    category: str | None = None,
    user_profile: dict[str, Any] | None = None,
    persist_to_session: bool = True,
    **rag_kwargs: Any,
) -> ChatTurnResult:
    """
    Run one user query through run_rag with optional server session memory.

    plan_type and category are resolved from args / user_profile, with session blob fallback.
    When chat_history is set, server session is not updated unless persist_to_session
    is True and chat_history is absent.
    """
    history = chat_history if chat_history is not None else conversation_history
    profile = user_profile if isinstance(user_profile, dict) else {}
    explicit_cat = category or (str(profile.get("category") or "").strip() or None)
    explicit_plan = plan_type or (str(profile.get("plan_type") or "").strip() or None)
    explicit_country = str(profile.get("country") or "").strip() or None

    sid, prior_summary, prior_recent, cat, plan, country = _resolve_prior_and_profile(
        session_id,
        history,
        explicit_cat,
        explicit_plan,
        explicit_country,
    )

    kwargs: dict[str, Any] = dict(rag_kwargs)
    if prior_summary.strip() or prior_recent:
        kwargs["conversation_summary"] = prior_summary
        kwargs["recent_turns"] = prior_recent
    if plan:
        kwargs["plan_type"] = plan
    if cat:
        kwargs["category"] = cat
    # Always pass a profile dict when any lens field is known (geo + tone continuity).
    if plan or cat or country:
        kwargs["user_profile"] = {
            "country": country,
            "plan_type": plan,
            "category": cat,
        }
    elif user_profile is not None:
        kwargs["user_profile"] = user_profile
    kwargs["session_id"] = sid
    kwargs["trace_tags"] = ["chat"]

    from ml.rag.chatbot.graph import run_rag  # defer heavy graph / torch imports

    with rag_trace_context(
        trace_name="rag.chat_turn",
        session_id=sid,
        user_id=user_id,
        plan_type=plan,
        category=cat,
        trace_input={"query": query[:500]},
        tags=["chat"],
    ) as trace_handle:
        result = run_rag(query.strip(), **kwargs)
        trace_handle.update_output(result)
        langfuse_trace_id = get_current_trace_id()
    flush_langfuse()

    answer = result.get("answer", "") or ""
    err = result.get("error")
    err_s = str(err).strip() if err is not None else None
    if err_s == "":
        err_s = None

    if persist_to_session and history is None:
        persist_session_turn(
            sid,
            query.strip(),
            answer,
            category=cat,
            plan_type=plan,
            country=country,
        )

    raw_citations = result.get("citations")
    citations = list(raw_citations) if isinstance(raw_citations, list) else []
    raw_artifacts = result.get("artifacts")
    artifacts = list(raw_artifacts) if isinstance(raw_artifacts, list) else []
    raw_usage = result.get("usage")
    usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}

    return ChatTurnResult(
        answer=answer,
        session_id=sid,
        citations=citations,
        artifacts=artifacts,
        usage=usage,
        pipeline_error=err_s,
        raw_result=dict(result),
        langfuse_trace_id=langfuse_trace_id,
    )
