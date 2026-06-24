"""
Shared chat execution entrypoint (execute_chat_turn) with durable server-side session
memory (summary + recent turns + category) backed by Redis when configured.

The facade in session_store.py provides the shared persistence; this module owns only
the category validation + memory folding logic on top of the blobs.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from ml.rag.chatbot.chat_history import normalize_messages
from ml.rag.chatbot.chat_memory import append_turn_and_compact, flat_messages_to_memory
from ml.rag.chatbot.stakeholder_prompts import is_valid_category
from ml.rag.session_store import get_session_blob, save_session_blob
from ml.rag.observability import get_langfuse


@dataclass
class ChatTurnResult:
    answer: str
    session_id: str
    citations: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    pipeline_error: str | None = None
    raw_result: dict[str, Any] | None = None


def _empty_session_blob() -> dict[str, Any]:
    """Default shape for a fresh session blob (internal)."""
    return {"conversation_summary": "", "recent_turns": [], "category": None}


def create_session(category: str) -> str:
    if not is_valid_category(category):
        raise ValueError("invalid category")
    sid = uuid.uuid4().hex
    blob = _empty_session_blob()
    blob["category"] = category.strip()
    save_session_blob(sid, blob)
    return sid


def _resolve_prior_and_category(
    session_id: str | None,
    conversation_history: list[dict[str, str]] | None,
    explicit_category: str | None,
) -> tuple[str, str, list[dict[str, str]], str | None]:
    """
    Returns (session_id, conversation_summary, recent_turns, category_for_rag).
    """
    if conversation_history is not None:
        sid = (session_id or "").strip() or uuid.uuid4().hex
        raw_msgs = list(conversation_history)
        prior = normalize_messages(raw_msgs)
        summary, recent = flat_messages_to_memory(prior)
        cat: str | None = None
        if explicit_category and is_valid_category(explicit_category):
            cat = explicit_category.strip()
        elif (session_id or "").strip():
            blob = get_session_blob((session_id or "").strip()) or _empty_session_blob()
            raw = blob.get("category")
            if isinstance(raw, str) and is_valid_category(raw):
                cat = raw.strip()
        return sid, summary, recent, cat

    sid = (session_id or "").strip() or uuid.uuid4().hex
    blob = get_session_blob(sid) or _empty_session_blob()
    summary = str(blob.get("conversation_summary") or "")
    recent = normalize_messages(blob.get("recent_turns"))
    blob_cat = blob.get("category")
    cat = None
    if explicit_category is not None and is_valid_category(explicit_category):
        cat = explicit_category.strip()
    elif isinstance(blob_cat, str) and is_valid_category(blob_cat):
        cat = blob_cat.strip()
    return sid, summary, recent, cat


def persist_session_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    blob = get_session_blob(session_id) or _empty_session_blob()
    summary, recent = append_turn_and_compact(
        str(blob.get("conversation_summary") or ""),
        blob.get("recent_turns"),
        user_msg,
        assistant_msg,
    )
    new_blob: dict[str, Any] = {
        "conversation_summary": summary,
        "recent_turns": recent,
        "category": blob.get("category"),
    }
    save_session_blob(session_id, new_blob)


def execute_chat_turn(
    query: str,
    *,
    session_id: str | None = None,
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

    plan_type and category are resolved by the caller from user_profile.
    When chat_history is set, server session is not updated unless persist_to_session
    is True and chat_history is absent.
    """
    history = chat_history if chat_history is not None else conversation_history
    sid, prior_summary, prior_recent, cat = _resolve_prior_and_category(
        session_id, history, category
    )

    lf = get_langfuse()
    turn_trace = lf.trace(name="chat_turn", session_id=sid, category=cat) if lf else None
    if turn_trace:
        turn_trace.update(input={"query": query[:200]})

    kwargs: dict[str, Any] = dict(rag_kwargs)
    if prior_summary.strip() or prior_recent:
        kwargs["conversation_summary"] = prior_summary
        kwargs["recent_turns"] = prior_recent
    if plan_type:
        kwargs["plan_type"] = plan_type
    if cat:
        kwargs["category"] = cat
    if user_profile is not None:
        kwargs["user_profile"] = user_profile

    from ml.rag.chatbot.graph import run_rag  # defer heavy graph / torch imports

    result = run_rag(query.strip(), **kwargs)
    answer = result.get("answer", "") or ""
    err = result.get("error")
    err_s = str(err).strip() if err is not None else None
    if err_s == "":
        err_s = None

    if persist_to_session and history is None:
        persist_session_turn(sid, query.strip(), answer)

    raw_citations = result.get("citations")
    citations = list(raw_citations) if isinstance(raw_citations, list) else []
    raw_usage = result.get("usage")
    usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}

    return ChatTurnResult(
        answer=answer,
        session_id=sid,
        citations=citations,
        usage=usage,
        pipeline_error=err_s,
        raw_result=dict(result),
    )
