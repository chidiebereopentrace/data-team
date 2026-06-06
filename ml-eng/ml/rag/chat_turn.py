"""
Shared chat execution entrypoint (execute_chat_turn) with durable server-side session
memory (summary + recent turns + stakeholder_type) backed by Redis when configured.

The facade in session_store.py provides the shared persistence; this module owns only
the stakeholder validation + memory folding logic on top of the blobs.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from ml.rag.chatbot.chat_history import normalize_messages
from ml.rag.chatbot.chat_memory import append_turn_and_compact, flat_messages_to_memory
from ml.rag.chatbot.stakeholder_prompts import is_valid_stakeholder_type
from ml.rag.session_store import get_session_blob, save_session_blob
from ml.rag.observability import get_langfuse


@dataclass
class ChatTurnResult:
    answer: str
    session_id: str
    pipeline_error: str | None = None
    raw_result: dict[str, Any] | None = None


def _empty_session_blob() -> dict[str, Any]:
    """Default shape for a fresh session blob (internal)."""
    return {"conversation_summary": "", "recent_turns": [], "stakeholder_type": None}


def create_session(stakeholder_type: str) -> str:
    if not is_valid_stakeholder_type(stakeholder_type):
        raise ValueError("invalid stakeholder_type")
    sid = uuid.uuid4().hex
    blob = _empty_session_blob()
    blob["stakeholder_type"] = stakeholder_type.strip()
    save_session_blob(sid, blob)
    return sid


def _resolve_prior_and_stakeholder(
    session_id: str | None,
    conversation_history: list[dict[str, str]] | None,
    explicit_stakeholder_type: str | None,
) -> tuple[str, str, list[dict[str, str]], str | None]:
    """
    Returns (session_id, conversation_summary, recent_turns, stakeholder_type_for_rag).
    """
    if conversation_history is not None:
        sid = (session_id or "").strip() or uuid.uuid4().hex
        raw_msgs = list(conversation_history)
        prior = normalize_messages(raw_msgs)
        summary, recent = flat_messages_to_memory(prior)
        st: str | None = None
        if explicit_stakeholder_type and is_valid_stakeholder_type(explicit_stakeholder_type):
            st = explicit_stakeholder_type.strip()
        elif (session_id or "").strip():
            blob = get_session_blob((session_id or "").strip()) or _empty_session_blob()
            raw = blob.get("stakeholder_type")
            if isinstance(raw, str) and is_valid_stakeholder_type(raw):
                st = raw.strip()
        return sid, summary, recent, st

    sid = (session_id or "").strip() or uuid.uuid4().hex
    blob = get_session_blob(sid) or _empty_session_blob()
    summary = str(blob.get("conversation_summary") or "")
    recent = normalize_messages(blob.get("recent_turns"))
    blob_st = blob.get("stakeholder_type")
    st = None
    if explicit_stakeholder_type is not None and is_valid_stakeholder_type(explicit_stakeholder_type):
        st = explicit_stakeholder_type.strip()
    elif isinstance(blob_st, str) and is_valid_stakeholder_type(blob_st):
        st = blob_st.strip()
    return sid, summary, recent, st


def persist_session_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    blob = get_session_blob(session_id) or _empty_session_blob()
    summary, recent = append_turn_and_compact(
        str(blob.get("conversation_summary") or ""),
        blob.get("recent_turns"),
        user_msg,
        assistant_msg,
    )
    # Preserve any stakeholder_type that was already in the session blob
    new_blob: dict[str, Any] = {
        "conversation_summary": summary,
        "recent_turns": recent,
        "stakeholder_type": blob.get("stakeholder_type"),
    }
    save_session_blob(session_id, new_blob)


def execute_chat_turn(
    query: str,
    *,
    session_id: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    stakeholder_type: str | None = None,
    persist_to_session: bool = True,
    **rag_kwargs: Any,
) -> ChatTurnResult:
    """
    Run one user query through run_rag with optional server session memory.

    stakeholder_type: when conversation_history is set, used if valid (or loaded from
    session_id if that session exists). When using server memory only, explicit value
    overrides session blob when valid.
    """
    sid, prior_summary, prior_recent, st = _resolve_prior_and_stakeholder(
        session_id, conversation_history, stakeholder_type
    )

    # Optional Langfuse span for session/memory resolution
    lf = get_langfuse()
    turn_trace = lf.trace(name="chat_turn", session_id=sid, stakeholder_type=st) if lf else None
    if turn_trace:
        turn_trace.update(input={"query": query[:200]})

    kwargs: dict[str, Any] = dict(rag_kwargs)
    if prior_summary.strip() or prior_recent:
        kwargs["conversation_summary"] = prior_summary
        kwargs["recent_turns"] = prior_recent
    if st:
        kwargs["stakeholder_type"] = st

    from ml.rag.chatbot.graph import run_rag  # defer heavy graph / torch imports

    result = run_rag(query.strip(), **kwargs)
    answer = result.get("answer", "") or ""
    err = result.get("error")
    err_s = str(err).strip() if err is not None else None
    if err_s == "":
        err_s = None

    if persist_to_session and conversation_history is None:
        persist_session_turn(sid, query.strip(), answer)

    return ChatTurnResult(
        answer=answer,
        session_id=sid,
        pipeline_error=err_s,
        raw_result=dict(result),
    )
