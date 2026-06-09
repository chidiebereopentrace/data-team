"""Shared request field resolution for POST /query and POST /v1/chat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.rag.api_schemas import UserProfile
from ml.rag.chatbot.stakeholder_prompts import is_valid_stakeholder_type
from ml.rag.session_store import get_session_blob


@dataclass(frozen=True)
class ResolvedRequestContext:
    stakeholder_type: str | None
    audience_instructions: str | None
    user_profile: dict[str, Any] | None
    history_messages: list[dict[str, str]] | None

    @property
    def has_client_history(self) -> bool:
        return self.history_messages is not None


def _normalize_stakeholder(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if not is_valid_stakeholder_type(s):
        raise ValueError(f"invalid stakeholder_type: {s!r}")
    return s


def _profile_dict(user_profile: UserProfile | dict[str, Any] | None) -> dict[str, Any]:
    if user_profile is None:
        return {}
    if isinstance(user_profile, UserProfile):
        return user_profile.model_dump()
    return dict(user_profile)


def effective_chat_history_messages(
    chat_history: list[Any] | None,
    conversation_history: list[Any] | None,
) -> list[dict[str, str]] | None:
    """Prefer chat_history; fall back to deprecated conversation_history."""
    hist = chat_history if chat_history is not None else conversation_history
    if hist is None:
        return None
    out: list[dict[str, str]] = []
    for m in hist:
        if hasattr(m, "model_dump"):
            dumped = m.model_dump()
        elif isinstance(m, dict):
            dumped = m
        else:
            dumped = {"role": getattr(m, "role", ""), "content": getattr(m, "content", "")}
        role = str(dumped.get("role") or "").strip()
        content = str(dumped.get("content") or "").strip()
        if role and content:
            out.append({"role": role, "content": content})
    return out


def resolve_request_context(
    *,
    user_profile: UserProfile | dict[str, Any] | None = None,
    chat_history: list[Any] | None = None,
    conversation_history: list[Any] | None = None,
    legacy_stakeholder_type: str | None = None,
    legacy_audience_instructions: str | None = None,
    session_id: str | None = None,
    use_session_stakeholder_fallback: bool = True,
) -> ResolvedRequestContext:
    """
    Resolve persona, audience, geo profile, and client history from a request.

    stakeholder_type precedence: user_profile.stakeholder_type → legacy top-level → session blob.
    """
    profile = _profile_dict(user_profile)
    country = str(profile.get("country") or "").strip() or None

    st = _normalize_stakeholder(profile.get("stakeholder_type"))
    if st is None:
        st = _normalize_stakeholder(legacy_stakeholder_type)
    if st is None and use_session_stakeholder_fallback:
        sid = (session_id or "").strip()
        if sid:
            blob = get_session_blob(sid) or {}
            raw = blob.get("stakeholder_type")
            if isinstance(raw, str) and is_valid_stakeholder_type(raw):
                st = raw.strip()

    aud_raw = profile.get("audience_instructions")
    if aud_raw is None:
        aud_raw = legacy_audience_instructions
    audience = str(aud_raw).strip() if aud_raw is not None else ""
    audience_instructions = audience or None

    geo_profile: dict[str, Any] | None = None
    if country is not None:
        geo_profile = {"country": country}
    elif profile:
        geo_profile = {}

    history_messages = effective_chat_history_messages(chat_history, conversation_history)

    return ResolvedRequestContext(
        stakeholder_type=st,
        audience_instructions=audience_instructions,
        user_profile=geo_profile,
        history_messages=history_messages,
    )


def bootstrap_stakeholder_type(
    user_profile: UserProfile | dict[str, Any] | None,
    legacy_stakeholder_type: str | None,
) -> str | None:
    """Stakeholder for session bootstrap (no session_id yet)."""
    profile = _profile_dict(user_profile)
    st = _normalize_stakeholder(profile.get("stakeholder_type"))
    if st is not None:
        return st
    return _normalize_stakeholder(legacy_stakeholder_type)
