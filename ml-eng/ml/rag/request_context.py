"""Shared request field resolution for POST /query and POST /v1/chat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.rag.api_schemas import UserProfile
from ml.rag.chatbot.plan_policy import is_valid_category, is_valid_plan_type
from ml.rag.session_store import get_session_blob


@dataclass(frozen=True)
class ResolvedRequestContext:
    plan_type: str | None
    category: str | None
    user_profile: dict[str, Any] | None
    history_messages: list[dict[str, str]] | None

    @property
    def has_client_history(self) -> bool:
        return self.history_messages is not None


def _normalize_plan_type(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if not is_valid_plan_type(s):
        raise ValueError(f"invalid plan_type: {s!r}")
    return s


def _normalize_category(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if not is_valid_category(s):
        raise ValueError(f"invalid category: {s!r}")
    return s


def _profile_dict(user_profile: UserProfile | dict[str, Any] | None) -> dict[str, Any]:
    if user_profile is None:
        return {}
    if isinstance(user_profile, UserProfile):
        return user_profile.model_dump()
    return dict(user_profile)


def _blob_str(blob: dict[str, Any], key: str) -> str | None:
    raw = blob.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


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
    session_id: str | None = None,
    use_session_category_fallback: bool = True,
    injected_plan_type: str | None = None,
) -> ResolvedRequestContext:
    """
    Resolve plan tier, category persona, geo profile, and client history from a request.

    plan_type and category come from user_profile when present.
    When omitted, fall back to session blob (category, plan_type, country) if session_id is set.

    injected_plan_type: when set (by plan-scoped routes such as POST /v1/chat/farmers),
        overrides any plan_type in user_profile. The route owns the tier; the payload
        cannot escalate it.
    """
    profile = _profile_dict(user_profile)
    country = str(profile.get("country") or "").strip() or None

    if injected_plan_type is not None:
        plan_type = _normalize_plan_type(injected_plan_type)
    else:
        plan_type = _normalize_plan_type(profile.get("plan_type"))
    category = _normalize_category(profile.get("category"))

    if use_session_category_fallback:
        sid = (session_id or "").strip()
        if sid and (category is None or plan_type is None or country is None):
            blob = get_session_blob(sid) or {}
            if category is None:
                raw_cat = _blob_str(blob, "category")
                if raw_cat and is_valid_category(raw_cat):
                    category = raw_cat
            if plan_type is None and injected_plan_type is None:
                raw_plan = _blob_str(blob, "plan_type")
                if raw_plan and is_valid_plan_type(raw_plan):
                    plan_type = raw_plan
            if country is None:
                country = _blob_str(blob, "country")

    geo_profile: dict[str, Any] | None = None
    if plan_type is not None or category is not None or country is not None:
        geo_profile = {
            "country": country,
            "plan_type": plan_type,
            "category": category,
        }

    history_messages = effective_chat_history_messages(chat_history, conversation_history)

    return ResolvedRequestContext(
        plan_type=plan_type,
        category=category,
        user_profile=geo_profile,
        history_messages=history_messages,
    )


def bootstrap_category(
    user_profile: UserProfile | dict[str, Any] | None,
) -> str | None:
    """Category for session bootstrap (no session_id yet)."""
    profile = _profile_dict(user_profile)
    return _normalize_category(profile.get("category"))
