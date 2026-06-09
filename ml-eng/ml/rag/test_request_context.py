"""Unit tests for shared API request resolution."""

from __future__ import annotations

import pytest

from ml.rag.api_schemas import UserProfile
from ml.rag.app.api import ChatMessage, QueryRequest, _resolve_prior_memory
from ml.rag.request_context import (
    bootstrap_stakeholder_type,
    effective_chat_history_messages,
    resolve_request_context,
)
from ml.serving.chat.schemas import ChatRequest


def test_nested_profile_resolves_stakeholder_and_audience() -> None:
    ctx = resolve_request_context(
        user_profile=UserProfile(
            country="Ghana",
            stakeholder_type="farmers_communities",
            audience_instructions="Keep it simple.",
        ),
    )
    assert ctx.stakeholder_type == "farmers_communities"
    assert ctx.audience_instructions == "Keep it simple."
    assert ctx.user_profile == {"country": "Ghana"}


def test_chat_history_preferred_over_conversation_history() -> None:
    chat = [{"role": "user", "content": "from chat_history"}]
    conv = [{"role": "user", "content": "from conversation_history"}]
    msgs = effective_chat_history_messages(chat, conv)
    assert msgs is not None
    assert msgs[0]["content"] == "from chat_history"


def test_legacy_top_level_fallbacks() -> None:
    ctx = resolve_request_context(
        legacy_stakeholder_type="government_public",
        legacy_audience_instructions="Formal tone.",
    )
    assert ctx.stakeholder_type == "government_public"
    assert ctx.audience_instructions == "Formal tone."


def test_invalid_stakeholder_raises() -> None:
    with pytest.raises(ValueError, match="invalid stakeholder_type"):
        resolve_request_context(
            user_profile=UserProfile(stakeholder_type="not_a_real_persona"),
        )


def test_bootstrap_stakeholder_from_profile() -> None:
    st = bootstrap_stakeholder_type(
        UserProfile(stakeholder_type="farmers_communities"),
        None,
    )
    assert st == "farmers_communities"


def test_query_request_backend_shape() -> None:
    req = QueryRequest.model_validate(
        {
            "query": "What are rice yield trends?",
            "session_id": "abc123",
            "user_profile": {
                "country": "Ghana",
                "audience_instructions": None,
                "stakeholder_type": "farmers_communities",
            },
            "chat_history": [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
            ],
            "include_trace": False,
        }
    )
    ctx = resolve_request_context(
        user_profile=req.user_profile,
        chat_history=req.chat_history,
        conversation_history=req.conversation_history,
        legacy_stakeholder_type=req.stakeholder_type,
        legacy_audience_instructions=req.audience_instructions,
        session_id=req.session_id,
    )
    assert ctx.stakeholder_type == "farmers_communities"
    assert ctx.has_client_history
    assert len(ctx.history_messages or []) == 2


def test_chat_request_query_or_message() -> None:
    q = ChatRequest.model_validate({"query": "Hello"})
    assert q.user_text() == "Hello"
    m = ChatRequest.model_validate({"message": "Hi there"})
    assert m.user_text() == "Hi there"


def test_chat_request_rejects_missing_user_text() -> None:
    with pytest.raises(ValueError, match="query or message is required"):
        ChatRequest.model_validate({"session_id": "x"})


def test_resolve_prior_memory_accepts_history_messages() -> None:
    sid, summary, recent = _resolve_prior_memory(
        "sess-1",
        [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer"},
        ],
    )
    assert sid == "sess-1"
    assert len(recent) == 2


def test_resolve_prior_memory_no_history_uses_empty_summary() -> None:
    sid, summary, recent = _resolve_prior_memory(None, None)
    assert sid
    assert summary == ""
    assert recent == []


def test_chat_request_allows_profile_stakeholder_with_session() -> None:
    req = ChatRequest.model_validate(
        {
            "query": "Hello",
            "session_id": "abc123",
            "user_profile": {"stakeholder_type": "farmers_communities", "country": "Ghana"},
        }
    )
    assert req.session_id == "abc123"
    assert req.user_profile is not None
    assert req.user_profile.stakeholder_type == "farmers_communities"
