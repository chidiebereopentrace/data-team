"""Unit tests for shared request context resolution."""

from __future__ import annotations

import pytest

from ml.rag.api_schemas import UserProfile
from ml.rag.app.api import ChatMessage, QueryRequest, _resolve_prior_memory
from ml.rag.request_context import (
    bootstrap_category,
    effective_chat_history_messages,
    resolve_request_context,
)


def test_resolve_from_user_profile() -> None:
    ctx = resolve_request_context(
        user_profile=UserProfile(
            country="Ghana",
            plan_type="Farmers",
            category="Farmers",
        ),
    )
    assert ctx.plan_type == "Farmers"
    assert ctx.category == "Farmers"
    assert ctx.user_profile == {
        "country": "Ghana",
        "plan_type": "Farmers",
        "category": "Farmers",
    }


def test_legacy_top_level_fields_removed() -> None:
    with pytest.raises(ValueError, match="invalid plan_type"):
        resolve_request_context(
            user_profile={"country": "Ghana", "plan_type": "bad", "category": "Farmers"},
        )


def test_invalid_category() -> None:
    with pytest.raises(ValueError, match="invalid category"):
        resolve_request_context(
            user_profile={
                "country": "Ghana",
                "plan_type": "Free",
                "category": "NotARealCategory",
            },
        )


def test_bootstrap_category() -> None:
    cat = bootstrap_category(
        UserProfile(country=None, plan_type="Integrated", category="Government"),
    )
    assert cat == "Government"


def test_chat_history_alias() -> None:
    msgs = effective_chat_history_messages(
        None,
        [{"role": "user", "content": "Hi"}],
    )
    assert msgs == [{"role": "user", "content": "Hi"}]


def test_query_request_accepts_canonical_shape() -> None:
    req = QueryRequest.model_validate(
        {
            "query": "Rice trends?",
            "user_profile": {
                "country": "Ghana",
                "plan_type": "Government",
                "category": "Government",
            },
            "include_trace": False,
        }
    )
    ctx = resolve_request_context(
        user_profile=req.user_profile,
        chat_history=req.chat_history,
        conversation_history=req.conversation_history,
        session_id=req.session_id,
    )
    assert ctx.plan_type == "Government"
    assert ctx.category == "Government"


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


def test_user_profile_rejects_old_fields() -> None:
    with pytest.raises(ValueError):
        UserProfile.model_validate(
            {
                "country": "Ghana",
                "plan_type": "Farmers",
                "category": "Farmers",
                "stakeholder_type": "farmers_communities",
            }
        )


def test_query_request_rejects_deprecated_top_level_fields() -> None:
    with pytest.raises(ValueError):
        QueryRequest.model_validate(
            {
                "query": "Hello",
                "stakeholder_type": "Government",
            }
        )
