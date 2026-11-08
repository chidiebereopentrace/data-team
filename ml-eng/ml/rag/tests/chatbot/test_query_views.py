"""Tests for three query views (user / context / detail)."""
from __future__ import annotations

from ml.rag.chatbot.query_views import (
    QueryViews,
    build_detail_rewrite,
    build_query_views,
    user_query_dropped,
)


def test_panel_query_keeps_full_user_and_detail_geos() -> None:
    q = (
        "Compare crop and livestock production yearly for Ghana versus Cameroon "
        "and include trade context"
    )
    views = build_query_views(
        q,
        decomposition={
            "geography": ["Ghana", "Cameroon"],
            "entities": ["crop", "livestock"],
            "primary_measures": ["production", "livestock"],
            "time_start": "2015-01-01",
            "time_end": "2024-12-31",
        },
    )
    assert "Ghana" in views.user_query
    assert "Cameroon" in views.user_query
    assert "livestock" in views.user_query.lower()
    assert len(views.user_query.split()) > 6
    texts = views.vector_texts()
    assert texts[0] == views.user_query
    assert not user_query_dropped(views, texts)
    joined = " ".join(texts).lower()
    assert "ghana" in joined and "cameroon" in joined
    assert "livestock" in joined


def test_unrelated_history_does_not_apply_context() -> None:
    views = build_query_views(
        "What is HDI in Senegal?",
        conversation_summary="User asked about maize production in Kenya.",
        recent_turns=[
            {"role": "user", "content": "Show maize production in Kenya 2020"},
            {"role": "assistant", "content": "..."},
        ],
    )
    assert views.context_applied is False
    assert views.context_rewrite == ""
    assert views.user_query.lower().startswith("what is hdi")


def test_elliptical_follow_up_applies_context() -> None:
    views = build_query_views(
        "what about Nigeria?",
        recent_turns=[
            {"role": "user", "content": "Maize production in Kenya 2020"},
            {"role": "assistant", "content": "Kenya produced ..."},
        ],
    )
    assert views.context_applied is True
    assert "maize" in views.context_rewrite.lower()
    assert "nigeria" in views.context_rewrite.lower()
    assert views.user_query.lower().startswith("what about")


def test_profile_alone_does_not_invent_measures() -> None:
    views = build_query_views(
        "How is rainfall looking?",
        user_profile={"country": "Kenya", "category": "Farmers"},
        plan_type="Farmers",
        category="Farmers",
    )
    assert "production" not in views.user_query.lower()
    assert "rainfall" in views.user_query.lower()
    # Scope hints may appear only in context companion
    if views.context_rewrite:
        assert "Focus country: Kenya" in views.context_rewrite or views.context_applied


def test_detail_rewrite_empty_when_equal() -> None:
    assert build_detail_rewrite("maize in Kenya", {}) == ""


def test_vector_texts_dedupe() -> None:
    views = QueryViews(
        user_query="maize Kenya",
        context_rewrite="maize Kenya",
        detail_rewrite="maize Kenya | Geography: Kenya",
        context_applied=False,
    )
    texts = views.vector_texts()
    assert texts[0] == "maize Kenya"
    assert len(texts) <= 2
