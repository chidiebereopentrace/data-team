"""Persona lens: Streamlit defaults, session profile, category instructions, domain fill."""

from __future__ import annotations

from unittest.mock import patch

from ml.rag.chat_turn import (
    _empty_session_blob,
    _resolve_prior_and_profile,
    persist_session_turn,
)
from ml.rag.chatbot.generator import _build_prompt
from ml.rag.chatbot.plan_policy import apply_category_domain_hints
from ml.rag.chatbot.stakeholder_prompts import instruction_for_category
from ml.rag.chatbot.qa_run_kwargs import build_run_kwargs
from ml.rag.request_context import resolve_request_context


def test_streamlit_defaults_always_send_farmers_profile() -> None:
    kwargs = build_run_kwargs(
        news_top_k=5,
        academic_top_k=5,
        bq_top_k=5,
        ota_top_k=5,
        rerank_top_k=5,
        plan_type="",
        category="",
        profile_country="",
        t_start="",
        t_end="",
        prior_summary="",
        prior_recent=[],
    )
    assert kwargs["plan_type"] == "Farmers"
    assert kwargs["category"] == "Farmers"
    assert kwargs["export_enabled"] is False
    assert kwargs["user_profile"] == {
        "country": None,
        "plan_type": "Farmers",
        "category": "Farmers",
    }


def test_streamlit_kwargs_respect_overrides() -> None:
    kwargs = build_run_kwargs(
        news_top_k=5,
        academic_top_k=5,
        bq_top_k=5,
        ota_top_k=5,
        rerank_top_k=5,
        plan_type="Government",
        category="Government",
        profile_country="Kenya",
        t_start="",
        t_end="",
        prior_summary="",
        prior_recent=[],
    )
    assert kwargs["plan_type"] == "Government"
    assert kwargs["category"] == "Government"
    assert kwargs["export_enabled"] is False
    assert kwargs["user_profile"]["country"] == "Kenya"


def test_streamlit_agribusinesses_enables_export() -> None:
    kwargs = build_run_kwargs(
        news_top_k=5,
        academic_top_k=5,
        bq_top_k=5,
        ota_top_k=5,
        rerank_top_k=5,
        plan_type="Agribusinesses",
        category="Agribusinesses",
        profile_country="Nigeria",
        t_start="",
        t_end="",
        prior_summary="",
        prior_recent=[],
    )
    assert kwargs["export_enabled"] is True
    assert kwargs["plan_type"] == "Agribusinesses"


def test_session_persist_and_fallback_profile() -> None:
    store: dict[str, dict] = {}

    def _get(sid: str):
        return store.get(sid)

    def _save(sid: str, blob: dict) -> None:
        store[sid] = dict(blob)

    with patch("ml.rag.chat_turn.get_session_blob", side_effect=_get), patch(
        "ml.rag.chat_turn.save_session_blob", side_effect=_save
    ):
        persist_session_turn(
            "sess-persona",
            "How is rainfall?",
            "Rain looks mixed.",
            category="Farmers",
            plan_type="Farmers",
            country="Ghana",
        )
        blob = store["sess-persona"]
        assert blob["category"] == "Farmers"
        assert blob["plan_type"] == "Farmers"
        assert blob["country"] == "Ghana"

        sid, _summary, _recent, cat, plan, country, found = _resolve_prior_and_profile(
            "sess-persona",
            None,
            None,
            None,
            None,
        )
        assert sid == "sess-persona"
        assert cat == "Farmers"
        assert plan == "Farmers"
        assert country == "Ghana"
        assert found is True


def test_request_context_session_fallback_plan_and_country() -> None:
    blob = _empty_session_blob()
    blob.update({"category": "Farmers", "plan_type": "Farmers", "country": "Ghana"})

    with patch("ml.rag.request_context.get_session_blob", return_value=blob):
        ctx = resolve_request_context(session_id="sess-1", user_profile=None)
    assert ctx.plan_type == "Farmers"
    assert ctx.category == "Farmers"
    assert ctx.user_profile == {
        "country": "Ghana",
        "plan_type": "Farmers",
        "category": "Farmers",
    }


def test_farmers_instruction_actionable_plain() -> None:
    text = instruction_for_category("Farmers").lower()
    assert "plain" in text
    assert "actionable" in text or "next step" in text


def test_category_domain_soft_fill_when_empty() -> None:
    out = apply_category_domain_hints({"domains": [], "geography": []}, "Farmers")
    assert out["domains"]
    assert "rainfall" in out["domains"] or "yield" in out["domains"]


def test_category_domain_keeps_query_domains() -> None:
    out = apply_category_domain_hints(
        {"domains": ["irrigation"], "geography": ["Kenya"]},
        "Farmers",
    )
    assert out["domains"] == ["irrigation"]


def test_build_prompt_couples_language_and_category() -> None:
    messages = _build_prompt(
        "Kedu ihe mere?",
        "[Source 1]\nx",
        category="Farmers",
        plan_type="Farmers",
        answer_lang="ig",
    )
    system = messages[0]["content"]
    assert "category audience rules" in system.lower() or "plainness" in system.lower()
