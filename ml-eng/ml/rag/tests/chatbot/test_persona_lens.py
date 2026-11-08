"""Persona lens: Streamlit defaults, session profile, category instructions, domain fill."""

from __future__ import annotations

from unittest.mock import patch

from ml.rag.chat_turn import empty_session_blob, persist_session_turn
from ml.rag.chatbot.generator import _build_prompt
from ml.rag.chatbot.qa_run_kwargs import build_run_kwargs
from ml.rag.chatbot.stakeholder_prompts import instruction_for_category
from ml.rag.request_context import resolve_request_context


def test_streamlit_blank_plan_and_persona_stay_unset() -> None:
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
    assert kwargs.get("plan_type") is None
    assert kwargs.get("category") is None
    assert kwargs["export_enabled"] is False
    assert not kwargs.get("user_profile")


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


def test_session_persist_profile_and_active_request() -> None:
    store: dict[tuple[str, str], dict] = {}

    def _save(owner: str, sid: str, blob: dict, *, expected_version: int) -> bool:
        store[(owner, sid)] = {**blob, "state_version": expected_version + 1}
        return True

    with patch("ml.rag.chat_turn.save_session_blob", side_effect=_save):
        ok = persist_session_turn(
            "tenant:t1",
            "sess-persona",
            "How is rainfall?",
            "Rain looks mixed.",
            loaded=empty_session_blob(),
            result={"active_request": {"slots": {"commodity": {"value": ["maize"]}}}},
            category="Farmers",
            plan_type="Farmers",
            country="Ghana",
        )
    assert ok
    blob = store[("tenant:t1", "sess-persona")]
    assert blob["category"] == "Farmers"
    assert blob["plan_type"] == "Farmers"
    assert blob["country"] == "Ghana"
    assert blob["active_request"]["slots"]["commodity"]["value"] == ["maize"]
    assert blob["state_version"] == 1


def test_request_context_session_fallback_plan_and_country() -> None:
    blob = empty_session_blob()
    blob.update({"category": "Farmers", "plan_type": "Farmers", "country": "Ghana"})

    ctx = resolve_request_context(session_blob=blob, user_profile=None)
    assert ctx.plan_type == "Farmers"
    assert ctx.category == "Farmers"
    assert ctx.user_profile == {
        "country": "GHA",
        "plan_type": "Farmers",
        "category": "Farmers",
    }


def test_farmers_instruction_actionable_plain() -> None:
    text = instruction_for_category("Farmers").lower()
    assert "plain" in text
    assert "actionable" in text or "next step" in text


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
