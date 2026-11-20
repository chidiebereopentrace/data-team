"""Generation prompt must carry the current date (ML-057, Sprint 2 Week 2).

The decomposer already grounds time ranges in ``date.today()``; before this
change the generation prompt did not, so the LLM fell back on its training
cut-off and could not separate upcoming events from past ones.
"""

from __future__ import annotations

from datetime import date

from ml.rag.chatbot.generation_prompt import _build_prompt, _current_date_block


def _system_message(messages: list[dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "system":
            return str(msg.get("content") or "")
    raise AssertionError("no system message in prompt")


def test_date_block_states_the_pinned_date() -> None:
    block = _current_date_block(date(2026, 10, 3))
    assert "2026-10-03" in block
    assert "year 2026" in block


def test_date_block_warns_against_prior_year_as_current() -> None:
    # 2025 figures must not be presented as current when it is 2026.
    block = _current_date_block(date(2026, 10, 3))
    assert "2025" in block


def test_date_block_defaults_to_today() -> None:
    block = _current_date_block()
    assert date.today().isoformat() in block


def test_build_prompt_injects_pinned_date_into_system_message() -> None:
    messages = _build_prompt(
        "maize production in Kenya",
        "Context block",
        reference_date=date(2026, 10, 3),
    )
    system = _system_message(messages)
    assert "CURRENT DATE" in system
    assert "2026-10-03" in system


def test_build_prompt_injects_date_without_explicit_reference() -> None:
    messages = _build_prompt("maize production in Kenya", "Context block")
    system = _system_message(messages)
    assert "CURRENT DATE" in system
    assert date.today().isoformat() in system


def test_date_block_precedes_evidence_rules() -> None:
    """Date must land before downstream addenda so they read in-year."""
    messages = _build_prompt(
        "maize production in Kenya",
        "Context block",
        evidence_tier="partial",
        reference_date=date(2026, 10, 3),
    )
    system = _system_message(messages)
    assert "CURRENT DATE" in system
    assert "PARTIAL EVIDENCE" in system
    assert system.index("CURRENT DATE") < system.index("PARTIAL EVIDENCE")


def test_user_message_does_not_carry_the_date_block() -> None:
    messages = _build_prompt(
        "maize production in Kenya",
        "Context block",
        reference_date=date(2026, 10, 3),
    )
    user = next(m for m in messages if m.get("role") == "user")
    assert "CURRENT DATE" not in str(user.get("content") or "")
