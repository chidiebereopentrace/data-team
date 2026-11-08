"""Context isolation: unrelated statements carry no slots; real follow-ups carry the active request."""

from __future__ import annotations

import pytest

from ml.rag.chatbot.active_request import resolve_turn

PRIOR = [{"role": "user", "content": "maize production in Kasese, Uganda"}]


@pytest.mark.parametrize(
    "message",
    ["my cow is sick", "treat a sick cow", "i need help", "hi", "merci", "asante", "sannu", "kedu", "sawubona"],
)
def test_unrelated_or_social_turn_carries_nothing(message: str) -> None:
    turn, persisted = resolve_turn(message, persisted=None, recent_turns=PRIOR)
    assert turn.carried() == {}
    assert persisted.value("commodity") == ["maize"]


@pytest.mark.parametrize("message", ["same crop", "and Nigeria?", "how about Ghana", "kenya"])
def test_follow_up_carries_the_commodity(message: str) -> None:
    turn, _ = resolve_turn(message, persisted=None, recent_turns=PRIOR)
    assert turn.value("commodity") == ["maize"]
    assert turn.slots["commodity"].source == "carried"


def test_named_country_overrides_the_carried_place() -> None:
    turn, _ = resolve_turn("and Nigeria?", persisted=None, recent_turns=PRIOR)
    assert turn.value("geographies") == ["Nigeria"]
    assert turn.slots["geographies"].source == "question"
