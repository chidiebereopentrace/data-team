"""Unit tests for greeting / out-of-scope query gate."""

from __future__ import annotations

from ml.rag.chatbot.query_gate import (
    classify_social_query,
    is_greeting_query,
    is_out_of_scope_query,
    static_social_answer,
)


def test_is_greeting_query() -> None:
    assert is_greeting_query("hi")
    assert is_greeting_query("Hello!")
    assert is_greeting_query("hey")
    assert is_greeting_query("good morning")
    assert is_greeting_query("thanks")
    assert is_greeting_query("bonjour")
    assert is_greeting_query("jambo")


def test_is_greeting_not_agronomy() -> None:
    assert not is_greeting_query("maize yields in Kenya 2020")
    assert not is_greeting_query("hello Africa fertilizer trends")
    assert not is_greeting_query("who are you")


def test_out_of_scope_joke_and_ok() -> None:
    empty = {"entities": [], "geography": [], "domains": []}
    assert is_out_of_scope_query("tell me a joke", empty)
    assert is_out_of_scope_query("ok", empty)
    assert classify_social_query("tell me a joke", empty) == "out_of_scope"
    assert classify_social_query("ok", empty) == "out_of_scope"


def test_out_of_scope_false_for_kenya_maize() -> None:
    dec = {
        "entities": ["maize"],
        "geography": ["Kenya"],
        "domains": ["agriculture"],
    }
    assert not is_out_of_scope_query("maize yields in Kenya", dec)
    assert not is_out_of_scope_query("maize yields in Kenya", {"entities": [], "geography": [], "domains": []})
    assert classify_social_query("maize yields in Kenya") is None


def test_greeting_wins_over_out_of_scope() -> None:
    empty = {"entities": [], "geography": [], "domains": []}
    assert classify_social_query("hello", empty) == "greeting"
    assert not is_out_of_scope_query("hello", empty)


def test_static_social_answers() -> None:
    g = static_social_answer("greeting")
    o = static_social_answer("out_of_scope")
    assert "Ask ADZA" in g
    assert "opentrace.africa" in g.lower()
    assert "outside" in o.lower() or "focused" in o.lower()
    assert "opentrace.africa" in o.lower()
