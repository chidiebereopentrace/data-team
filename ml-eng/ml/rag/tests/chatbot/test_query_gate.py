"""Unit tests for greeting / out-of-scope query gate."""

from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.query_gate import (
    classify_social_query,
    early_non_rag_route,
    generate_social_answer,
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
    assert is_greeting_query("merci")
    assert is_greeting_query("asante sana")
    assert is_greeting_query("sannu")
    assert is_greeting_query("kedu")
    assert is_greeting_query("bawo")
    assert is_greeting_query("sawubona")
    assert is_greeting_query("molo")
    assert is_greeting_query("muraho")
    assert is_greeting_query("olá")
    assert is_greeting_query("how far")


def test_is_greeting_not_agronomy() -> None:
    assert not is_greeting_query("maize yields in Kenya 2020")
    assert not is_greeting_query("hello Africa fertilizer trends")
    assert not is_greeting_query("who are you")


def test_early_non_rag_route_priority() -> None:
    assert early_non_rag_route("Who are you?") == "meta"
    assert early_non_rag_route("What is OpenTrace?") == "product"
    assert early_non_rag_route("hi") == "greeting"
    assert early_non_rag_route("tell me a joke") == "out_of_scope"
    assert early_non_rag_route("maize yields in Kenya 2020") is None


def test_early_non_rag_route_help_incident() -> None:
    incident = "what is your use, and what can i use AskADZA for"
    assert early_non_rag_route(incident) == "help"
    assert early_non_rag_route("what can I use you for") == "help"
    assert early_non_rag_route("what questions can I ask") == "help"


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
    assert "natural-language interface" in g.lower()
    assert "agricultural intelligence layer" in g.lower()
    assert "advisory assistant" not in g.lower()
    assert "opentrace.africa" in g.lower()
    assert "outside" in o.lower() or "focused" in o.lower() or "interface" in o.lower()
    assert "advisory assistant" not in o.lower()
    assert "opentrace.africa" in o.lower()


def test_french_social_template_no_llm() -> None:
    with mock.patch("ml.rag.chatbot.generator._call_llama") as llm:
        out = generate_social_answer("greeting", "bonjour", answer_lang="fr")
    llm.assert_not_called()
    assert "Ask ADZA" in out or "ADZA" in out


def test_english_social_no_llm() -> None:
    with mock.patch("ml.rag.chatbot.generator._call_llama") as llm:
        out = generate_social_answer("greeting", "hi", answer_lang="en")
    llm.assert_not_called()
    assert "Ask ADZA" in out
