"""Unit tests for product knowledge routing and KB loading."""

from __future__ import annotations

from unittest.mock import patch

from ml.rag.chatbot.assistant_identity import META_ANSWER_FOOTER
from ml.rag.chatbot.product_knowledge import (
    classify_product_subroute,
    format_product_kb_for_prompt,
    generate_product_answer,
    is_help_query,
    is_product_query,
    load_product_kb,
    static_capability_answer,
)


def test_load_product_kb_has_expected_keys() -> None:
    load_product_kb.cache_clear()
    kb = load_product_kb()
    assert "partnership_briefing" in str(kb.get("source", ""))
    assert "aim" in kb
    assert "capabilities" in kb
    assert "pillars" in kb
    assert "ask_adza" in kb["pillars"]
    assert "acf" in kb["pillars"]
    assert "ofia" in kb["pillars"]
    assert "contact" in kb


def test_format_product_kb_under_budget() -> None:
    load_product_kb.cache_clear()
    text = format_product_kb_for_prompt()
    assert "OpenTrace Africa" in text
    assert "OFIA" in text or "ofia" in text.lower()
    assert "ACF" in text or "acf" in text.lower()
    assert "data reconstruction" in text.lower() or "Data reconstruction" in text
    assert len(text) < 25000


def test_is_product_query_positive() -> None:
    assert is_product_query("What is the aim of OpenTrace?")
    assert is_product_query("Explain ACF confidence tiers")
    assert is_product_query("Tell me about OpenTrace")
    assert is_product_query("What is OFIA?")
    assert is_product_query("Why partner with OpenTrace?")
    assert is_product_query("c'est quoi OpenTrace")
    assert is_product_query("OpenTrace ni nini")
    assert is_product_query("wetin be OpenTrace")
    assert is_product_query("what is Ask ADZA")
    assert is_product_query("what is AskADZA")


def test_is_product_query_negative_ag_data() -> None:
    assert not is_product_query(
        "OpenTrace data on maize yields in Kenya",
        {"geography": ["Kenya"], "entities": ["maize"]},
    )
    assert not is_product_query("maize yields in Kenya 2020")
    assert not is_product_query("Rice policy in Senegal")
    assert not is_product_query(
        "What is the trend of rice production in Senegal",
        {"geography": ["Senegal"], "entities": ["rice"]},
    )


def test_is_product_query_identity_not_product() -> None:
    # Identity queries are handled by is_meta_query, not product
    assert not is_product_query("Who are you?")


_INCIDENT_QUERY = "what is your use, and what can i use AskADZA for"


def test_is_help_query_incident() -> None:
    assert is_help_query(_INCIDENT_QUERY)
    assert is_product_query(_INCIDENT_QUERY)
    assert classify_product_subroute(_INCIDENT_QUERY) == "help"


def test_is_help_query_capability_phrases() -> None:
    assert is_help_query("what can I use you for")
    assert is_help_query("how do I use Ask ADZA")
    assert is_help_query("what questions can I ask")
    assert is_product_query("what can I use you for")
    assert is_product_query("how do I use Ask ADZA")


def test_askadza_one_word_brand_normalization() -> None:
    assert is_help_query("what can i use AskADZA for")
    assert is_product_query("what is AskADZA")


def test_is_help_query_negative_methodology() -> None:
    assert not is_help_query("how does IPC work")
    assert not is_product_query("how does IPC work")


def test_is_help_query_product_brand_mechanics() -> None:
    assert is_help_query("how does Ask ADZA work")
    assert is_product_query("how does Ask ADZA work")


def test_static_capability_answer_content() -> None:
    ans = static_capability_answer("what can I use you for")
    assert "natural-language interface" in ans.lower()
    assert "maize" in ans.lower()
    assert "rice" in ans.lower()
    assert "Citations" not in ans
    assert META_ANSWER_FOOTER.strip() in ans or "opentrace.africa" in ans.lower()


def test_generate_product_answer_help_uses_static() -> None:
    with patch("ml.rag.chatbot.generator._call_llama") as llama:
        ans = generate_product_answer("what can I use you for")
    llama.assert_not_called()
    assert "maize" in ans.lower()
    assert "Citations" not in ans


def test_generate_product_answer_uses_kb_and_footer() -> None:
    with patch("ml.rag.chatbot.generator._call_llama", return_value="OpenTrace builds Africa's intelligence layer."):
        ans = generate_product_answer("What is the aim of OpenTrace?")
    assert "intelligence layer" in ans.lower()
    assert "Citations" not in ans
    assert META_ANSWER_FOOTER.strip() in ans or "opentrace.africa" in ans.lower()
