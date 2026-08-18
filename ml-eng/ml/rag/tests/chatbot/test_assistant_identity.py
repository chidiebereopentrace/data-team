"""Unit tests for assistant identity meta-query detection and static answers."""

from __future__ import annotations

from ml.rag.chatbot.assistant_identity import (
    META_ANSWER_FOOTER,
    classify_meta_query,
    is_meta_query,
    static_meta_answer,
)
from ml.rag.chatbot.product_knowledge import is_product_query


def test_is_meta_query_identity() -> None:
    assert is_meta_query("Who are you?")
    assert is_meta_query("what is your name")
    assert is_meta_query("What are you doing?")
    assert is_meta_query("Are you a bot?")


def test_is_meta_query_multilingual() -> None:
    assert is_meta_query("qui es-tu")
    assert is_meta_query("wewe ni nani")
    assert is_meta_query("who you be")
    assert is_meta_query("ungubani")


def test_is_meta_query_product_moved_to_product_knowledge() -> None:
    assert not is_meta_query("Tell me about OpenTrace")
    assert not is_meta_query("what is ask adza")
    assert not is_meta_query("what is OFIA")
    assert is_product_query("Tell me about OpenTrace")
    assert is_product_query("what is ask adza")
    assert is_product_query("what is OFIA")


def test_is_meta_query_non_meta() -> None:
    assert not is_meta_query("maize yields in Kenya 2020")
    assert not is_meta_query("which regions have high rainfall")


def test_classify_meta_query_buckets() -> None:
    assert classify_meta_query("Who are you?") == "identity"
    assert classify_meta_query("what is your name") == "name"
    assert classify_meta_query("what do you do") == "role"
    assert classify_meta_query("Tell me about OpenTrace") is None
    assert classify_meta_query("what is OFIA") is None


def test_static_meta_answer_identity() -> None:
    ans = static_meta_answer("identity")
    assert ans is not None
    assert "Ask ADZA" in ans
    assert "OpenTrace" in ans


def test_meta_answer_includes_footer() -> None:
    from ml.rag.chatbot.assistant_identity import generate_meta_answer

    ans = generate_meta_answer("Who are you?")
    assert META_ANSWER_FOOTER.strip() in ans or "opentrace.africa" in ans.lower()
