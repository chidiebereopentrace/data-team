"""Unit tests for African / local-language English enrich."""

from __future__ import annotations

from unittest import mock

import pytest

from ml.rag.chatbot.language_enrich import (
    clear_language_enrich_cache,
    enrich_query_to_english,
    lang_enrich_enabled,
    needs_language_enrich,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_language_enrich_cache()
    yield
    clear_language_enrich_cache()


def test_needs_enrich_respects_flag_and_lang(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    assert not lang_enrich_enabled()
    assert not needs_language_enrich("ig")

    monkeypatch.setenv("RAG_LANG_ENRICH", "1")
    assert needs_language_enrich("ig")
    assert needs_language_enrich("sw")
    assert needs_language_enrich("yo")
    assert needs_language_enrich("fr")
    assert needs_language_enrich("ha")
    assert needs_language_enrich("pt")
    assert not needs_language_enrich("en")
    assert not needs_language_enrich("mixed")
    assert not needs_language_enrich("unknown")


def test_english_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LANG_ENRICH", "1")
    with mock.patch("ml.rag.chatbot.language_enrich.llm_chat_complete") as llm:
        out = enrich_query_to_english("maize yields in Kenya 2020", answer_lang="en")
    llm.assert_not_called()
    assert out.needs_enrich is False
    assert out.english_query == ""
    assert out.enrich_source == "none"


def test_igbo_enrich_calls_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LANG_ENRICH", "1")
    q = "kedu ihe mere ka mmeputa akpu n'obodo niajiria gbawe kemgbe afo ise ugbua?"
    with mock.patch(
        "ml.rag.chatbot.language_enrich.llm_chat_complete",
        return_value="What caused cassava production in Nigeria to change over the last five years?",
    ) as llm:
        out = enrich_query_to_english(q, answer_lang="ig")
    llm.assert_called_once()
    assert out.needs_enrich is True
    assert out.enrich_source == "llm"
    assert "cassava" in out.english_query.lower()
    assert "nigeria" in out.english_query.lower()


@pytest.mark.parametrize(
    "lang,query",
    [
        ("sw", "habari ya bei za mahindi Kenya"),
        ("yo", "bawo ni owo agbado se ye ni Naijiria"),
        ("fr", "quel est le prix du maïs au Kenya"),
        ("ha", "yaya farashin masara a Najeriya"),
        ("pt", "qual o preço do milho na Nigéria"),
    ],
)
def test_local_langs_trigger_enrich(
    monkeypatch: pytest.MonkeyPatch, lang: str, query: str
) -> None:
    monkeypatch.setenv("RAG_LANG_ENRICH", "1")
    with mock.patch(
        "ml.rag.chatbot.language_enrich.llm_chat_complete",
        return_value="Maize prices in Kenya or Nigeria",
    ) as llm:
        out = enrich_query_to_english(query, answer_lang=lang)
    llm.assert_called_once()
    assert out.needs_enrich is True
    assert out.english_query


def test_enrich_cache_second_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LANG_ENRICH", "1")
    q = "habari ya bei za mahindi Kenya"
    with mock.patch(
        "ml.rag.chatbot.language_enrich.llm_chat_complete",
        return_value="What are maize prices in Kenya?",
    ) as llm:
        first = enrich_query_to_english(q, answer_lang="sw")
        second = enrich_query_to_english(q, answer_lang="sw")
    assert llm.call_count == 1
    assert first.enrich_source == "llm"
    assert second.enrich_source == "cache"
    assert second.english_query == first.english_query


def test_llm_failure_returns_empty_enrich(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LANG_ENRICH", "1")
    with mock.patch(
        "ml.rag.chatbot.language_enrich.llm_chat_complete",
        return_value="",
    ):
        out = enrich_query_to_english("kedu ihe mere ka mmeputa akpu", answer_lang="ig")
    assert out.needs_enrich is True
    assert out.english_query == ""
    assert out.enrich_source == "none"
