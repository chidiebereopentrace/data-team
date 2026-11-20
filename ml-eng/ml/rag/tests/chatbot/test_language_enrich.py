"""Unit tests for African / local-language English enrich."""

from __future__ import annotations

from typing import Generator
from unittest import mock

import pytest

from ml.rag.chatbot.language_enrich import (
    LanguageEnrichment,
    clear_language_enrich_cache,
    enrich_query_to_english,
    lang_enrich_enabled,
    needs_language_enrich,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> Generator[None, None, None]:
    clear_language_enrich_cache()
    yield
    clear_language_enrich_cache()


def test_enrich_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    assert lang_enrich_enabled()
    assert needs_language_enrich("ig", confidence=0.9)


def test_enrich_explicit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LANG_ENRICH", "0")
    assert not lang_enrich_enabled()
    assert not needs_language_enrich("ig", confidence=0.9)


def test_needs_enrich_unknown_and_mixed() -> None:
    assert needs_language_enrich("unknown", confidence=0.2)
    assert needs_language_enrich("mixed", confidence=0.0)
    assert needs_language_enrich("sw", confidence=0.9)


def test_needs_enrich_skips_high_conf_english() -> None:
    assert not needs_language_enrich("en", confidence=0.95)
    assert needs_language_enrich("en", confidence=0.4)


def test_english_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    with mock.patch(
        "ml.rag.chatbot.language_enrich.detect_language_with_confidence",
        return_value=("en", 0.95),
    ):
        with mock.patch("ml.rag.chatbot.language_enrich.llm_chat_complete") as llm:
            out = enrich_query_to_english("maize yields in Kenya 2020")
    llm.assert_not_called()
    assert out.needs_enrich is False
    assert out.english_query == ""
    assert out.enrich_source == "none"


def test_osikapa_igbo_enriches_even_if_previously_mistagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    q = "ego ole ka ana-ele osikapa na kenya"
    with mock.patch(
        "ml.rag.chatbot.language_enrich.llm_chat_complete",
        return_value="What is the retail price of rice in Kenya?",
    ) as llm:
        out = enrich_query_to_english(q, answer_lang="ig", confidence=0.85)
    llm.assert_called_once()
    assert out.needs_enrich is True
    assert "rice" in out.english_query.lower()
    assert "kenya" in out.english_query.lower()


def test_osikapa_enriches_when_tagged_en_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mis-tagged English with low confidence still enriches."""
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    q = "ego ole ka ana-ele osikapa na kenya"
    with mock.patch(
        "ml.rag.chatbot.language_enrich.llm_chat_complete",
        return_value="How much does rice cost in Kenya?",
    ) as llm:
        out = enrich_query_to_english(q, answer_lang="en", confidence=0.3)
    llm.assert_called_once()
    assert out.needs_enrich is True
    assert "rice" in out.english_query.lower()


@pytest.mark.parametrize("country", ["Kenya", "Rwanda"])
def test_current_igbo_rice_price_keeps_its_original_crop_and_place(monkeypatch, country):
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    with mock.patch("ml.rag.chatbot.language_enrich.llm_chat_complete") as llm:
        out = enrich_query_to_english(
            f"Kedu ego ole ka ana-ele osikapa na {country} ugbu a?", answer_lang="ig", confidence=0.9,
        )
    llm.assert_not_called()
    assert out.english_query == f"What is the current market price of rice in {country}?"


def test_akpu_cassava_enriches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    q = "kedu ihe mere ka mmeputa akpu n'obodo niajiria gbawe kemgbe afo ise ugbua?"
    with mock.patch(
        "ml.rag.chatbot.language_enrich.llm_chat_complete",
        return_value=(
            "What caused cassava production in Nigeria to change over the last five years?"
        ),
    ) as llm:
        out = enrich_query_to_english(q, answer_lang="ig", confidence=0.9)
    llm.assert_called_once()
    assert "cassava" in out.english_query.lower()
    assert "nigeria" in out.english_query.lower()


def test_enrich_cache_second_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    q = "habari ya bei za mahindi Kenya"
    with mock.patch(
        "ml.rag.chatbot.language_enrich.llm_chat_complete",
        return_value="What are maize prices in Kenya?",
    ) as llm:
        first = enrich_query_to_english(q, answer_lang="sw", confidence=0.9)
        second = enrich_query_to_english(q, answer_lang="sw", confidence=0.9)
    assert llm.call_count == 1
    assert first.enrich_source == "llm"
    assert second.enrich_source == "cache"


def test_llm_failure_returns_empty_enrich(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LANG_ENRICH", raising=False)
    with mock.patch(
        "ml.rag.chatbot.language_enrich.llm_chat_complete",
        return_value="",
    ):
        out = enrich_query_to_english(
            "kedu ihe mere ka mmeputa akpu",
            answer_lang="ig",
            confidence=0.9,
        )
    assert out.needs_enrich is True
    assert out.english_query == ""
    assert out.enrich_source == "none"
    assert isinstance(out, LanguageEnrichment)
