"""Unit tests for language-agnostic answer mirroring."""
from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    insufficient_context_answer,
    is_english_answer_lang,
    language_instruction,
)
from ml.rag.chatbot.assistant_identity import generate_meta_answer, is_meta_query
from ml.rag.chatbot.generator import _build_prompt


def test_english_default() -> None:
    assert detect_answer_language("What are maize yields in Kenya?") == "en"
    assert is_english_answer_lang("en")
    assert "business English" in language_instruction("en")


def test_swahili_non_en_and_mirror() -> None:
    lang = detect_answer_language("Habari, nipe taarifa za kilimo na mazao Kenya.")
    assert lang == "non_en"
    instr = language_instruction(lang)
    assert "business English" not in instr
    assert "Igbo" in instr or "user" in instr.lower()
    assert "Swahili" in instr or "Pidgin" in instr


def test_french_non_en() -> None:
    assert detect_answer_language("Bonjour, pourquoi la sécheresse affecte le rendement?") == "non_en"


def test_pidgin_non_en() -> None:
    assert detect_answer_language("Wetin be the maize price for Abuja?") in ("non_en", "mixed")


def test_igbo_marker_non_en() -> None:
    assert detect_answer_language("Kedu, biko gwa m banyere ọrụ ugbo.") == "non_en"


def test_arabic_script() -> None:
    assert detect_answer_language("ما هي أسعار الذرة في كينيا؟") == "ar"
    assert not is_english_answer_lang("ar")
    assert "Igbo" in language_instruction("ar")


def test_amharic_script() -> None:
    assert detect_answer_language("የበቆሎ ምርት ምንድን ነው?") == "am"


def test_build_prompt_mirrors_swahili() -> None:
    messages = _build_prompt(
        "Habari, nipe taarifa za kilimo Kenya.",
        "[Source 1]\nMaize yields rose.",
    )
    system = messages[0]["content"]
    assert "business English" not in system
    assert "Igbo" in system or "Pidgin" in system


def test_build_prompt_english_keeps_business_english() -> None:
    messages = _build_prompt("What are maize yields in Kenya?", "[Source 1]\nx")
    assert "business English" in messages[0]["content"]


def test_meta_skips_static_for_non_english(monkeypatch) -> None:
    monkeypatch.setenv("RAG_META_RESPONSES", "hybrid")
    with mock.patch("ml.rag.chatbot.generator._call_llama", return_value="Mimi ni Ask ADZA.") as llm:
        out = generate_meta_answer("Wewe ni nani?")
    llm.assert_called_once()
    assert "Ask ADZA" in out
    # Must not be the long English static identity blurb alone
    assert "AI-powered advisory interface" not in out


def test_meta_static_for_english(monkeypatch) -> None:
    monkeypatch.setenv("RAG_META_RESPONSES", "hybrid")
    with mock.patch("ml.rag.chatbot.generator._call_llama") as llm:
        out = generate_meta_answer("Who are you?")
    llm.assert_not_called()
    assert "Ask ADZA" in out
    assert "AI-powered advisory interface" in out


def test_meta_patterns_fr_sw_pcm() -> None:
    assert is_meta_query("Qui es-tu?")
    assert is_meta_query("Wewe ni nani?")
    assert is_meta_query("Who you be?")


def test_insufficient_fr_sw() -> None:
    fr = insufficient_context_answer(query="Bonjour, pourquoi pas de données?")
    sw = insufficient_context_answer(query="Habari, nipatie taarifa za kilimo.")
    assert "fiables" in fr.lower() or "informations" in fr.lower()
    assert "taarifa" in sw.lower() or "Sina" in sw
    assert fr != insufficient_context_answer("en")


def test_insufficient_non_en_falls_back_english() -> None:
    # Soft tag non_en without specialty canned → English
    text = insufficient_context_answer("non_en")
    assert "reliable information" in text
    # Igbo-style query without specialty canned lang → English canned
    ig = insufficient_context_answer(query="Kedu, biko gwa m.")
    assert "reliable information" in ig
