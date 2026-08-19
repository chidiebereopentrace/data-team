"""Unit tests for named answer-language detection and mirroring."""
from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.answer_language import (
    SUPPORTED_ANSWER_LANGUAGES,
    _COMMONLINGUA_TO_ADZA,
    _commonlingua_detect,
    detect_answer_language,
    insufficient_context_answer,
    is_english_answer_lang,
    language_instruction,
    language_unclear_answer,
)
from ml.rag.chatbot.assistant_identity import generate_meta_answer, is_meta_query
from ml.rag.chatbot.generator import _build_prompt, _strip_doc_table_figure_labels


def test_english_default() -> None:
    assert detect_answer_language("What are maize yields in Kenya?") == "en"
    assert is_english_answer_lang("en")
    assert "business English" in language_instruction("en")


def test_swahili_named_and_mirror() -> None:
    lang = detect_answer_language("Habari, nipe taarifa za kilimo na mazao Kenya.")
    assert lang == "sw"
    instr = language_instruction(lang)
    assert "business English" not in instr
    assert "Swahili" in instr


def test_french_named() -> None:
    assert detect_answer_language("Bonjour, pourquoi la secheresse affecte le rendement?") == "fr"


def test_pidgin_named_or_mixed() -> None:
    assert detect_answer_language("Wetin be the maize price for Abuja?") in ("pcm", "mixed")


def test_igbo_named() -> None:
    assert detect_answer_language("Kedu, biko gwa m banyere oru ugbo.") == "ig"
    assert detect_answer_language("kedu obodo kacha ako ji na mba africa") == "ig"
    assert "Igbo" in language_instruction("ig")


def test_arabic_script() -> None:
    # Arabic script sample (maize prices question).
    assert detect_answer_language("\u0645\u0627 \u0647\u064a \u0623\u0633\u0639\u0627\u0631 \u0627\u0644\u0630\u0631\u0629\u061f") == "ar"
    assert not is_english_answer_lang("ar")
    assert "Arabic" in language_instruction("ar")


def test_amharic_script() -> None:
    # Amharic script sample.
    assert detect_answer_language("\u12e8\u1260\u1246\u120e \u121d\u122d\u1275 \u121d\u1295\u12f5\u1295 \u1290\u12cd\u1362") == "am"


def test_unknown_lists_supported_languages() -> None:
    lang = detect_answer_language("\xff\xf8\xfc \xff\xf8\xfc \xff\xf8\xfc")
    assert lang == "unknown"
    help_text = language_unclear_answer()
    assert "Igbo" in help_text
    assert "Swahili" in help_text
    assert "Zulu" in help_text
    assert "Kinyarwanda" in help_text
    assert "(ig)" in help_text
    assert "(zu)" in help_text


def test_zulu_somali_wolof_kinyarwanda_named() -> None:
    assert detect_answer_language("Sawubona, unjani? Ngiyabonga ngokulima.") == "zu"
    assert detect_answer_language("Molo, enkosi. Kunjani?") == "xh"
    assert detect_answer_language("Salaan, sidee tahay? Beeraha.") == "so"
    assert detect_answer_language("Salaamalekum, jerejef. Nanga def?") == "wo"
    assert detect_answer_language("Muraho, murakoze. Amakuru y'ubuhinzi?") == "rw"
    assert "Zulu" in language_instruction("zu")
    assert "Somali" in language_instruction("so")


def test_build_prompt_mirrors_swahili() -> None:
    messages = _build_prompt(
        "Habari, nipe taarifa za kilimo Kenya.",
        "[Source 1]\nMaize yields rose.",
        answer_lang="sw",
    )
    system = messages[0]["content"]
    assert "business English" not in system
    assert "Swahili" in system
    assert "Never refer" in system


def test_build_prompt_english_keeps_business_english() -> None:
    messages = _build_prompt("What are maize yields in Kenya?", "[Source 1]\nx")
    assert "business English" in messages[0]["content"]


def test_strip_doc_table_figure_labels() -> None:
    raw = (
        "According to Table 6.1 in the report, yields rose.[2] "
        "See Figure 2 for trends."
    )
    cleaned = _strip_doc_table_figure_labels(raw)
    assert "Table 6.1" not in cleaned
    assert "Figure 2" not in cleaned
    assert "yields rose" in cleaned


def test_meta_skips_static_for_non_english(monkeypatch) -> None:
    monkeypatch.setenv("RAG_META_RESPONSES", "hybrid")
    with mock.patch("ml.rag.chatbot.generator._call_llama", return_value="Mimi ni Ask ADZA.") as llm:
        out = generate_meta_answer("Wewe ni nani?")
    llm.assert_called_once()
    assert "Ask ADZA" in out
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
    fr = insufficient_context_answer(query="Bonjour, pourquoi pas de donnees?")
    sw = insufficient_context_answer(query="Habari, nipatie taarifa za kilimo.")
    assert "fiables" in fr.lower() or "informations" in fr.lower()
    assert "taarifa" in sw.lower() or "Sina" in sw
    assert fr != insufficient_context_answer("en")


def test_commonlingua_fallback_triggers_on_en() -> None:
    """When regex returns 'en' but CommonLingua detects an African language, use the model."""
    with mock.patch(
        "ml.rag.chatbot.answer_language._commonlingua_detect",
        return_value="ha",
    ):
        assert detect_answer_language("Ina so in san farashin hatsi") == "ha"


def test_commonlingua_fallback_triggers_on_unknown() -> None:
    """When regex returns 'unknown' but CommonLingua identifies a language, use it."""
    with mock.patch(
        "ml.rag.chatbot.answer_language._commonlingua_detect",
        return_value="yo",
    ):
        assert detect_answer_language("\xff\xf8\xfc \xff\xf8\xfc \xff\xf8\xfc") == "yo"


def test_commonlingua_graceful_import_error() -> None:
    """If commonlid is not installed, _commonlingua_detect returns None."""
    with mock.patch(
        "ml.rag.chatbot.answer_language._load_commonlingua",
        side_effect=ImportError("No module named 'commonlid'"),
    ):
        assert _commonlingua_detect("some long enough text for detection") is None


def test_commonlingua_skips_short_text() -> None:
    assert _commonlingua_detect("hi") is None


def test_commonlingua_mapping_covers_supported_codes() -> None:
    """Every non-mixed supported ADZA code has at least one CommonLingua source."""
    adza_codes = {code for code, _ in SUPPORTED_ANSWER_LANGUAGES if code != "mixed"}
    mapped_codes = set(_COMMONLINGUA_TO_ADZA.values())
    missing = adza_codes - mapped_codes
    assert not missing, f"ADZA codes without CommonLingua mapping: {missing}"


def test_insufficient_igbo_falls_back_english() -> None:
    text = insufficient_context_answer("non_en")
    assert "reliable information" in text
    ig = insufficient_context_answer(query="Kedu, biko gwa m.")
    assert "reliable information" in ig
