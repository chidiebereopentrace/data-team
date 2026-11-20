"""Degenerate/looping non-English output detection and gating (ML-055)."""

from __future__ import annotations

from ml.rag.chatbot.answer_language import (
    generation_failure_answer,
    looks_like_degenerate_repetition,
)
from ml.rag.chatbot.generator import _finalize_generation_result, _guard_degenerate_language_output, generate


def test_french_report_can_repeat_indicator_names_without_being_rejected() -> None:
    text = (
        "Au Kenya, la part de l’emploi agricole baisse entre les deux observations. "
        "En Ouganda, la part de l’emploi agricole suit une évolution différente. "
        "Au Ghana, la part de l’emploi agricole demeure proche de son niveau initial. "
        "Ces différences ne démontrent pas un effet causal des politiques."
    )
    assert not looks_like_degenerate_repetition(text)
    assert _guard_degenerate_language_output(text, "Rédigez en français une analyse de l’emploi agricole.") == text


def test_repeated_labels_and_units_in_an_analytical_table_are_valid() -> None:
    text = "\n".join([
        "| Pays | Indicateur | Année | Valeur |",
        "|---|---|---|---|",
        "| Kenya | Part de l’emploi agricole | 2018 | 40 % |",
        "| Kenya | Part de l’emploi agricole | 2023 | 35 % |",
        "| Ouganda | Part de l’emploi agricole | 2018 | 65 % |",
        "| Ouganda | Part de l’emploi agricole | 2023 | 60 % |",
    ])
    assert not looks_like_degenerate_repetition(text)


def test_repeated_phrase_is_detected() -> None:
    # Sprint 2 repro: same phrase repeated up to five times, no coherent answer.
    text = "mazao ya kilimo yanaongezeka " * 5
    assert looks_like_degenerate_repetition(text) is True


def test_normal_prose_is_not_flagged() -> None:
    text = (
        "Kilimo cha mahindi kinaongezeka nchini Kenya kutokana na mvua za kutosha. "
        "Wakulima wengi wamejitolea kupanda mazao zaidi msimu huu wa kilimo."
    )
    assert looks_like_degenerate_repetition(text) is False


def test_short_answer_is_not_flagged() -> None:
    # Too few words to ever reach 3 occurrences of a 4-word phrase.
    assert looks_like_degenerate_repetition("Asante sana kwa swali lako") is False


def test_empty_answer_is_not_flagged() -> None:
    assert looks_like_degenerate_repetition("") is False
    assert looks_like_degenerate_repetition(None) is False


def test_repetition_needs_at_least_four_word_phrase() -> None:
    # A single repeated short connector word must not trip the guard.
    # This has short 2-word overlaps but let's check a case with genuinely
    # varied short prose reusing common words -- should not false-positive.
    varied = "the maize yield rose this year. rice yield also rose this year in the region."
    assert looks_like_degenerate_repetition(varied) is False


def test_generation_failure_is_transient_and_localized() -> None:
    assert "Tafadhali jaribu tena" in generation_failure_answer("sw")
    assert "Veuillez réessayer" in generation_failure_answer("fr")
    assert "English" not in generation_failure_answer("fr")


def test_guard_replaces_repetitive_swahili_answer() -> None:
    repeated = "mazao ya kilimo yanaongezeka sana " * 5
    out = _guard_degenerate_language_output(repeated, "Je, mazao ya kilimo yanaongezekaje?")
    assert out != repeated
    assert "Tafadhali jaribu tena" in out


def test_guard_leaves_normal_swahili_answer_untouched() -> None:
    answer = (
        "Kilimo cha mahindi kinaongezeka nchini Kenya kutokana na mvua za kutosha "
        "na juhudi za wakulima wa ndani."
    )
    out = _guard_degenerate_language_output(answer, "Mazao ya mahindi yanaendeleaje Kenya?")
    assert out == answer


def test_guard_does_not_gate_repetitive_english_answer() -> None:
    # Sprint 2 only reported this failure for non-English languages; English
    # generation was not flagged as affected, so English repetition (however
    # unlikely) should not be silently replaced.
    repeated_en = "maize yield increased significantly " * 5
    out = _guard_degenerate_language_output(repeated_en, "How is maize yield trending?")
    assert out == repeated_en


def test_guard_leaves_empty_answer_untouched() -> None:
    assert _guard_degenerate_language_output("", "Habari za mazao?") == ""


def test_genuine_french_generation_loop_is_not_served_with_evidence_claims() -> None:
    result = _finalize_generation_result(
        "les prix alimentaires augmentent " * 8, [], query="Rédigez en français une analyse des prix alimentaires.",
    )
    assert "Veuillez réessayer" in result.answer
    assert result.citations == []
    assert result.acf.band == "no_evidence"


def test_french_report_keeps_valid_indicator_repetition_and_citations(monkeypatch):
    answer = (
        "Au Kenya, la part de l’emploi agricole est de 40 % en 2018 [1]. "
        "Au Kenya, la part de l’emploi agricole est de 35 % en 2023 [2]. "
        "En Ouganda, la part de l’emploi agricole est de 65 % en 2023 [3]. "
        "Ces observations ne permettent pas d’attribuer les changements à une politique."
    )
    items = [
        {"source": "bigquery", "content": f"Agricultural employment share in {country}, {year}: {value} %.",
         "metadata": {"status": "ok", "table_id": "fct_employment", "country_name": country,
                      "year": year, "organisation_name": "ILOSTAT", "unit": "%",
                      "value_semantics": {"measure_value": value, "unit": "%"},
                      "raw_row": {"value": value, "unit": "%", "year": year, "country_name": country}}}
        for country, year, value in [("Kenya", 2018, 40), ("Kenya", 2023, 35), ("Uganda", 2023, 65)]
    ]
    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", lambda *args, **kwargs: answer)
    result = generate(
        "Rédigez en français une analyse comparative de l’emploi agricole au Kenya et en Ouganda.",
        items, answer_lang="fr", task_mode="analytical",
        decomposition={"geography": ["Kenya", "Uganda"], "primary_measures": ["employment_share"]},
    )
    assert "la part de l’emploi agricole" in result.answer
    assert "rephrase" not in result.answer
    assert len(result.citations) == 3
