"""ACF must not stay Strong when figures are untraceable (ML-058)."""

from __future__ import annotations

from ml.rag.chatbot.acf_scoring import ACFResult
from ml.rag.chatbot.generator import _cap_acf_for_unsourced_figures


def _strong() -> ACFResult:
    return ACFResult(
        band="strong",
        band_label="Strong confidence",
        score=85,
        explanation="Multiple corroborating sources.",
        note="Multiple corroborating sources.",
    )


def test_strong_is_capped_when_figures_unsourced() -> None:
    out = _cap_acf_for_unsourced_figures(_strong(), ["4200"])
    assert out.band == "limited"
    assert out.score <= 45
    assert out.applied_ceiling == "unsourced_figures"


def test_explanation_records_that_figures_were_removed() -> None:
    out = _cap_acf_for_unsourced_figures(_strong(), ["4200"])
    assert "removed" in out.explanation
    assert "4200" not in out.explanation


def test_original_explanation_is_preserved() -> None:
    out = _cap_acf_for_unsourced_figures(_strong(), ["4200"])
    assert "Multiple corroborating sources." in out.explanation


def test_no_cap_when_nothing_unsourced() -> None:
    original = _strong()
    assert _cap_acf_for_unsourced_figures(original, []) is original


def test_already_low_band_is_left_alone() -> None:
    low = ACFResult(
        band="low",
        band_label="Low confidence",
        score=20,
        explanation="Thin evidence.",
        note="Thin evidence.",
    )
    assert _cap_acf_for_unsourced_figures(low, ["4200"]) is low


def test_many_figures_are_not_copied_into_the_explanation() -> None:
    out = _cap_acf_for_unsourced_figures(
        _strong(), ["11", "22", "33", "44", "55"]
    )
    assert "55" not in out.explanation
    assert "removed" in out.explanation
