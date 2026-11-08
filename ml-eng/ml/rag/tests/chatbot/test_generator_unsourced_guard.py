"""Unsupported figures are detected from cited evidence and removed from the answer."""
from __future__ import annotations

from ml.rag.chatbot.numeric_grounding import strip_unsourced_figures, unsourced_figures


def test_unsourced_figures_flags_uncited_number() -> None:
    cited = [
        {
            "text": "Maize production in the region has been rising steadily.",
            "title": "Source",
            "doc_kind": "academic_article",
        }
    ]
    assert unsourced_figures(
        "Maize production reached 4200 tonnes last season.",
        cited,
    ) == ["4200"]


def test_stripped_answer_does_not_keep_the_uncited_figure() -> None:
    prose = "Maize production reached 4200 tonnes last season. The season was late."
    flagged = unsourced_figures(prose, [{"text": "The season was late."}])
    out = strip_unsourced_figures(prose, flagged)
    assert "4200" not in out
    assert "late" in out
