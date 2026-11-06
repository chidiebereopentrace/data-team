"""A missing unsourced_figures name must not fail the answer path."""
from __future__ import annotations

import pytest

import ml.rag.chatbot.generator as gen


def test_unsourced_grounding_nameerror_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(gen, "unsourced_figures")
    assert gen._safe_unsourced_figures("Maize was 4200.", []) == []


def test_safe_unsourced_figures_flags_uncited_number() -> None:
    cited = [
        {
            "text": "Maize production in the region has been rising steadily.",
            "title": "Source",
            "doc_kind": "academic_article",
        }
    ]
    assert gen._safe_unsourced_figures(
        "Maize production reached 4200 tonnes last season.",
        cited,
    ) == ["4200"]
