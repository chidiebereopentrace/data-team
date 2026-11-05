"""ACF source-basis band ceiling (§7)."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.acf_scoring import ACFResult, cap_acf_band
from ml.rag.chatbot.answer_outcome import decide_outcome
from ml.rag.chatbot.ontology_routing import acf_band_ceiling


def _strong() -> ACFResult:
    return ACFResult(
        band="strong",
        band_label="Strong confidence",
        score=80,
        explanation="base",
        note="base",
    )


def test_cap_acf_band_clamps_to_moderate() -> None:
    capped = cap_acf_band(_strong(), "moderate")
    assert capped.band == "moderate"
    assert capped.score <= 69
    assert capped.applied_ceiling == "source_basis:moderate"
    assert "documents" in capped.explanation.lower()


def test_cap_acf_band_noop_when_already_weaker() -> None:
    base = ACFResult(
        band="limited",
        band_label="Limited confidence",
        score=45,
        explanation="base",
        note="base",
    )
    assert cap_acf_band(base, "moderate") is base


def test_acf_band_ceiling_documents_only() -> None:
    assert acf_band_ceiling(None, {"qdrant", "news"}) == "moderate"
    assert acf_band_ceiling(None, {"web"}) == "limited"
    assert acf_band_ceiling(None, {"structured_data"}) is None


def test_empty_policy_keeps_vector_hits_as_evidence() -> None:
    """qdrant_only / documents_only must not look like no-data when vector hits exist."""
    items = [
        {
            "content": "Mushroom cultivation notes for Ghana smallholders.",
            "_context_kind": "news",
            "metadata": {"context_kind": "news"},
        }
    ]
    assert decide_outcome(items).kind == "documents_only"


@pytest.mark.parametrize("flag", ["0", "1"])
def test_cap_flag_default_documented(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADZA_ACF_SOURCE_BASIS_CAP", flag)
    # Smoke: helper remains callable regardless of flag (flag is read in generator).
    assert cap_acf_band(_strong(), "moderate").band == "moderate"
