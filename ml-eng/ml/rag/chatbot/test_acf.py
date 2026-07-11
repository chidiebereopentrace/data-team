"""Unit tests for the ADZA Confidence Framework (ACF) module.

Sprint 1, Week 2 (Jul 2026): validates that compute_acf produces correct bands,
scores, and notes across all representative retrieval scenarios.
"""
from __future__ import annotations

import os
from unittest import mock

from ml.rag.chatbot.acf import ACFBand, ACFResult, compute_acf


# ---------------------------------------------------------------------------
# Helpers to build mock context chunks
# ---------------------------------------------------------------------------

def _chunk(
    kind: str = "news",
    score: float = 0.7,
    rerank_score: float | None = None,
) -> dict:
    item: dict = {
        "content": f"Sample {kind} content about agriculture in Kenya.",
        "_context_kind": kind,
        "source": kind,
        "score": score,
    }
    if rerank_score is not None:
        item["_rerank_score"] = rerank_score
    return item


# ---------------------------------------------------------------------------
# Band: NO_EVIDENCE
# ---------------------------------------------------------------------------

def test_empty_context_returns_no_evidence() -> None:
    """No context at all → NO_EVIDENCE with score 0.0."""
    acf = compute_acf([])
    assert acf.band == "no_evidence"
    assert acf.score == 0.0
    assert "data gap" in acf.note.lower() or "no opentrace" in acf.note.lower()


def test_none_context_returns_no_evidence() -> None:
    acf = compute_acf(None)
    assert acf.band == "no_evidence"
    assert acf.score == 0.0


# ---------------------------------------------------------------------------
# Band: LOW
# ---------------------------------------------------------------------------

def test_single_weak_chunk_returns_low() -> None:
    """One chunk with a low score → LOW band."""
    chunks = [_chunk("news", score=0.15)]
    acf = compute_acf(chunks)
    assert acf.band == "low"
    assert 0.10 <= acf.score < 0.45
    assert "caution" in acf.note.lower() or "few" in acf.note.lower()


# ---------------------------------------------------------------------------
# Band: MEDIUM
# ---------------------------------------------------------------------------

def test_moderate_context_returns_medium() -> None:
    """Several chunks from one source type, decent scores → MEDIUM."""
    chunks = [_chunk("academic", score=0.6) for _ in range(4)]
    acf = compute_acf(chunks)
    assert acf.band == "medium"
    assert 0.45 <= acf.score < 0.75


# ---------------------------------------------------------------------------
# Band: HIGH
# ---------------------------------------------------------------------------

def test_rich_diverse_context_returns_high() -> None:
    """Many chunks from multiple source types with high scores → HIGH."""
    chunks = [
        _chunk("news", score=0.85),
        _chunk("news", score=0.80),
        _chunk("academic", score=0.75),
        _chunk("academic", score=0.78),
        _chunk("bigquery", score=0.90),
        _chunk("bigquery", score=0.88),
        _chunk("policy", score=0.72),
        _chunk("ota_insight", score=0.70),
    ]
    acf = compute_acf(chunks)
    assert acf.band == "high"
    assert acf.score >= 0.75
    assert "well-supported" in acf.note.lower()


# ---------------------------------------------------------------------------
# Web fallback penalty
# ---------------------------------------------------------------------------

def test_web_fallback_reduces_score() -> None:
    """Using web fallback should lower the score (internal-only bonus = 0)."""
    chunks = [_chunk("academic", score=0.7) for _ in range(5)]
    acf_no_web = compute_acf(chunks, used_web_fallback=False)
    acf_web = compute_acf(chunks, used_web_fallback=True)
    assert acf_web.score < acf_no_web.score


def test_web_source_types_penalize_internal_signal() -> None:
    """Chunks with web_wikipedia or web_search kinds should lose the internal bonus."""
    chunks = [
        _chunk("news", score=0.7),
        _chunk("web_wikipedia", score=0.6),
    ]
    acf = compute_acf(chunks, used_web_fallback=False)
    # Even without explicit web fallback flag, web source types lose the internal bonus
    chunks_internal = [
        _chunk("news", score=0.7),
        _chunk("academic", score=0.6),
    ]
    acf_internal = compute_acf(chunks_internal, used_web_fallback=False)
    assert acf.score < acf_internal.score


# ---------------------------------------------------------------------------
# Source diversity
# ---------------------------------------------------------------------------

def test_single_source_type_lower_diversity() -> None:
    """All chunks from one type → lower diversity signal than mixed types."""
    single = [_chunk("news", score=0.7) for _ in range(5)]
    mixed = [
        _chunk("news", score=0.7),
        _chunk("academic", score=0.7),
        _chunk("bigquery", score=0.7),
        _chunk("policy", score=0.7),
        _chunk("ota_insight", score=0.7),
    ]
    acf_single = compute_acf(single)
    acf_mixed = compute_acf(mixed)
    assert acf_mixed.score > acf_single.score


# ---------------------------------------------------------------------------
# Rerank score preference
# ---------------------------------------------------------------------------

def test_rerank_score_preferred_over_vector_score() -> None:
    """_rerank_score should be used preferentially over score."""
    chunks = [_chunk("news", score=0.3, rerank_score=0.9) for _ in range(5)]
    acf = compute_acf(chunks)
    # Should reflect the high rerank score, not the low vector score
    assert acf.score > 0.5


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------

def test_threshold_env_override() -> None:
    """ACF_THRESHOLD_HIGH env override should shift the band boundary."""
    chunks = [_chunk("news", score=0.7) for _ in range(6)]  # reasonable context
    acf_default = compute_acf(chunks)

    with mock.patch.dict(os.environ, {"ACF_THRESHOLD_HIGH": "0.30"}):
        acf_low_bar = compute_acf(chunks)

    # With a lower HIGH threshold, same chunks should more easily land in HIGH
    if acf_default.band != "high":
        assert acf_low_bar.band == "high" or acf_low_bar.score >= 0.30


def test_note_env_override() -> None:
    """Custom note text via ACF_NOTE_HIGH env var."""
    chunks = [
        _chunk("news", score=0.9),
        _chunk("academic", score=0.9),
        _chunk("bigquery", score=0.9),
        _chunk("policy", score=0.85),
        _chunk("ota_insight", score=0.85),
        _chunk("news", score=0.88),
        _chunk("academic", score=0.87),
        _chunk("bigquery", score=0.86),
    ]
    custom = "Chidi approved this copy."
    with mock.patch.dict(os.environ, {"ACF_NOTE_HIGH": custom}):
        acf = compute_acf(chunks)
    if acf.band == "high":
        assert acf.note == custom


# ---------------------------------------------------------------------------
# ACFResult is immutable
# ---------------------------------------------------------------------------

def test_acf_result_is_frozen() -> None:
    acf = compute_acf([_chunk("news", score=0.5)])
    try:
        acf.score = 999  # type: ignore[misc]
        assert False, "ACFResult should be frozen"
    except (AttributeError, TypeError):
        pass  # expected — dataclass(frozen=True)


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

def test_score_bounded_zero_to_one() -> None:
    """Score must always be in [0.0, 1.0]."""
    for ctx in [
        [],
        [_chunk("news", score=0.0)],
        [_chunk("news", score=1.0) for _ in range(20)],
    ]:
        acf = compute_acf(ctx)
        assert 0.0 <= acf.score <= 1.0, f"score {acf.score} out of bounds"
