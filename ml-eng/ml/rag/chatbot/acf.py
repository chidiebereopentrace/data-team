"""
ADZA Confidence Framework (ACF) — computes a confidence signal for every RAG response.

Sprint 1, Week 2 (Jul 2026): Every response now surfaces ``{ band, score, note }``
by default so testers and users can immediately see how well-supported an answer is.

The score is derived from retrieval quality signals — **not** from the LLM output —
so it reflects evidence strength, not generation fluency.

Scoring dimensions (weights sum to 1.0):

    chunk_count_signal   × 0.30   more usable chunks → more evidence
    avg_relevance_score  × 0.40   higher rerank / similarity → better match
    source_diversity     × 0.20   evidence from multiple source types is stronger
    internal_only_bonus  × 0.10   no web fallback needed → internal corpus was sufficient

Band thresholds and plain-language copy are intentionally configurable via env vars
so Chidi can adjust them from the PR review without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ACFBand(str, Enum):
    """Confidence bands — ordered from strongest to weakest."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_EVIDENCE = "no_evidence"


# ---------------------------------------------------------------------------
# Default thresholds (overridable via env for reviewer convenience)
# ---------------------------------------------------------------------------

def _threshold(env: str, default: float) -> float:
    try:
        return float(os.environ.get(env, str(default)) or default)
    except (ValueError, TypeError):
        return default


def _high_threshold() -> float:
    return _threshold("ACF_THRESHOLD_HIGH", 0.75)


def _medium_threshold() -> float:
    return _threshold("ACF_THRESHOLD_MEDIUM", 0.45)


def _low_threshold() -> float:
    return _threshold("ACF_THRESHOLD_LOW", 0.10)


# ---------------------------------------------------------------------------
# Plain-language notes (overridable via env)
# ---------------------------------------------------------------------------

_DEFAULT_NOTES: dict[ACFBand, str] = {
    ACFBand.HIGH: (
        "This response is well-supported by multiple OpenTrace sources."
    ),
    ACFBand.MEDIUM: (
        "This response is supported by limited OpenTrace sources. "
        "Some details may be incomplete."
    ),
    ACFBand.LOW: (
        "Few relevant sources were found. Treat this response with caution."
    ),
    ACFBand.NO_EVIDENCE: (
        "No OpenTrace sources matched this query. "
        "This is a data gap, not a low-confidence answer."
    ),
}

_NOTE_ENV_KEYS: dict[ACFBand, str] = {
    ACFBand.HIGH: "ACF_NOTE_HIGH",
    ACFBand.MEDIUM: "ACF_NOTE_MEDIUM",
    ACFBand.LOW: "ACF_NOTE_LOW",
    ACFBand.NO_EVIDENCE: "ACF_NOTE_NO_EVIDENCE",
}


def _note_for_band(band: ACFBand) -> str:
    env_key = _NOTE_ENV_KEYS.get(band, "")
    custom = os.environ.get(env_key, "").strip() if env_key else ""
    return custom or _DEFAULT_NOTES[band]


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

_W_CHUNK_COUNT = 0.30
_W_RELEVANCE = 0.40
_W_DIVERSITY = 0.20
_W_INTERNAL = 0.10

# Chunk count scaling: this many usable chunks gets the full 1.0 on the chunk dimension.
_CHUNK_COUNT_FULL = 8


# ---------------------------------------------------------------------------
# Known source types for diversity calculation
# ---------------------------------------------------------------------------

_KNOWN_SOURCE_TYPES = frozenset({
    "bigquery",
    "news",
    "academic",
    "policy",
    "public_report",
    "ota_insight",
    "ota_metric",
    "research",
})

_WEB_SOURCE_TYPES = frozenset({
    "web_wikipedia",
    "web_search",
})


@dataclass(frozen=True)
class ACFResult:
    """Immutable ACF result returned alongside every generation."""

    band: str          # ACFBand value string (e.g. "high", "medium")
    score: float       # 0.0 – 1.0
    note: str          # plain-language explanation


def _extract_score(item: dict[str, Any]) -> float:
    """Best available relevance score from a context chunk."""
    for key in ("_rerank_score", "_llm_score", "score"):
        val = item.get(key)
        if val is not None:
            try:
                f = float(val)
                if f >= 0:
                    return min(f, 1.0)
            except (ValueError, TypeError):
                continue
    return 0.0


def _extract_source_type(item: dict[str, Any]) -> str:
    """Normalized source type from a context chunk."""
    raw = str(
        item.get("_context_kind")
        or item.get("source")
        or ""
    ).strip().lower()
    return raw


def compute_acf(
    context_items: list[dict[str, Any]] | None = None,
    *,
    used_web_fallback: bool = False,
) -> ACFResult:
    """
    Compute the ADZA Confidence Framework signal from retrieval context.

    Parameters
    ----------
    context_items:
        The usable context chunks that will be (or were) sent to the generator.
        Each chunk may carry ``_rerank_score``, ``_llm_score``, or ``score``
        from the retrieval/rerank pipeline.
    used_web_fallback:
        True if the pipeline had to invoke Tavily / Wikipedia because internal
        retrieval was insufficient.

    Returns
    -------
    ACFResult with band, score (0.0–1.0), and plain-language note.
    """
    items = context_items or []

    # ------------------------------------------------------------------
    # 1. No context at all → immediate NO_EVIDENCE
    # ------------------------------------------------------------------
    if not items:
        return ACFResult(
            band=ACFBand.NO_EVIDENCE.value,
            score=0.0,
            note=_note_for_band(ACFBand.NO_EVIDENCE),
        )

    # ------------------------------------------------------------------
    # 2. Chunk-count signal (0.0 – 1.0)
    # ------------------------------------------------------------------
    n = len(items)
    chunk_signal = min(1.0, n / _CHUNK_COUNT_FULL)

    # ------------------------------------------------------------------
    # 3. Average relevance signal (0.0 – 1.0)
    # ------------------------------------------------------------------
    scores = [_extract_score(it) for it in items]
    positive_scores = [s for s in scores if s > 0]
    if positive_scores:
        relevance_signal = sum(positive_scores) / len(positive_scores)
    else:
        # No scores available (reranker off, no vector scores) — neutral 0.5
        relevance_signal = 0.5

    # ------------------------------------------------------------------
    # 4. Source diversity signal (0.0 – 1.0)
    # ------------------------------------------------------------------
    source_types: set[str] = set()
    has_web = False
    for it in items:
        st = _extract_source_type(it)
        if st in _KNOWN_SOURCE_TYPES:
            source_types.add(st)
        elif st in _WEB_SOURCE_TYPES:
            has_web = True

    # Diversity: 1 type → 0.3, 2 types → 0.6, 3+ types → 1.0
    n_types = len(source_types)
    if n_types >= 3:
        diversity_signal = 1.0
    elif n_types == 2:
        diversity_signal = 0.6
    elif n_types == 1:
        diversity_signal = 0.3
    else:
        # Only web sources or unrecognized types
        diversity_signal = 0.1

    # ------------------------------------------------------------------
    # 5. Internal-only bonus (0.0 or 1.0)
    # ------------------------------------------------------------------
    internal_signal = 0.0 if (used_web_fallback or has_web) else 1.0

    # ------------------------------------------------------------------
    # 6. Weighted composite score
    # ------------------------------------------------------------------
    raw_score = (
        _W_CHUNK_COUNT * chunk_signal
        + _W_RELEVANCE * relevance_signal
        + _W_DIVERSITY * diversity_signal
        + _W_INTERNAL * internal_signal
    )
    score = round(min(1.0, max(0.0, raw_score)), 2)

    # ------------------------------------------------------------------
    # 7. Map to band
    # ------------------------------------------------------------------
    if score >= _high_threshold():
        band = ACFBand.HIGH
    elif score >= _medium_threshold():
        band = ACFBand.MEDIUM
    elif score >= _low_threshold():
        band = ACFBand.LOW
    else:
        band = ACFBand.NO_EVIDENCE

    return ACFResult(
        band=band.value,
        score=score,
        note=_note_for_band(band),
    )
