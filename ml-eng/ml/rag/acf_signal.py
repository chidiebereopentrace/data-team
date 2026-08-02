"""Build API ACFSignal from graph / ChatTurnResult state."""
from __future__ import annotations

from typing import Any

from ml.rag.api_schemas import ACFBandLiteral, ACFSignal

_VALID_BANDS = frozenset(
    {"very_strong", "strong", "moderate", "limited", "low", "no_evidence"}
)


def acf_signal_from_result(result: dict[str, Any] | None) -> ACFSignal:
    """Map run_rag / ChatTurnResult.raw_result fields onto ACFSignal."""
    r = result or {}
    band_raw = str(r.get("acf_band") or "no_evidence").strip().lower()
    if band_raw not in _VALID_BANDS:
        # Legacy stub bands
        legacy = {"high": "strong", "medium": "moderate", "low": "limited"}
        band_raw = legacy.get(band_raw, "no_evidence")
    band: ACFBandLiteral = band_raw  # type: ignore[assignment]

    score_raw = r.get("acf_score")
    try:
        score_f = float(score_raw) if score_raw is not None else 0.0
    except (TypeError, ValueError):
        score_f = 0.0
    # Legacy stub used 0.0–1.0; Path B uses 0–100.
    if 0.0 <= score_f <= 1.0 and str(r.get("acf_band") or "").lower() in {
        "high",
        "medium",
        "low",
        "no_evidence",
    }:
        score = int(round(score_f * 100))
    else:
        score = int(round(score_f))
    score = max(0, min(100, score))

    explanation = str(
        r.get("acf_explanation") or r.get("acf_note") or "No confidence signal available."
    )
    band_label = str(
        r.get("acf_band_label")
        or {
            "very_strong": "Very strong confidence",
            "strong": "Strong confidence",
            "moderate": "Moderate confidence",
            "limited": "Limited confidence",
            "low": "Low confidence",
            "no_evidence": "No evidence",
        }.get(band, band.replace("_", " ").title())
    )
    components = r.get("acf_components")
    if components is not None and not isinstance(components, dict):
        components = None

    return ACFSignal(
        band=band,
        band_label=band_label,
        score=score,
        explanation=explanation,
        note=explanation,
        components=components,
        applied_ceiling=r.get("acf_applied_ceiling"),
        config_version=r.get("acf_config_version"),
        claim_level=r.get("acf_claim_level"),
        question_type=r.get("acf_question_type"),
    )
