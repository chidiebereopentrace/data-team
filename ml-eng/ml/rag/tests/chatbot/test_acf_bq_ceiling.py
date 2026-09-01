"""ACF ceiling when warehouse execute state is weak."""
from __future__ import annotations

from ml.rag.chatbot.acf_scoring import ACFResult, apply_bq_execute_ceiling, no_evidence_acf, score_cited_evidence


def test_apply_bq_execute_ceiling_partial_panel() -> None:
    base = ACFResult(
        band="moderate",
        band_label="Moderate confidence",
        score=65,
        explanation="test",
        note="test",
        components={"coverage": 0.9},
    )
    capped = apply_bq_execute_ceiling(
        base,
        {"structured_bq_timed_out": True},
        usable_bq=True,
        bq_sql_debug=[
            {"sql": "SELECT 1", "status": "ok", "job_id": "j1"},
            {"sql": "SELECT 2", "status": "timeout", "job_id": "j2"},
        ],
    )
    assert capped.applied_ceiling == "partial_panel"
    assert capped.score > 35


def test_apply_bq_execute_ceiling_timeout() -> None:
    base = ACFResult(
        band="moderate",
        band_label="Moderate confidence",
        score=65,
        explanation="test",
        note="test",
    )
    capped = apply_bq_execute_ceiling(
        base,
        {"structured_bq_timed_out": True},
        usable_bq=False,
    )
    assert capped.score <= 35
    assert capped.band == "low"


def test_apply_bq_execute_ceiling_never_executed() -> None:
    base = ACFResult(
        band="strong",
        band_label="Strong",
        score=80,
        explanation="test",
        note="test",
    )
    capped = apply_bq_execute_ceiling(base, {"structured_bq_never_executed": True})
    assert capped.score == 0
    assert capped.band == "no_evidence"


def test_no_evidence_when_no_citations() -> None:
    result = score_cited_evidence([], query="maize production Ghana")
    assert result.band == "no_evidence"
