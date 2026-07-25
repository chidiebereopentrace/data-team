"""Unit tests for shared API response models."""

from __future__ import annotations

from ml.rag.api_schemas import ACFSignal, CitationItem, UsageStats
from ml.rag.app.api import QueryResponse


def test_usage_stats_from_dict_aliases() -> None:
    stats = UsageStats.from_usage_dict(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    assert stats.input_tokens == 10
    assert stats.output_tokens == 5
    assert stats.total_tokens == 15


def test_usage_stats_serializes_three_fields_only() -> None:
    stats = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150)
    data = stats.model_dump()
    assert set(data.keys()) == {"input_tokens", "output_tokens", "total_tokens"}


def test_query_response_serializes_citations_and_usage() -> None:
    resp = QueryResponse(
        answer="Rice yields rose [1].",
        session_id="abc123",
        citations=[
            CitationItem(id=1, kind="news", text="[News] Senegal policy", url="https://example.com"),
        ],
        acf=ACFSignal(
            band="high",
            score=0.85,
            note="This response is well-supported by multiple OpenTrace sources.",
        ),
        usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
    )
    data = resp.model_dump()
    assert data["answer"] == "Rice yields rose [1]."
    assert data["citations"][0]["kind"] == "news"
    assert data["usage"] == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }


def test_acf_signal_serializes_all_fields() -> None:
    """Sprint 1 Wk2: ACF signal must appear in every QueryResponse."""
    acf = ACFSignal(band="medium", score=0.55, note="Partially supported.")
    data = acf.model_dump()
    assert data == {"band": "medium", "score": 0.55, "note": "Partially supported."}


def test_query_response_includes_acf() -> None:
    """QueryResponse.acf must be present and correctly serialized."""
    resp = QueryResponse(
        answer="Maize production in Nigeria...",
        session_id="xyz789",
        acf=ACFSignal(
            band="no_evidence",
            score=0.0,
            note="No OpenTrace sources matched this query.",
        ),
    )
    data = resp.model_dump()
    assert data["acf"]["band"] == "no_evidence"
    assert data["acf"]["score"] == 0.0
    assert "no opentrace" in data["acf"]["note"].lower()
