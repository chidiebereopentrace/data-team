"""Unit tests for shared API response models."""

from __future__ import annotations

from ml.rag.api_schemas import CitationItem, UsageStats
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
