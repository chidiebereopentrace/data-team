"""Unit tests for shared API response models."""

from __future__ import annotations

from ml.rag.acf_signal import acf_signal_from_result
from ml.rag.api_schemas import ACFSignal, ArtifactItem, CitationItem, UsageStats
from ml.rag.app.api import QueryResponse
from ml.rag.chatbot.plan_policy import allows_export


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
            band="strong",
            band_label="Strong confidence",
            score=76,
            explanation="Well-supported by recent national evidence.",
            note="Well-supported by recent national evidence.",
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
    assert data["acf"]["score"] == 76


def test_acf_signal_serializes_path_b_fields() -> None:
    acf = ACFSignal(
        band="moderate",
        band_label="Moderate confidence",
        score=55,
        explanation="Partially supported.",
        note="Partially supported.",
    )
    data = acf.model_dump()
    assert data["band"] == "moderate"
    assert data["score"] == 55
    assert data["band_label"] == "Moderate confidence"
    assert data["explanation"] == "Partially supported."


def test_query_response_includes_acf() -> None:
    resp = QueryResponse(
        answer="Maize production in Nigeria...",
        session_id="xyz789",
        acf=ACFSignal(
            band="no_evidence",
            band_label="No evidence",
            score=0,
            explanation="No OpenTrace sources matched this query.",
            note="No OpenTrace sources matched this query.",
        ),
    )
    data = resp.model_dump()
    assert data["acf"]["band"] == "no_evidence"
    assert data["acf"]["score"] == 0
    assert "no opentrace" in data["acf"]["explanation"].lower()


def test_query_response_includes_artifacts() -> None:
    resp = QueryResponse(
        answer="See attached CSV.",
        session_id="sid1",
        session_found=True,
        acf=ACFSignal(
            band="moderate",
            band_label="Moderate confidence",
            score=55,
            explanation="Partial structured support.",
            note="Partial structured support.",
        ),
        artifacts=[
            ArtifactItem(
                id="a1",
                kind="csv",
                filename="maize.csv",
                mime_type="text/csv",
                url="https://example.com/maize.csv",
                summary="CSV export (10 rows)",
                citation_ids=[1],
                byte_size=120,
            )
        ],
    )
    data = resp.model_dump()
    assert data["artifacts"][0]["kind"] == "csv"
    assert data["artifacts"][0]["url"].endswith("maize.csv")
    assert data["session_found"] is True
    assert data["session_ttl_seconds"] == 86400


def test_allows_export_gates_query_plans() -> None:
    assert allows_export("Integrated") is True
    assert allows_export("Agribusinesses") is True
    assert allows_export("Farmers") is False


def test_acf_signal_from_result_path_b() -> None:
    sig = acf_signal_from_result(
        {
            "acf_band": "strong",
            "acf_band_label": "Strong confidence",
            "acf_score": 76,
            "acf_explanation": "Triangulated national evidence.",
            "acf_claim_level": "national",
            "acf_question_type": "time_sensitive",
        }
    )
    assert sig.score == 76
    assert sig.band == "strong"
    assert sig.claim_level == "national"
