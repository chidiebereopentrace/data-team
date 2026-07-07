"""Unit tests for streamlit_inspector route inference (no Streamlit UI)."""
from __future__ import annotations

from ml.rag.chatbot.streamlit_inspector import infer_pipeline_route, normalize_http_response


def test_infer_pipeline_route_meta() -> None:
    assert infer_pipeline_route({"is_meta_query": True}) == "meta"


def test_infer_pipeline_route_product() -> None:
    assert infer_pipeline_route({"is_product_query": True}) == "product"


def test_infer_pipeline_route_insufficient() -> None:
    assert infer_pipeline_route({"insufficient_context": True}) == "insufficient"


def test_infer_pipeline_route_web_fallback() -> None:
    assert infer_pipeline_route({"web_results": [{"content": "wiki"}]}) == "full_rag + web_fallback"


def test_infer_pipeline_route_full_rag() -> None:
    assert infer_pipeline_route({"vector_news_results": [{"content": "x"}]}) == "full_rag"


def test_infer_pipeline_route_meta_wins_over_product() -> None:
    assert infer_pipeline_route({"is_meta_query": True, "is_product_query": True}) == "meta"


def test_normalize_http_response_counts() -> None:
    payload = {
        "answer": "hello",
        "session_id": "abc123",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "trace": {
            "decomposition": {"geography": ["Kenya"]},
            "bq_table_candidates_count": 2,
            "vector_news_count": 3,
            "vector_academic_count": 1,
            "merged_context_count": 4,
            "reranked_context_count": 2,
        },
    }
    out = normalize_http_response(payload, latency_ms=100.0, query="q", kwargs={})
    assert out["answer"] == "hello"
    assert out["session_id"] == "abc123"
    assert len(out["vector_news_results"]) == 3
    assert out["usage"]["total_tokens"] == 15
    assert out["_backend_mode"] == "http_api"
