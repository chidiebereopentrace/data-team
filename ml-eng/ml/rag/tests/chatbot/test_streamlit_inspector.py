"""Unit tests for streamlit_inspector route inference (no Streamlit UI)."""
from __future__ import annotations

from ml.rag.chatbot.streamlit_inspector import (
    infer_pipeline_route,
    normalize_query_response,
)


def test_infer_pipeline_route_meta() -> None:
    assert infer_pipeline_route({"is_meta_query": True}) == "meta"


def test_infer_pipeline_route_product() -> None:
    assert infer_pipeline_route({"is_product_query": True}) == "product"


def test_infer_pipeline_route_help() -> None:
    assert infer_pipeline_route({"is_help_query": True, "is_product_query": True}) == "help"


def test_infer_pipeline_route_greeting() -> None:
    assert infer_pipeline_route({"is_greeting_query": True}) == "greeting"


def test_infer_pipeline_route_out_of_scope() -> None:
    assert infer_pipeline_route({"is_out_of_scope_query": True}) == "out_of_scope"


def test_infer_pipeline_route_language_unknown() -> None:
    assert infer_pipeline_route({"is_language_unknown": True}) == "language_unknown"


def test_infer_pipeline_route_insufficient() -> None:
    assert infer_pipeline_route({"insufficient_context": True}) == "insufficient"


def test_infer_pipeline_route_web_fallback() -> None:
    assert infer_pipeline_route({"web_results": [{"content": "wiki"}]}) == "full_rag + web_fallback"


def test_infer_pipeline_route_full_rag() -> None:
    assert infer_pipeline_route({"vector_news_results": [{"content": "x"}]}) == "full_rag"


def test_infer_pipeline_route_meta_wins_over_product() -> None:
    assert infer_pipeline_route({"is_meta_query": True, "is_product_query": True}) == "meta"


def test_normalize_query_response_maps_sql_trace() -> None:
    payload = {
        "answer": "Kenya maize data.",
        "session_id": "sess-1",
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        "acf": {
            "band": "moderate",
            "band_label": "Moderate confidence",
            "score": 55,
            "explanation": "Structured data cited.",
        },
        "langfuse_trace_id": "trace-abc",
        "trace": {
            "decomposition": {"geography": ["Kenya"]},
            "vector_news_count": 4,
            "bq_sql_queries": ["SELECT 1"],
            "bq_sql_debug": [{"sql": "SELECT 1", "status": "ok", "sql_source": "nl2sql"}],
            "bq_sql_plan": {
                "selected_tables": ["fct_production"],
                "query_intents": [{"goal": "yields", "subquestion_id": "sq1"}],
                "slot_path": True,
            },
            "sql_source": "nl2sql",
            "bq_cache_hit": False,
            "bq_nl2sql_ms": 99.0,
            "bq_execute_ms": 12.0,
        },
    }
    kwargs = {
        "plan_type": "Government",
        "user_profile": {"plan_type": "Government", "category": "Government"},
    }
    out = normalize_query_response(payload, latency_ms=50.0, query="q", kwargs=kwargs)

    assert out["answer"] == "Kenya maize data."
    assert out["_backend_mode"] == "http_api"
    assert out["bq_sql_queries"] == ["SELECT 1"]
    assert out["bq_sql_debug"][0]["sql_source"] == "nl2sql"
    assert out["bq_sql_plan"]["selected_tables"] == ["fct_production"]
    assert out["sql_source"] == "nl2sql"
    assert out["bq_nl2sql_ms"] == 99.0
    assert out["decomposition"]["geography"] == ["Kenya"]
    assert out["_http_trace"]["vector_news_count"] == 4
