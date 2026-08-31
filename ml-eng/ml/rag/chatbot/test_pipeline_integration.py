"""End-to-end RAG pipeline integration tests.

Sprint 1, Week 3 (Jul 2026): exercises the real compiled LangGraph via
``run_rag()`` while mocking only the external boundaries (vector/BQ retrievers,
the LLM call, reranker model, and web fallback) so the test is deterministic and
runs anywhere — no Qdrant, BigQuery, or LLM backend required.

What this locks in (query → response), across the Weeks 1–3 work:
  1. Data-query path returns answer + citations + an ACF signal.
  2. Direct-answer-first: a preamble-opening LLM answer is cleaned end-to-end.
  3. Inline [N] citations resolve to real structured citation objects.
  4. No-data path returns the structured gap message (no fabrication).
  5. Meta/product path short-circuits with ACF HIGH.
  6. session_id threads through cleanly (Week-2 session isolation regression guard).
  7. `usage` accounting key is present on the result.
"""
from __future__ import annotations

from unittest import mock

import pytest

# The compiled graph needs langgraph at import time; skip gracefully if absent
# (e.g. a minimal dev venv). Runs fully in CI / the provisioned environment.
pytest.importorskip("langgraph")

from ml.rag.chatbot import graph as graph_mod
from ml.rag.chatbot.graph import run_rag


# ---------------------------------------------------------------------------
# Canned retrieval fixtures
# ---------------------------------------------------------------------------

def _news_chunk() -> dict:
    return {
        "content": "Senegal raised its rice self-sufficiency target for 2024.",
        "score": 0.91,
        "metadata": {
            "doc_kind": "news_article",
            "title": "Senegal rice policy shift",
            "publisher": "AgriNews",
            "published_at": "2024-06-01",
            "geo_country_primary": "Senegal",
            "url": "https://example.com/senegal-rice",
        },
    }


def _academic_chunk() -> dict:
    return {
        "content": "Field trials show drought-tolerant rice varieties raise yields.",
        "score": 0.87,
        "metadata": {
            "doc_kind": "academic_article",
            "article_title": "Drought-tolerant rice in West Africa",
            "authors": "Diallo, A.",
            "publication_year": "2024",
            "geo_country_primary": "Senegal",
            "doi": "10.1234/dt-rice",
        },
    }


def _ota_chunk() -> dict:
    return {
        "content": "Rice output projected to rise 8% next season.",
        "score": 0.80,
        "metadata": {
            "doc_kind": "ota_insight",
            "metric_text": "Rice output +8%",
            "geo_country_primary": "Senegal",
            "as_of_date": "2024-01-01",
            "ota_record_id": "ota-senegal-rice-1",
        },
    }


def _ghana_rice_bq_chunk() -> dict:
    return {
        "content": "Ghana produced 973,000 metric tons of rice in 2020.",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "source_id": "stg_faostat_production:country_name=Ghana:year=2020",
            "country_name": "Ghana",
            "product_name": "Rice",
            "element": "Production",
            "year": 2020,
            "value": 973000,
            "unit": "t",
            "geo_country_primary": "Ghana",
            "value_semantics": {
                "measure_value": 973000,
                "measure_column": "production_t",
                "element": "production",
                "metric": "production",
            },
        },
    }


def _public_report_chunk() -> dict:
    return {
        "content": "Kenya IPC Phase 2 areas expanded in the arid north.",
        "score": 0.88,
        "metadata": {
            "doc_kind": "public_report",
            "title": "Kenya IPC update",
            "published_at": "2024-02-01",
            "geo_country_primary": "Kenya",
        },
    }


def _maize_bq_chunk(*, country: str, year: int, value: int = 500000) -> dict:
    return {
        "content": f"{country} produced {value:,} metric tons of maize in {year}.",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "source_id": f"stg_faostat_production:country_name={country}:year={year}",
            "country_name": country,
            "product_name": "Maize",
            "element": "Production",
            "year": year,
            "value": value,
            "unit": "t",
            "geo_country_primary": country,
            "value_semantics": {
                "measure_value": value,
                "measure_column": "production_t",
                "element": "production",
                "metric": "production",
            },
        },
    }


def _install_pipeline_mocks(
    stack,
    *,
    news=None,
    academic=None,
    policies=None,
    public_reports=None,
    formation=None,
    ota=None,
    bq_results=None,
    llm_answer="ok",
):
    """Patch every external boundary of the graph for deterministic runs."""
    if bq_results:
        stack.enter_context(
            mock.patch.object(
                graph_mod,
                "reason_bq_sql_plan",
                return_value={
                    "skip_bq": False,
                    "selected_tables": ["stg_faostat_production"],
                    "table_hints": ["Table: staging_dev.stg_faostat_production"],
                    "query_intents": [],
                },
            )
        )
        stack.enter_context(
            mock.patch(
                "ml.rag.chatbot.graph.BQRetriever.retrieve",
                return_value=list(bq_results),
            )
        )
    else:
        stack.enter_context(
            mock.patch.object(
                graph_mod,
                "reason_bq_sql_plan",
                return_value={"skip_bq": True, "selected_tables": [], "query_intents": []},
            )
        )
        stack.enter_context(
            mock.patch(
                "ml.rag.chatbot.graph.BQRetriever.retrieve",
                return_value=[],
            )
        )
    stack.enter_context(
        mock.patch.object(graph_mod, "_retrieve_news", return_value=news or [])
    )
    stack.enter_context(
        mock.patch.object(graph_mod, "_retrieve_academic_papers", return_value=academic or [])
    )
    stack.enter_context(
        mock.patch.object(graph_mod, "_retrieve_policies", return_value=policies or [])
    )
    stack.enter_context(
        mock.patch.object(
            graph_mod, "_retrieve_public_reports", return_value=public_reports or []
        )
    )
    stack.enter_context(
        mock.patch.object(graph_mod, "_retrieve_formation", return_value=formation or [])
    )
    stack.enter_context(
        mock.patch.object(graph_mod, "_retrieve_ota", return_value=ota or [])
    )
    # Deterministic reranker: identity passthrough truncated to top_k.
    stack.enter_context(
        mock.patch.object(
            graph_mod,
            "rerank",
            side_effect=lambda q, items, top_k=20, **kwargs: list(items)[:top_k],
        )
    )
    # Keep web fallback out of the way (internal context decides the path).
    stack.enter_context(
        mock.patch.object(graph_mod, "needs_web_fallback", return_value=False)
    )
    # Mock the LLM call inside the generator.
    stack.enter_context(
        mock.patch("ml.rag.chatbot.generator._call_llama", return_value=llm_answer)
    )
    import ml.rag.chatbot.capability_registry as cap_reg
    from dataclasses import replace

    _real_resolve = cap_reg.resolve_capability

    def _test_resolve(contract):
        resolved = _real_resolve(contract)
        if resolved.job == "brief":
            return replace(
                resolved,
                skip_vector_retrieval=False,
                serve_status="served",
                vector_policy="companion",
                vector_allow=["news", "public_reports", "academic_papers"],
            )
        if resolved.job == "diagnose":
            return replace(
                resolved,
                skip_vector_retrieval=False,
                serve_status="served",
                vector_policy="fallback_only",
            )
        return resolved

    stack.enter_context(
        mock.patch.object(graph_mod, "resolve_capability", side_effect=_test_resolve)
    )
    stack.enter_context(
        mock.patch.object(
            graph_mod,
            "typed_context_pack",
            side_effect=lambda items, contract, **kwargs: list(items),
        )
    )


# ---------------------------------------------------------------------------
# 1. Data-query happy path — answer + citations + ACF signal
# ---------------------------------------------------------------------------

def test_pipeline_data_query_returns_answer_citations_and_acf() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            news=[_news_chunk()],
            academic=[_academic_chunk()],
            ota=[_ota_chunk()],
            llm_answer="Senegal raised its rice self-sufficiency target for 2024.[1]",
        )
        result = run_rag("What is Senegal's rice policy for 2024?")

    assert result.get("answer")
    assert "I don't have OpenTrace data" not in result["answer"]
    # ACF Path B signal present on every response.
    assert result.get("acf_band")
    assert isinstance(result.get("acf_score"), (int, float))
    assert result.get("acf_note") or result.get("acf_explanation")
    # Citations populated from packed sources (inline [N] stripped in default chat).
    assert result.get("citations")
    assert result["citations"][0]["id"] == 1
    assert "[1]" not in result["answer"]
    # Usage accounting wired.
    assert "usage" in result


# ---------------------------------------------------------------------------
# 2. Direct-answer-first backstop fires end-to-end
# ---------------------------------------------------------------------------

def test_pipeline_strips_preamble_end_to_end() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            news=[_news_chunk()],
            academic=[_academic_chunk()],
            llm_answer="Based on the context, Senegal raised its rice target.[1]",
        )
        result = run_rag("Senegal rice policy?")

    assert not result["answer"].lower().startswith("based on the context")
    assert "Senegal raised its rice target." in result["answer"]


# ---------------------------------------------------------------------------
# 3. No-data path — structured gap message, no fabrication
# ---------------------------------------------------------------------------

def test_pipeline_no_context_returns_structured_gap() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        # All retrievers empty → nothing reaches the generator.
        _install_pipeline_mocks(stack, news=[], academic=[], ota=[], llm_answer="should not appear")
        result = run_rag("What are tulip exports from Antarctica?")

    assert result.get("citations") == []
    assert result.get("task_mode") == "clarify"
    answer = result.get("answer") or ""
    assert "I need a bit more detail" in answer or "Cannot ground" in answer


# ---------------------------------------------------------------------------
# 4. Meta path short-circuits with ACF HIGH
# ---------------------------------------------------------------------------

def test_pipeline_meta_query_short_circuits_high_acf() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(graph_mod, "is_meta_query", return_value=True))
        stack.enter_context(
            mock.patch(
                "ml.rag.chatbot.assistant_identity.generate_meta_answer",
                return_value="I am ADZA, OpenTrace's agricultural intelligence assistant.",
            )
        )
        result = run_rag("Who are you?")

    assert "ADZA" in result["answer"]
    assert result.get("acf_band") == "strong"
    assert result.get("acf_score") == 90


# ---------------------------------------------------------------------------
# 5. session_id threads through cleanly (Week-2 regression guard)
# ---------------------------------------------------------------------------

def test_pipeline_accepts_session_id() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            news=[_news_chunk()],
            academic=[_academic_chunk()],
            llm_answer="Senegal rice policy summary.[1]",
        )
        result = run_rag("Senegal rice policy?", session_id="itest_session_123")

    assert result.get("answer")
    assert result.get("acf_band")
    assert "usage" in result


def _gen_plan(result: dict) -> dict:
    plan = result.get("generation_plan")
    assert isinstance(plan, dict), "generation_plan missing from pipeline result"
    return plan


# ---------------------------------------------------------------------------
# 6. Generation plan golden paths (post-retrieval strategy)
# ---------------------------------------------------------------------------


def test_pipeline_ghana_rice_generation_plan_numeric_fact_bq_first() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            bq_results=[_ghana_rice_bq_chunk()],
            llm_answer="Ghana produced 973,000 metric tons of rice in 2020.",
        )
        result = run_rag("What was Ghana rice production in 2020?")

    plan = _gen_plan(result)
    assert plan.get("answer_shape") == "numeric_fact"
    assert (plan.get("evidence_priority") or [None])[0] == "bigquery"
    assert plan.get("must_ground_in") == "bigquery"


def test_pipeline_compare_generation_plan() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            bq_results=[
                _maize_bq_chunk(country="Nigeria", year=2022),
                _maize_bq_chunk(country="Kenya", year=2022, value=410000),
            ],
            news=[_news_chunk()],
            llm_answer="Nigeria and Kenya differ in maize output.",
        )
        result = run_rag(
            "Compare Nigeria and Kenya maize production in 2022.",
            plan_type="Agribusinesses",
        )

    plan = _gen_plan(result)
    assert plan.get("answer_shape") == "comparison"


def test_pipeline_briefing_generation_plan() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            public_reports=[_public_report_chunk()],
            ota=[_ota_chunk()],
            llm_answer="- Kenya IPC Phase 2 expanded.\n- Rice prices stable.",
        )
        result = run_rag("Give me a food security briefing for Kenya in 2024.")

    plan = _gen_plan(result)
    assert plan.get("answer_shape") == "briefing_digest"


def test_pipeline_research_generation_plan() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            academic=[_academic_chunk()],
            policies=[_news_chunk()],
            llm_answer="Research shows drought-tolerant varieties improve yields.",
        )
        result = run_rag("What does research say about drought-tolerant rice varieties?")

    plan = _gen_plan(result)
    assert plan.get("answer_shape") == "research_synthesis"


def test_pipeline_export_generation_plan() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(
            stack,
            bq_results=[_ghana_rice_bq_chunk()],
            llm_answer="Table caption for Ghana rice production export.",
        )
        result = run_rag(
            "Export csv Ghana rice production 2015-2020",
            plan_type="Agribusinesses",
            export_enabled=True,
        )

    plan = _gen_plan(result)
    assert plan.get("answer_shape") == "export_table"
    assert (plan.get("evidence_priority") or [None])[0] == "bigquery"


def test_pipeline_gap_generation_plan() -> None:
    import contextlib

    with contextlib.ExitStack() as stack:
        _install_pipeline_mocks(stack, news=[], academic=[], ota=[], llm_answer="should not appear")
        result = run_rag("What was Ghana rice production in 2020?")

    plan = _gen_plan(result)
    assert plan.get("output_type") == "insufficient"
    assert plan.get("answer_shape") == "gap_ack"
    assert plan.get("must_ground_in") == "any"
