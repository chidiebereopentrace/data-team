"""Parametrized tests for post-retrieval generation strategy plans."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.agri_measure_ontology import MEASURES, MeasureHit
from ml.rag.chatbot.generation_plan import (
    build_generation_plan,
    generation_plan_addendum,
)
from ml.rag.tests.chatbot.fixtures.generation_plan_matrix import (
    CORPUS_KINDS,
    MEASURE_IDS,
    QUESTION_SHAPES,
    TASK_MODES,
    build_matrix_cases,
)


def _measure_hit(measure_id: str | None) -> MeasureHit | None:
    if not measure_id or measure_id not in MEASURES:
        return None
    spec = MEASURES[measure_id]
    return MeasureHit(spec, score=100, matched_alias=measure_id)


@pytest.mark.parametrize("task_mode", TASK_MODES)
def test_every_task_mode_has_expected_shape(task_mode: str) -> None:
    from ml.rag.tests.chatbot.fixtures.generation_plan_matrix import (
        CASE_QUERIES,
        context_for_task_mode,
        expected_shape_for_task_mode,
    )

    ctx = context_for_task_mode(task_mode)
    query_by_mode = {
        "fact_lookup": CASE_QUERIES["numeric_fact"],
        "briefing": CASE_QUERIES["briefing"],
        "research": CASE_QUERIES["research_synthesis"],
        "data_export_only": CASE_QUERIES["export_only"],
        "analytical": CASE_QUERIES["comparison"],
        "clarify": "rice production?",
        "chat": CASE_QUERIES["policy_narrative"],
    }
    plan = build_generation_plan(
        query_by_mode[task_mode],
        task_mode=task_mode,
        reranked_context=ctx,
        measure_id="production",
    )
    assert plan.answer_shape == expected_shape_for_task_mode(task_mode)
    if ctx:
        assert generation_plan_addendum(plan)


@pytest.mark.parametrize("kind", CORPUS_KINDS)
def test_every_corpus_kind_can_be_priority_head(kind: str) -> None:
    from ml.rag.tests.chatbot.fixtures.generation_plan_matrix import context_for_kind

    ctx = context_for_kind(kind, 2)
    plan = build_generation_plan(
        "Ghana agricultural data?",
        task_mode="chat",
        reranked_context=ctx,
        measure_id="production" if kind == "bigquery" else "news_briefing",
    )
    assert plan.evidence_priority
    assert plan.evidence_priority[0] == kind or kind in plan.evidence_priority
    assert generation_plan_addendum(plan)


@pytest.mark.parametrize("measure_id", MEASURE_IDS)
def test_every_measure_has_plan_stub(measure_id: str) -> None:
    from ml.rag.tests.chatbot.fixtures.generation_plan_matrix import mixed_bq_news_context

    plan = build_generation_plan(
        f"Question about {measure_id}",
        task_mode="chat",
        measure_hit=_measure_hit(measure_id),
        reranked_context=mixed_bq_news_context(),
    )
    assert plan.ontology.measure_id == measure_id
    assert plan.rationale
    assert generation_plan_addendum(plan)


@pytest.mark.parametrize("shape", QUESTION_SHAPES)
def test_question_shapes_produce_distinct_stable_plans(shape: str) -> None:
    from ml.rag.tests.chatbot.fixtures.generation_plan_matrix import (
        CASE_QUERIES,
        decomposition_for_shape,
        empty_context,
        ghana_rice_bq_context,
        mixed_bq_news_context,
        mixed_narrative_context,
        academic_heavy_context,
        context_for_kind,
    )

    ctx_map = {
        "gap": empty_context(),
        "numeric_fact": ghana_rice_bq_context(),
        "export_only": ghana_rice_bq_context(),
        "ranking": context_for_kind("bigquery", 3),
        "briefing": mixed_narrative_context(),
        "research_synthesis": academic_heavy_context(),
    }
    ctx = ctx_map.get(shape, mixed_bq_news_context())
    task_mode = {
        "gap": "chat",
        "numeric_fact": "fact_lookup",
        "ranking": "fact_lookup",
        "comparison": "analytical",
        "trend": "analytical",
        "briefing": "briefing",
        "research_synthesis": "research",
        "export_only": "data_export_only",
        "policy_narrative": "chat",
    }[shape]
    plan = build_generation_plan(
        CASE_QUERIES[shape],
        task_mode=task_mode,
        decomposition=decomposition_for_shape(shape),
        reranked_context=ctx,
        measure_id="production",
    )
    expected = {
        "numeric_fact": "numeric_fact",
        "ranking": "ranking",
        "comparison": "comparison",
        "trend": "trend",
        "briefing": "briefing_digest",
        "research_synthesis": "research_synthesis",
        "export_only": "export_table",
        "gap": "gap_ack",
        "policy_narrative": "policy_narrative",
    }[shape]
    assert plan.answer_shape == expected
    assert generation_plan_addendum(plan)


@pytest.mark.parametrize("case", build_matrix_cases(), ids=lambda c: c["id"])
def test_matrix_case(case: dict) -> None:
    plan = build_generation_plan(
        case["query"],
        task_mode=case["task_mode"],
        decomposition=case.get("decomposition"),
        measure_hit=_measure_hit(case.get("measure_id")),
        reranked_context=case["context"],
        plan_type=case.get("plan_type"),
        measure_id=case.get("measure_id"),
    )
    assert plan.answer_shape == case["expect_shape"]
    if case.get("expect_priority_head"):
        assert plan.evidence_priority
        assert plan.evidence_priority[0] == case["expect_priority_head"]
    if case.get("expect_ground"):
        assert plan.must_ground_in == case["expect_ground"]
    addendum = generation_plan_addendum(plan)
    assert addendum or case["expect_shape"] == "gap_ack"
    if addendum:
        assert len(addendum) <= 400


def test_ghana_rice_numeric_fact_bq_first() -> None:
    from ml.rag.tests.chatbot.fixtures.generation_plan_matrix import ghana_rice_bq_context

    plan = build_generation_plan(
        "What was Ghana rice production in 2020?",
        task_mode="fact_lookup",
        measure_hit=_measure_hit("production"),
        reranked_context=ghana_rice_bq_context(),
        decomposition={"geography": ["Ghana"], "time_start": "2020", "time_end": "2020"},
    )
    assert plan.answer_shape == "numeric_fact"
    assert plan.evidence_priority[0] == "bigquery"
    assert plan.must_ground_in == "bigquery"
    assert plan.lead_with == "structured_value"
