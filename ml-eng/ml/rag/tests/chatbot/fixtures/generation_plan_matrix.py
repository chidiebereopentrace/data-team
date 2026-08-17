"""Synthetic fixtures for generation-plan parametrized tests."""
from __future__ import annotations

from typing import Any

from ml.rag.chatbot.agri_measure_ontology import MEASURES
from ml.rag.chatbot.task_mode import TaskMode

TASK_MODES: tuple[TaskMode, ...] = (
    "clarify",
    "analytical",
    "fact_lookup",
    "briefing",
    "data_export_only",
    "research",
    "chat",
)

CORPUS_KINDS: tuple[str, ...] = (
    "bigquery",
    "news",
    "academic",
    "policy",
    "public_report",
    "formation",
    "ota_insight",
    "web",
)

MEASURE_IDS: tuple[str, ...] = tuple(MEASURES.keys())

QUESTION_SHAPES: tuple[str, ...] = (
    "numeric_fact",
    "ranking",
    "comparison",
    "trend",
    "policy_narrative",
    "research_synthesis",
    "briefing",
    "export_only",
    "gap",
)

PLAN_TYPES: tuple[str, ...] = (
    "Free",
    "Farmers",
    "Government",
    "NGOs",
    "Agribusinesses",
    "Integrated",
)


def _stub_item(kind: str, *, content: str = "Sample evidence text.") -> dict[str, Any]:
    meta: dict[str, Any] = {"doc_kind": kind}
    if kind == "bigquery":
        meta.update(
            {
                "source_id": f"bq:{kind}:1",
                "country_name": "Ghana",
                "product_name": "Rice",
                "year": 2020,
                "value": 973000,
                "unit": "t",
            }
        )
    elif kind == "news":
        meta.update({"title": "News headline", "publisher": "AgriNews", "published_at": "2024-01-01"})
    elif kind == "academic":
        meta.update({"article_title": "Paper", "authors": "Author A.", "publication_year": "2022"})
    elif kind == "policy":
        meta.update({"title": "Policy brief", "published_at": "2023-06-01"})
    elif kind == "public_report":
        meta.update({"title": "IPC report", "published_at": "2024-03-01"})
    elif kind == "ota_insight":
        meta.update({"metric_text": "Output +5%", "as_of_date": "2024-01-01"})
    elif kind == "formation":
        meta.update({"title": "Training module"})
    elif kind == "web":
        meta.update({"title": "Web page", "url": "https://example.com/page"})
    return {
        "content": content,
        "_context_kind": kind,
        "source": kind,
        "metadata": meta,
    }


def context_for_kind(kind: str, count: int = 2) -> list[dict[str, Any]]:
    return [_stub_item(kind) for _ in range(max(1, count))]


def mixed_bq_news_context() -> list[dict[str, Any]]:
    return context_for_kind("bigquery", 2) + context_for_kind("news", 2)


def mixed_narrative_context() -> list[dict[str, Any]]:
    return (
        context_for_kind("news", 2)
        + context_for_kind("public_report", 1)
        + context_for_kind("academic", 1)
    )


def ota_heavy_context() -> list[dict[str, Any]]:
    return context_for_kind("ota_insight", 3) + context_for_kind("news", 1)


def academic_heavy_context() -> list[dict[str, Any]]:
    return context_for_kind("academic", 3) + context_for_kind("policy", 1)


def empty_context() -> list[dict[str, Any]]:
    return []


def error_bq_only_context() -> list[dict[str, Any]]:
    return [
        {
            "content": "[bq execution error] no rows returned",
            "_context_kind": "bigquery",
            "metadata": {"execution_error": True},
        }
    ]


def ghana_rice_bq_context() -> list[dict[str, Any]]:
    return [
        {
            "content": "Ghana produced 973,000 metric tons of rice in 2020.",
            "_context_kind": "bigquery",
            "source": "bigquery",
            "metadata": {
                "source_id": "stg_faostat_production:country_name=Ghana:year=2020",
                "country_name": "Ghana",
                "product_name": "Rice",
                "element": "Production",
                "year": 2020,
                "value": 973000,
                "unit": "t",
            },
        }
    ]


CASE_QUERIES: dict[str, str] = {
    "numeric_fact": "What was Ghana rice production in 2020?",
    "ranking": "Which African countries produced the most rice in 2020?",
    "comparison": "Compare Nigeria and Kenya maize production in 2022.",
    "trend": "How has Kenya maize production changed since 2010?",
    "policy_narrative": "What are the main rice policy challenges in West Africa?",
    "research_synthesis": "What does research say about drought-tolerant rice varieties?",
    "briefing": "Give me a food security briefing for Kenya.",
    "export_only": "Export Ghana rice production table for 2015-2020.",
    "gap": "What are tulip exports from Antarctica?",
}


def decomposition_for_shape(shape: str) -> dict[str, Any]:
    base: dict[str, Any] = {"domains": ["production"]}
    if shape == "numeric_fact":
        base.update({"geography": ["Ghana"], "time_start": "2020", "time_end": "2020", "intent": "descriptive"})
    elif shape == "ranking":
        base.update({"geography": ["Africa"], "time_start": "2020", "time_end": "2020", "intent": "descriptive"})
    elif shape == "comparison":
        base.update({"geography": ["Nigeria", "Kenya"], "intent": "compare", "time_start": "2022", "time_end": "2022"})
    elif shape == "trend":
        base.update({"geography": ["Kenya"], "intent": "diagnostic", "time_start": "2010", "time_end": "2024"})
    elif shape == "briefing":
        base.update({"geography": ["Kenya"], "intent": "descriptive"})
    elif shape == "research_synthesis":
        base.update({"domains": ["research"], "intent": "descriptive"})
    elif shape == "export_only":
        base.update({"geography": ["Ghana"], "africa_panel": True, "time_start": "2015", "time_end": "2020"})
    elif shape == "gap":
        base.update({"geography": ["Antarctica"]})
    else:
        base.update({"intent": "descriptive"})
    return base


def context_for_task_mode(task_mode: str) -> list[dict[str, Any]]:
    if task_mode == "fact_lookup":
        return context_for_kind("bigquery", 2)
    if task_mode == "briefing":
        return mixed_narrative_context()
    if task_mode == "research":
        return academic_heavy_context()
    if task_mode == "data_export_only":
        return context_for_kind("bigquery", 3)
    if task_mode == "analytical":
        return mixed_bq_news_context()
    if task_mode == "clarify":
        return empty_context()
    return mixed_bq_news_context()


def expected_shape_for_task_mode(task_mode: str) -> str:
    mapping = {
        "fact_lookup": "numeric_fact",
        "briefing": "briefing_digest",
        "research": "research_synthesis",
        "data_export_only": "export_table",
        "analytical": "comparison",
        "clarify": "gap_ack",
        "chat": "policy_narrative",
    }
    return mapping.get(task_mode, "policy_narrative")


def build_matrix_cases() -> list[dict[str, Any]]:
    """Representative parametrized cases (~80–120), not full Cartesian product."""
    cases: list[dict[str, Any]] = []

    for task_mode in TASK_MODES:
        ctx = context_for_task_mode(task_mode)
        query = CASE_QUERIES.get("numeric_fact" if task_mode == "fact_lookup" else "policy_narrative", "Ghana rice?")
        if task_mode == "briefing":
            query = CASE_QUERIES["briefing"]
        elif task_mode == "research":
            query = CASE_QUERIES["research_synthesis"]
        elif task_mode == "data_export_only":
            query = CASE_QUERIES["export_only"]
        elif task_mode == "analytical":
            query = CASE_QUERIES["comparison"]
        elif task_mode == "clarify":
            query = "rice production?"
        cases.append(
            {
                "id": f"task_{task_mode}",
                "query": query,
                "task_mode": task_mode,
                "context": ctx,
                "measure_id": "production" if task_mode != "research" else "research_synthesis",
                "decomposition": decomposition_for_shape(
                    "briefing" if task_mode == "briefing" else "numeric_fact"
                ),
                "expect_shape": expected_shape_for_task_mode(task_mode),
                "expect_priority_head": "bigquery"
                if task_mode in ("fact_lookup", "data_export_only")
                else ("academic" if task_mode == "research" else None),
                "expect_ground": "bigquery"
                if task_mode in ("fact_lookup", "data_export_only") and ctx
                else ("any" if task_mode == "clarify" else None),
            }
        )

    for kind in CORPUS_KINDS:
        ctx = context_for_kind(kind, 2)
        numeric = kind == "bigquery"
        cases.append(
            {
                "id": f"corpus_{kind}",
                "query": CASE_QUERIES["numeric_fact"] if numeric else CASE_QUERIES["policy_narrative"],
                "task_mode": "fact_lookup" if numeric else "chat",
                "context": ctx,
                "measure_id": "production",
                "decomposition": decomposition_for_shape("numeric_fact" if numeric else "policy_narrative"),
                "expect_shape": "numeric_fact" if numeric else "policy_narrative",
                "expect_priority_head": kind,
                "expect_ground": "bigquery" if numeric else "any",
            }
        )

    for measure_id in MEASURE_IDS:
        if measure_id in ("news_briefing", "research_synthesis", "research_meta"):
            ctx = mixed_narrative_context() if measure_id != "research_synthesis" else academic_heavy_context()
            task_mode = "briefing" if measure_id == "news_briefing" else "research"
            shape = "briefing_digest" if measure_id == "news_briefing" else "research_synthesis"
        elif measure_id == "data_export_panel":
            ctx = context_for_kind("bigquery", 2)
            task_mode = "data_export_only"
            shape = "export_table"
        elif measure_id in ("production", "yield", "trade", "market_price"):
            ctx = mixed_bq_news_context()
            task_mode = "fact_lookup"
            shape = "numeric_fact"
            query = CASE_QUERIES["numeric_fact"] if measure_id == "production" else f"Ghana {measure_id.replace('_', ' ')} volume 2020?"
        elif measure_id == "food_security_ipc":
            ctx = context_for_kind("public_report", 2) + context_for_kind("news", 1)
            task_mode = "briefing"
            shape = "briefing_digest"
        else:
            ctx = mixed_bq_news_context()
            task_mode = "chat"
            shape = "policy_narrative"
        cases.append(
            {
                "id": f"measure_{measure_id}",
                "query": query if measure_id in ("production", "yield", "trade", "market_price") else f"Query about {measure_id.replace('_', ' ')} in Ghana",
                "task_mode": task_mode,
                "context": ctx,
                "measure_id": measure_id,
                "decomposition": {"geography": ["Ghana"], "primary_measures": [measure_id]},
                "expect_shape": shape,
                "expect_priority_head": None,
                "expect_ground": None,
            }
        )

    for shape in QUESTION_SHAPES:
        ctx = empty_context() if shape == "gap" else (
            ghana_rice_bq_context() if shape in ("numeric_fact", "export_only") else mixed_bq_news_context()
        )
        if shape == "ranking":
            ctx = context_for_kind("bigquery", 3)
        elif shape == "briefing":
            ctx = mixed_narrative_context()
        elif shape == "research_synthesis":
            ctx = academic_heavy_context()
        task_mode = {
            "numeric_fact": "fact_lookup",
            "ranking": "fact_lookup",
            "comparison": "analytical",
            "trend": "analytical",
            "briefing": "briefing",
            "research_synthesis": "research",
            "export_only": "data_export_only",
            "gap": "chat",
            "policy_narrative": "chat",
        }[shape]
        expect_shape = {
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
        cases.append(
            {
                "id": f"shape_{shape}",
                "query": CASE_QUERIES[shape],
                "task_mode": task_mode,
                "context": ctx,
                "measure_id": "production",
                "decomposition": decomposition_for_shape(shape),
                "expect_shape": expect_shape,
                "expect_priority_head": "bigquery" if shape in ("numeric_fact", "ranking", "export_only") else None,
                "expect_ground": "bigquery" if shape in ("numeric_fact", "export_only") else ("any" if shape == "gap" else None),
            }
        )

    for plan_type in PLAN_TYPES:
        cases.append(
            {
                "id": f"plan_{plan_type.lower()}",
                "query": CASE_QUERIES["comparison"],
                "task_mode": "analytical",
                "context": mixed_bq_news_context(),
                "measure_id": "production",
                "plan_type": plan_type,
                "decomposition": decomposition_for_shape("comparison"),
                "expect_shape": "comparison",
                "expect_priority_head": "bigquery",
                "expect_ground": "any",
            }
        )

    gap_variants = [
        ("gap_empty", empty_context()),
        ("gap_error_bq", error_bq_only_context()),
    ]
    for gid, ctx in gap_variants:
        cases.append(
            {
                "id": gid,
                "query": CASE_QUERIES["gap"],
                "task_mode": "chat",
                "context": ctx,
                "measure_id": None,
                "decomposition": decomposition_for_shape("gap"),
                "expect_shape": "gap_ack",
                "expect_priority_head": None,
                "expect_ground": "any",
            }
        )

    return cases
