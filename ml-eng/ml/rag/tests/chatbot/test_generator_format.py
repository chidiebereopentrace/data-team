"""Prompt format: typed output geometry and persona register."""
from __future__ import annotations

from ml.rag.chatbot.generation_plan import build_generation_plan
from ml.rag.chatbot.generator import _build_prompt
from ml.rag.tests.chatbot.fixtures.generation_plan_matrix import mixed_bq_news_context


def _system_prompt(**kwargs) -> str:
    messages = _build_prompt("Compare maize production in Kenya and Tanzania", "Context line", **kwargs)
    return messages[0]["content"]


def test_analytical_prompt_uses_typed_geometry_not_legacy_outline() -> None:
    ctx = mixed_bq_news_context()
    for item in ctx:
        if item.get("_context_kind") == "bigquery":
            item["metadata"]["country_name"] = "Kenya"
            item["metadata"]["product_name"] = "Maize"
        if item.get("_context_kind") == "news":
            item["content"] = "[News] Maize production compared across Kenya and Tanzania."
            item["metadata"]["country"] = "Kenya"
    plan = build_generation_plan(
        "Compare maize production in Kenya and Tanzania",
        task_mode="analytical",
        category="Government",
        reranked_context=ctx,
        measure_id="production",
        decomposition={
            "intent": "compare",
            "geography": ["Kenya", "Tanzania"],
            "entities": ["maize"],
        },
    )
    system = _system_prompt(
        analytical_mode=True,
        task_mode="analytical",
        category="Government",
        generation_plan=plan.to_dict(),
        evidence_tier="strong",
    )
    assert "Regional & Country Picture" not in system
    assert "REPORT OUTLINE" not in system
    assert "ANALYTICAL VOICE RULES" not in system
    assert "OUTPUT TYPE:" in system
    assert "Kenya, Tanzania" in system
    assert "Implications (after spine)" in system


def test_government_analytical_allows_tables_in_register() -> None:
    plan = build_generation_plan(
        "Compare maize production in Kenya and Tanzania",
        task_mode="analytical",
        category="Government",
        reranked_context=mixed_bq_news_context(),
        measure_id="production",
    )
    system = _system_prompt(
        analytical_mode=True,
        task_mode="analytical",
        category="Government",
        generation_plan=plan.to_dict(),
    )
    assert "PROSE REGISTER" in system
    assert "markdown tables" in system.lower()


def test_farmers_analytical_forbids_tables_and_fixed_headings() -> None:
    plan = build_generation_plan(
        "Will maize prices hurt my feed costs this season?",
        task_mode="analytical",
        category="Farmers",
        reranked_context=mixed_bq_news_context(),
        measure_id="market_price",
    )
    system = _system_prompt(
        analytical_mode=True,
        task_mode="analytical",
        category="Farmers",
        generation_plan=plan.to_dict(),
    )
    assert "## Key Findings" not in system
    assert "Do not use markdown tables" in system
    assert "plain bullet points" in system.lower()


def test_same_task_mode_different_register_by_category() -> None:
    ctx = mixed_bq_news_context()
    gov_plan = build_generation_plan(
        "Compare production",
        task_mode="analytical",
        category="Government",
        reranked_context=ctx,
    )
    farm_plan = build_generation_plan(
        "Compare production",
        task_mode="analytical",
        category="Farmers",
        reranked_context=ctx,
    )
    gov_sys = _system_prompt(
        analytical_mode=True,
        task_mode="analytical",
        generation_plan=gov_plan.to_dict(),
    )
    farm_sys = _system_prompt(
        analytical_mode=True,
        task_mode="analytical",
        generation_plan=farm_plan.to_dict(),
    )
    assert "markdown tables" in gov_sys.lower()
    assert "Do not use markdown tables" in farm_sys


def test_investor_measure_uses_diagnosis_type_without_report_sections() -> None:
    plan = build_generation_plan(
        "Best African country for agricultural investment",
        task_mode="analytical",
        category="Agribusinesses",
        reranked_context=mixed_bq_news_context(),
        measure_id="investor_best_country",
    )
    assert plan.report_sections == ()
    assert plan.output_type in ("diagnosis", "compare", "fact")
    assert plan.has_usable_spine is True
