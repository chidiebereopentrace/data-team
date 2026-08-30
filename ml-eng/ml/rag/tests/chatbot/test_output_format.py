"""Property tests for typed output geometry."""
from __future__ import annotations

import pytest

from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.generation_plan import build_generation_plan
from ml.rag.chatbot.generator import _clean_answer, _strip_forbidden_headings
from ml.rag.chatbot.output_format import (
    answer_subtopics,
    format_prompt_for_type,
    output_type_from_contract,
    output_type_from_job,
    persona_implications_block,
    render_insufficient,
)
from ml.rag.chatbot.query_decomposer import decompose_query
from ml.rag.chatbot.turn_contract import TurnContract
from ml.rag.tests.chatbot.fixtures.generation_plan_matrix import ghana_rice_bq_context


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ("fact", "fact"),
        ("trend", "trend"),
        ("rank", "compare"),
        ("compare", "compare"),
        ("list", "list"),
        ("outlook", "outlook"),
        ("diagnose", "diagnosis"),
        ("brief", "brief"),
    ],
)
def test_job_maps_to_output_type(job: str, expected: str) -> None:
    assert output_type_from_job(job) == expected
    assert output_type_from_job("outlook") != "trend"


def test_outlook_query_compiles_to_outlook_output_type() -> None:
    query = "What is the food security outlook for Somalia next lean season?"
    dec = decompose_query(query)
    contract = compile_turn_contract(query, dec, task_mode_hint="chat")
    assert contract.job == "outlook"
    out = output_type_from_contract(contract, evidence_tier="strong", has_usable_context=True)
    assert out == "outlook"


def test_fact_prompt_has_no_key_findings() -> None:
    prompt = format_prompt_for_type("fact", grain_window_line="national; calendar 2024; maize")
    assert "Lead:" in prompt
    assert "Spine:" in prompt
    assert "Key Findings" not in prompt
    assert "No ## headings" in prompt


def test_outlook_prompt_allows_phase_headings() -> None:
    prompt = format_prompt_for_type("outlook")
    assert "Now" in prompt
    assert "Next lean season" in prompt


def test_insufficient_renderer_leads_with_miss() -> None:
    contract = TurnContract(
        measure_id="employment_share",
        geo_grain="admin2",
        geo=["Uganda"],
        serve_status="unsupported_grain",
        job="fact",
    )
    text = render_insufficient(contract)
    assert "Cannot ground" in text
    assert "admin2" in text.lower() or "admin2" in text


def test_generation_plan_sets_output_type_for_served_production() -> None:
    contract = TurnContract(measure_id="production", geo_grain="country", job="fact")
    plan = build_generation_plan(
        "What is rice production in Ghana in 2023?",
        task_mode="fact_lookup",
        reranked_context=ghana_rice_bq_context(),
        turn_contract=contract.to_dict(),
    )
    assert plan.output_type == "fact"
    assert plan.report_sections == ()


def test_generation_plan_insufficient_on_empty_context() -> None:
    contract = TurnContract(measure_id="production", job="fact")
    plan = build_generation_plan(
        "rice production Ghana",
        turn_contract=contract.to_dict(),
        reranked_context=[],
    )
    assert plan.output_type == "insufficient"
    assert plan.answer_shape == "gap_ack"


def test_strip_forbidden_headings_on_fact() -> None:
    raw = "## Key Findings\nNothing here.\n\nProduction was 1Mt in 2023."
    cleaned = _strip_forbidden_headings(raw, "fact")
    assert "Key Findings" not in cleaned
    assert "Production was 1Mt" in cleaned


def test_clean_answer_strips_trend_heading_for_fact() -> None:
    raw = "## Trend\nUp 5%.\n\nMaize production rose in 2024."
    cleaned = _clean_answer(raw, output_type="fact")
    assert "## Trend" not in cleaned


def test_compare_three_geo_subtopics_in_prompt() -> None:
    contract = TurnContract(
        measure_id="market_price",
        geo_grain="country",
        geo=["Kenya", "Tanzania", "Uganda"],
        job="compare",
    )
    subtopics = answer_subtopics(
        {"geography": ["Kenya", "Tanzania", "Uganda"]},
        contract,
        "compare",
    )
    assert subtopics == ("Kenya", "Tanzania", "Uganda")
    prompt = format_prompt_for_type("compare", answer_subtopics=subtopics)
    assert "Kenya, Tanzania, Uganda" in prompt
    assert "### subheads ONLY for" in prompt


def test_fact_single_geo_has_no_subtopics() -> None:
    contract = TurnContract(
        measure_id="production",
        geo_grain="country",
        geo=["Ghana"],
        job="fact",
    )
    assert answer_subtopics(None, contract, "fact") == ()


def test_fact_breakdown_subtopics_for_sex() -> None:
    contract = TurnContract(
        measure_id="employment_share",
        geo_grain="country",
        geo=["Uganda"],
        job="fact",
        breakdown=["sex"],
    )
    assert answer_subtopics(None, contract, "fact") == ("Male", "Female")


def test_persona_implications_same_spine_different_voice() -> None:
    spine = format_prompt_for_type(
        "compare",
        grain_window_line="national; calendar 2024; maize",
        answer_subtopics=("Kenya", "Tanzania"),
    )
    gov = format_prompt_for_type(
        "compare",
        persona="Government",
        grain_window_line="national; calendar 2024; maize",
        include_implications=True,
        answer_subtopics=("Kenya", "Tanzania"),
        implications_text=persona_implications_block(
            "Government", "compare", has_spine=True
        )
        or "",
    )
    agb = format_prompt_for_type(
        "compare",
        persona="Agribusinesses",
        grain_window_line="national; calendar 2024; maize",
        include_implications=True,
        answer_subtopics=("Kenya", "Tanzania"),
        implications_text=persona_implications_block(
            "Agribusinesses", "compare", has_spine=True
        )
        or "",
    )
    assert spine.count("Lead:") == gov.count("Lead:") == agb.count("Lead:")
    assert "policy or fiscal" in gov.lower()
    assert "sourcing" in agb.lower()
    assert gov != agb


def test_compare_without_spine_has_no_agribusiness_exposure() -> None:
    assert persona_implications_block("Agribusinesses", "compare", has_spine=False) is None


def test_strip_preserves_compare_country_subhead() -> None:
    raw = "## Key Findings\nSkip.\n\n### Kenya\nVolatility was 12%.\n\n### Tanzania\nVolatility was 9%."
    cleaned = _strip_forbidden_headings(
        raw,
        "compare",
        allowed_subtopics=("Kenya", "Tanzania"),
    )
    assert "Key Findings" not in cleaned
    assert "### Kenya" in cleaned
    assert "### Tanzania" in cleaned


def test_insufficient_persona_rephrase() -> None:
    contract = TurnContract(
        measure_id="market_price",
        geo_grain="country",
        geo=["Kenya", "Tanzania", "Uganda"],
        job="compare",
    )
    text = render_insufficient(contract, category="Agribusinesses")
    assert "calendar window" in text.lower()
