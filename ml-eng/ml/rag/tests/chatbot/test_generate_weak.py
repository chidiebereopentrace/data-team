"""Law tests for generate_weak evidence tier."""
from __future__ import annotations

from unittest.mock import patch

from ml.rag.chatbot.acf_scoring import weak_orientation_acf
from ml.rag.chatbot.empty_policy import (
    empty_policy_for_class,
    parse_empty_policy,
    should_generate_weak,
)
from ml.rag.chatbot.generator import generate
from ml.rag.chatbot.schema_card import load_schema_card
from ml.rag.chatbot.turn_contract import TurnContract


def test_schema_cards_default_generate_weak() -> None:
    for code in ("PROD", "PRC", "FS", "FVC", "HDI"):
        card = load_schema_card(code)
        assert card is not None
        assert parse_empty_policy(str(card.get("empty_policy"))) == "generate_weak"


def test_weak_acf_score_bounded() -> None:
    acf = weak_orientation_acf()
    assert acf.band in ("low", "no_evidence")
    assert acf.score <= 25
    assert acf.applied_ceiling == "weak_orientation"


def test_should_generate_weak_requires_executed_retrieve() -> None:
    state = {
        "task_mode": "fact_lookup",
        "supervisor_plan": {"classes": ["PROD"]},
        "turn_contract": TurnContract(measure_id="production", job="fact").to_dict(),
    }
    assert should_generate_weak(state, context_items=[], exec_flags={"structured_bq_empty": True}) is False


def test_should_generate_weak_on_empty_after_execute() -> None:
    state = {
        "task_mode": "fact_lookup",
        "supervisor_plan": {"classes": ["PROD"]},
        "bq_sql_plan": {"bq_sql_queries": ["SELECT 1"]},
        "turn_contract": TurnContract(
            measure_id="production",
            job="fact",
            geo_grain="africa",
            serve_status="served",
        ).to_dict(),
    }
    assert should_generate_weak(
        state,
        context_items=[],
        exec_flags={"structured_bq_empty": True},
    )


def test_typed_gap_card_blocks_weak() -> None:
    assert empty_policy_for_class("PROD") == "generate_weak"


@patch("ml.rag.chatbot.generator._call_llama")
def test_generate_weak_calls_llm_with_forced_acf(mock_llm) -> None:
    mock_llm.return_value = (
        "Yam and cassava are widely grown across West and Central Africa, "
        "but I do not have federated production rows for your filter."
    )
    result = generate(
        "how many countries in africa produce yam",
        [{"content": "[No federated evidence]", "_context_kind": "filter_miss"}],
        generate_weak=True,
        task_mode="fact_lookup",
    )
    assert mock_llm.called
    assert result.citations == []
    assert result.acf is not None
    assert result.acf.score <= 25
    assert "federated" in (result.acf.explanation or "").lower() or result.acf.band == "low"
