"""Unit tests for advisory metacognition (diagnosis vs fact, apply, verify)."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from ml.rag.chatbot.global_reasoner import _pin_job
from ml.rag.chatbot.graph import node_metacog_revise
from ml.rag.chatbot.metacog import (
    MetacogPlan,
    MetacogReviseDecision,
    MetacogSelfCheck,
    MetacogVerify,
    apply_metacog_to_turn,
    apply_revise_decision,
    build_metacog_plan,
    clear_metacog_cache,
    complete_self_check,
    corpora_for_evidence_needs,
    evidence_sample,
    heuristic_metacog_plan,
    metacog_diagnosis_addendum,
    metacog_enabled,
    metacog_revise_enabled,
    metacog_self_check_enabled,
    propose_metacog_revise,
    self_check_applies,
    verify_metacog_evidence,
)
from ml.rag.chatbot.turn_contract import TimeSpec, TurnContract


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_metacog_cache()
    yield
    clear_metacog_cache()


def test_metacog_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_METACOG", raising=False)
    assert not metacog_enabled()


def test_metacog_explicit_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    assert metacog_enabled()


def test_heuristic_diagnose_vs_fact() -> None:
    diag = heuristic_metacog_plan(
        "What caused cassava production in Nigeria to change over the last five years?"
    )
    assert diag.advisory_mode == "diagnose"
    assert diag.claim_budget == "causal_careful"
    assert "bq" in diag.evidence_needs

    fact = heuristic_metacog_plan("How much maize was produced in Kenya in 2020?")
    assert fact.advisory_mode == "fact"
    assert fact.must_ground_in == "bigquery"
    assert fact.claim_budget == "strict_numeric"


def test_apply_metacog_pins_diagnose() -> None:
    turn = TurnContract(job="fact", measure_id="production")
    plan = MetacogPlan(
        advisory_mode="diagnose",
        confidence=0.9,
        evidence_needs=("bq", "research", "ota"),
        claim_budget="causal_careful",
        source="heuristic",
    )
    out = apply_metacog_to_turn(turn, plan)
    assert out.job == "diagnose"
    assert out.plan_type == "narrative"
    assert out.skip_bq is False
    assert "academic_papers" in out.vector_allow or "ota" in out.vector_allow


def test_apply_metacog_skips_low_confidence() -> None:
    turn = TurnContract(job="fact", measure_id="production")
    plan = MetacogPlan(
        advisory_mode="diagnose",
        confidence=0.2,
        source="heuristic",
    )
    out = apply_metacog_to_turn(turn, plan)
    assert out.job == "fact"


def test_apply_metacog_preserves_non_rag() -> None:
    turn = TurnContract(job="help")
    plan = MetacogPlan(advisory_mode="diagnose", confidence=0.99, source="heuristic")
    out = apply_metacog_to_turn(turn, plan)
    assert out.job == "help"


def test_pin_job_preserves_diagnose() -> None:
    turn = TurnContract(job="diagnose", measure_id="production")
    assert _pin_job("why did yields change over time?", turn, "chat") == "diagnose"
    assert _pin_job("compare Kenya vs Nigeria", turn, "chat") == "diagnose"


def test_corpora_for_evidence_needs() -> None:
    keys = corpora_for_evidence_needs(("research", "news", "bq"))
    assert "news" in keys
    assert "academic_papers" in keys
    assert "bq" not in keys


def test_diagnosis_addendum() -> None:
    plan = MetacogPlan(
        advisory_mode="diagnose",
        claim_budget="causal_careful",
        composer_hints=("Separate levels from drivers.",),
        source="heuristic",
    )
    text = metacog_diagnosis_addendum(plan)
    assert "diagnosis" in text.lower()
    assert "causal_careful" in text
    assert "Separate levels" in text


def test_metacog_plan_rejects_strings_in_place_of_lists() -> None:
    plan = MetacogPlan.from_dict({
        "advisory_mode": "diagnose",
        "evidence_needs": "research",
        "verify_checks": "geo_match;year_match;measure_match",
        "composer_hints": "Separate prices and livelihood impacts.",
    })
    assert plan is not None
    assert {"geo_match", "year_match", "measure_match"}.issubset(plan.verify_checks)
    assert plan.evidence_needs == ()
    assert plan.composer_hints == ()


def test_graph_keeps_new_verification_over_initial_empty_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.setenv("RAG_METACOG_VERIFY", "1")
    monkeypatch.setenv("RAG_METACOG_REVISE", "0")
    out = node_metacog_revise({
        "query": "How did policy affect food prices in Ghana?",
        "metacog_plan": MetacogPlan(advisory_mode="diagnose").to_dict(),
        "metacog_verify": None,
    })
    verification = out.get("metacog_verify")
    assert verification and verification["status"] == "gap"
    assert out["metacog_next"] == "generate"


def test_verify_gap_when_empty_evidence() -> None:
    plan = MetacogPlan(
        advisory_mode="diagnose",
        verify_checks=("geo_match", "no_invented_drivers"),
        source="heuristic",
    )
    with mock.patch("ml.rag.chatbot.metacog.metacog_verify_enabled", return_value=True):
        verified = verify_metacog_evidence(
            plan,
            query="why cassava in Nigeria",
            decomposition={"geography": ["Nigeria"]},
            bq_results=[],
            reranked_context=[],
        )
    assert verified.status == "gap"
    assert verified.evidence_tier_hint == "empty"
    assert "no_invented_drivers" in verified.failed_checks or "evidence_present" in verified.failed_checks
    assert verified.addendum


def test_verify_ok_with_bq_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.setenv("RAG_METACOG_VERIFY", "1")
    plan = MetacogPlan(
        advisory_mode="fact",
        verify_checks=("geo_match", "year_match", "measure_match"),
        source="heuristic",
    )
    verified = verify_metacog_evidence(
        plan,
        query="maize production Kenya 2020",
        decomposition={
            "geography": ["Kenya"],
            "time_start": "2020-01-01",
            "primary_measures": ["production"],
        },
        bq_results=[
            {
                "content": "Kenya maize production 2020: 4.2 Mt",
                "metadata": {"country": "Kenya", "year": 2020},
            }
        ],
        reranked_context=[],
    )
    assert verified.status == "ok"
    assert verified.evidence_tier_hint == "strong"
    assert not verified.failed_checks


def test_build_metacog_plan_uses_llm_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    payload = {
        "advisory_mode": "diagnose",
        "confidence": 0.88,
        "evidence_needs": ["bq", "research"],
        "must_ground_in": "any",
        "claim_budget": "causal_careful",
        "verify_checks": ["geo_match", "no_invented_drivers"],
        "composer_hints": ["Do not invent drivers."],
        "skip_verify": False,
        "rationale": "why-question",
    }
    with mock.patch(
        "ml.rag.chatbot.metacog.llm_chat_complete",
        return_value=json.dumps(payload),
    ) as llm:
        plan = build_metacog_plan(
            "Why did cassava production fall in Nigeria?",
            turn=TurnContract(job="fact"),
        )
    llm.assert_called_once()
    assert plan.advisory_mode == "diagnose"
    assert plan.source == "llm"
    assert plan.confidence == 0.88


@pytest.mark.parametrize("measure,commodity,place", [
    ("market_price", "maize", "Kenya"),
    ("production", "rice", "Senegal"),
])
def test_resolved_warehouse_fact_keeps_checks_without_advisory_llm(monkeypatch, measure, commodity, place) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    turn = TurnContract(
        job="fact", measure_id=measure, geo=[place], entities=[commodity],
        time_spec=TimeSpec(start="2024-01-01", end="2024-12-31"),
    )
    with mock.patch("ml.rag.chatbot.metacog.llm_chat_complete") as llm:
        plan = build_metacog_plan(f"{commodity} {measure} in {place} in 2024", turn=turn)
    llm.assert_not_called()
    assert plan.must_ground_in == "bigquery"
    assert "geo_match" in plan.verify_checks


def test_build_metacog_disabled_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "0")
    with mock.patch("ml.rag.chatbot.metacog.llm_chat_complete") as llm:
        plan = build_metacog_plan("why yields fell", turn=TurnContract(job="fact"))
    llm.assert_not_called()
    assert plan.source == "none"


def test_build_metacog_falls_back_on_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    with mock.patch(
        "ml.rag.chatbot.metacog.llm_chat_complete",
        return_value="not json at all",
    ):
        plan = build_metacog_plan(
            "What caused rice prices to rise in Kenya?",
            turn=TurnContract(job="fact"),
        )
    assert plan.source == "heuristic"
    assert plan.advisory_mode == "diagnose"


def test_revise_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.setenv("RAG_METACOG_REVISE", "1")
    assert metacog_revise_enabled()


def test_revise_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.delenv("RAG_METACOG_REVISE", raising=False)
    assert not metacog_revise_enabled()


def test_self_check_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.setenv("RAG_METACOG_SELF_CHECK", "1")
    assert metacog_self_check_enabled()


def test_self_check_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.delenv("RAG_METACOG_SELF_CHECK", raising=False)
    assert not metacog_self_check_enabled()


def test_propose_revise_skip_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.setenv("RAG_METACOG_REVISE", "0")
    plan = MetacogPlan(advisory_mode="diagnose", confidence=0.9, source="heuristic")
    verified = MetacogVerify(status="gap", evidence_tier_hint="empty")
    decision = propose_metacog_revise(
        plan, verified, query="", already_revised=0, bq_results=[], reranked_context=[]
    )
    assert decision.action == "skip"


def test_propose_revise_accept_when_gap_but_already_revised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.setenv("RAG_METACOG_REVISE", "1")
    plan = MetacogPlan(advisory_mode="diagnose", confidence=0.9, source="heuristic")
    verified = MetacogVerify(status="gap", evidence_tier_hint="empty")
    decision = propose_metacog_revise(
        plan, verified, query="", already_revised=1, bq_results=[], reranked_context=[]
    )
    assert decision.action == "accept"


def test_propose_revise_accept_when_ok() -> None:
    plan = MetacogPlan(advisory_mode="diagnose", confidence=0.9, source="heuristic")
    verified = MetacogVerify(status="ok", evidence_tier_hint="strong")
    decision = propose_metacog_revise(
        plan, verified, query="", already_revised=0, bq_results=[], reranked_context=[]
    )
    assert decision.action == "accept"


def test_propose_revise_accept_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.setenv("RAG_METACOG_REVISE", "1")
    with mock.patch(
        "ml.rag.chatbot.metacog.llm_chat_complete",
        return_value="invalid json",
    ):
        plan = MetacogPlan(advisory_mode="diagnose", confidence=0.9, source="heuristic")
        verified = MetacogVerify(status="gap", evidence_tier_hint="empty")
        decision = propose_metacog_revise(
            plan, verified, query="", already_revised=0, bq_results=[], reranked_context=[]
        )
    assert decision.action == "accept"


def test_propose_revise_replan() -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.setenv("RAG_METACOG_REVISE_MODEL_ID", "deepseek/deepseek-r1")
    payload = {
        "action": "replan",
        "advisory_mode": "diagnose",
        "evidence_needs": ["bq", "news"],
        "geography": ["Nigeria", "Ghana"],
    }
    with mock.patch(
        "ml.rag.chatbot.metacog.llm_chat_complete",
        return_value=json.dumps(payload),
    ):
        plan = MetacogPlan(advisory_mode="diagnose", confidence=0.9, source="heuristic")
        verified = MetacogVerify(status="gap", failed_checks=("geo_match",))
        decision = propose_metacog_revise(
            plan, verified, query="", already_revised=0, bq_results=[], reranked_context=[]
        )
    assert decision.action == "replan"
    assert decision.advisory_mode == "diagnose"
    assert list(decision.evidence_needs) == ["bq", "news"]
    assert list(decision.geography) == ["Nigeria", "Ghana"]


def test_apply_revise_decision_skip() -> None:
    turn = TurnContract(job="fact")
    plan = MetacogPlan(advisory_mode="diagnose", confidence=0.9, source="heuristic")
    decision = MetacogReviseDecision(action="skip")
    out_turn, out_plan, counter = apply_revise_decision(turn, plan, decision)
    assert out_turn.job == "fact"
    assert out_plan.advisory_mode == "diagnose"
    assert counter == 0


def test_apply_revise_decision_replan() -> None:
    turn = TurnContract(job="fact", measure_id="production")
    plan = MetacogPlan(
        advisory_mode="diagnose",
        confidence=0.9,
        evidence_needs=("bq", "news"),
        claim_budget="causal_careful",
        source="llm",
    )
    decision = MetacogReviseDecision(action="replan", advisory_mode="diagnose", evidence_needs=("bq",), geography=())
    out_turn, out_plan, counter = apply_revise_decision(turn, plan, decision)
    assert out_turn.job == "diagnose"
    assert out_plan.advisory_mode == "diagnose"
    assert counter == 1


def test_self_check_applies_diagnose() -> None:
    plan = MetacogPlan(advisory_mode="diagnose", confidence=0.9, source="heuristic")
    verified = MetacogVerify(status="strong", evidence_tier_hint="strong")
    assert self_check_applies(plan, verified, job="diagnose")


def test_self_check_applies_partial_verify() -> None:
    plan = MetacogPlan(advisory_mode="fact", confidence=0.9, source="heuristic")
    verified = MetacogVerify(status="partial", evidence_tier_hint="weak")
    assert self_check_applies(plan, verified, job="fact")


def test_self_check_does_not_apply_ok_verify() -> None:
    plan = MetacogPlan(advisory_mode="diagnose", confidence=0.9, source="heuristic")
    verified = MetacogVerify(status="ok", evidence_tier_hint="strong")
    assert not self_check_applies(plan, verified, job="diagnose")


def test_self_check_keep_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_METACOG", "1")
    monkeypatch.setenv("RAG_METACOG_SELF_CHECK", "1")
    with mock.patch(
        "ml.rag.chatbot.metacog.llm_chat_complete",
        return_value="not json",
    ):
        check = self_check_draft("Draft answer here", query="", evidence="Evidence here")
    assert check.verdict == "keep"


def test_self_check_keep_no_revise() -> None:
    payload = {"verdict": "keep"}
    with mock.patch(
        "ml.rag.chatbot.metacog.llm_chat_complete",
        return_value=json.dumps(payload),
    ):
        check = self_check_draft("Draft answer", query="", evidence="Evidence")
    assert check.verdict == "keep"


def test_self_check_revise() -> None:
    payload = {"verdict": "revise", "problems": ["wrong country", "invented number"]}
    with mock.patch(
        "ml.rag.chatbot.metacog.llm_chat_complete",
        return_value=json.dumps(payload),
    ):
        check = self_check_draft("Draft answer", query="", evidence="Evidence")
    assert check.verdict == "revise"
    assert list(check.problems) == ["wrong country", "invented number"]


def test_self_check_applies_fact_with_partial_verify() -> None:
    plan = MetacogPlan(advisory_mode="fact", confidence=0.9, source="heuristic")
    verified = MetacogVerify(status="partial", evidence_tier_hint="weak")
    assert self_check_applies(plan, verified, job="fact")
