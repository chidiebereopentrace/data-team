"""Property tests for typed turn contract invariants."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml.rag.chatbot.agri_measure_ontology import MEASURES, MeasureHit, resolve_measure
from ml.rag.chatbot.capability_registry import resolve_capability
from ml.rag.chatbot.corpus_catalog import select_corpora
from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.generation_plan import build_generation_plan
from ml.rag.chatbot.ontology_context import sanitize_decomposition_for_bq
from ml.rag.chatbot.query_decomposer import decompose_query
from ml.rag.chatbot.task_mode import resolve_task_mode
from ml.rag.chatbot.time_retrieval import time_kwargs_from_contract
from ml.rag.chatbot.typed_pack import typed_context_pack
from ml.rag.chatbot.turn_contract import TimeSpec, TurnContract
from ml.rag.tests.chatbot.fixtures.synthetic_facets import facet_to_query, synthetic_numeric_facets

_GOLD_PATH = Path(__file__).resolve().parent / "fixtures" / "gold_traces.yaml"


def test_production_served_companion_corpora() -> None:
    contract = resolve_capability(
        TurnContract(measure_id="production", geo_grain="country", job="fact", serve_status="served")
    )
    sel = select_corpora(
        {},
        query="rice production Kenya",
        contract_job=contract.job,
        vector_allow=contract.vector_allow,
        vector_block=contract.vector_block,
        vector_policy=contract.vector_policy,
    )
    assert contract.vector_policy == "companion"
    assert sel.active
    assert "policies" not in sel.active


def test_employment_served_academic_only() -> None:
    contract = resolve_capability(
        TurnContract(
            measure_id="employment_share",
            geo_grain="country",
            job="fact",
            time_spec=TimeSpec(grain="year"),
        )
    )
    sel = select_corpora(
        {},
        contract_job=contract.job,
        vector_allow=contract.vector_allow,
        vector_block=contract.vector_block,
        vector_policy=contract.vector_policy,
    )
    assert contract.vector_policy == "companion"
    assert set(sel.active) <= {"academic_papers"}
    assert "policies" not in sel.active


def test_employment_unsupported_grain_keeps_vector() -> None:
    contract = resolve_capability(
        TurnContract(measure_id="employment_share", geo_grain="admin2", job="list")
    )
    assert contract.skip_bq is True
    assert contract.skip_vector_retrieval is False
    assert contract.should_retrieve_vector()
    items = [{"content": "Academic essay", "metadata": {"context_kind": "academic"}}]
    packed = typed_context_pack(items, contract).items
    assert any("Academic essay" in str(i.get("content") or "") for i in packed)


def test_companion_pack_includes_narrative_when_bq_present() -> None:
    contract = resolve_capability(
        TurnContract(measure_id="production", geo_grain="country", job="fact")
    )
    items = [
        {"content": "[Structured data] 1000 tonnes", "metadata": {"context_kind": "bigquery", "status": "ok"}},
        {"content": "Drought affected yields", "metadata": {"context_kind": "news"}},
    ]
    packed = typed_context_pack(items, contract).items
    assert any("[Structured data]" in str(i.get("content") or "") for i in packed)
    assert any("Drought" in str(i.get("content") or "") for i in packed)


def test_help_social_skip_retrieval_via_contract_job() -> None:
    for job in ("help", "social"):
        sel = select_corpora({}, contract_job=job, vector_policy="none")
        assert sel.active == []


def test_unsupported_grain_keeps_documents() -> None:
    contract = TurnContract(
        measure_id="employment_share",
        geo_grain="admin2",
        serve_status="unsupported_grain",
        job="list",
        vector_policy="none",
    )
    items = [{"content": "District employment survey", "metadata": {"context_kind": "academic"}}]
    assert typed_context_pack(items, contract).items == items


def test_no_valid_sql_numeric_zero_pack() -> None:
    contract = TurnContract(
        measure_id="production",
        job="fact",
        serve_status="served",
        vector_policy="companion",
    )
    items = [
        {"content": "News about rice", "metadata": {"context_kind": "news"}},
        {"content": "[Structured data] 1000", "metadata": {"context_kind": "bigquery", "status": "ok"}},
    ]
    packed = typed_context_pack(items, contract).items
    assert len(packed) >= 1
    assert any("News about rice" in it.get("content", "") for it in packed)


def test_bq_timeout_keeps_documents() -> None:
    contract = TurnContract(measure_id="production", job="fact", serve_status="served")
    items = [
        {"content": "timeout", "metadata": {"context_kind": "bigquery", "status": "bq_timeout"}},
        {"content": "Maize harvest fell after drought", "metadata": {"context_kind": "news"}},
    ]
    pack = typed_context_pack(items, contract)
    assert [i["content"] for i in pack.items] == ["Maize harvest fell after drought"]
    assert sum(pack.dropped.values()) == 1


def test_document_budget_caps_documents() -> None:
    contract = TurnContract(measure_id="production", job="fact", serve_status="served")
    items = [{"content": f"report {n}", "metadata": {"context_kind": "news"}} for n in range(10)]
    pack = typed_context_pack(items, contract)
    assert len(pack.items) == 6
    assert pack.dropped == {"document_budget": 4}


def test_relevance_floor_drops_weak_documents() -> None:
    contract = TurnContract(measure_id="production", job="fact", serve_status="served")
    items = [
        {"content": "on topic maize harvest", "score": 0.8, "metadata": {"context_kind": "news"}},
        {"content": "unrelated page", "score": 0.01, "metadata": {"context_kind": "web"}},
    ]
    pack = typed_context_pack(items, contract)
    assert [i["content"] for i in pack.items] == ["on topic maize harvest"]
    assert pack.dropped.get("relevance_floor") == 1


def test_disease_prevalence_fallback_only() -> None:
    query = "What is the prevalence of east coast fever in smallholder dairy herds in Tanzania?"
    dec = decompose_query(query)
    hit = resolve_measure(query, dec)
    contract = resolve_capability(
        compile_turn_contract(query, dec, measure_hit=hit, task_mode_hint="research")
    )
    assert contract.measure_id == "disease_prevalence"
    assert contract.serve_status == "unsupported_grain"
    assert contract.vector_policy == "fallback_only"
    assert set(contract.vector_allow) <= {"academic_papers", "public_reports"}
    sel = select_corpora(
        {},
        vector_allow=contract.vector_allow,
        vector_block=contract.vector_block,
        vector_policy=contract.vector_policy,
    )
    assert "policies" not in sel.active


def test_hard_filter_time_kwargs() -> None:
    contract = TurnContract(
        measure_id="market_price",
        geo_grain="country",
        job="compare",
        time_spec=TimeSpec(start="2024-01-01", end="2024-12-31", hard_filter=True),
    )
    kw = time_kwargs_from_contract(contract)
    assert kw["hard_filter"] is True
    assert kw["published_at_from"] == "2024-01-01"


def test_out_of_window_documents_rank_last() -> None:
    contract = TurnContract(
        measure_id="production",
        job="fact",
        serve_status="served",
        vector_policy="companion",
        time_spec=TimeSpec(start="2024-01-01", end="2024-12-31", hard_filter=True),
    )
    items = [
        {"content": "Old report from 2006", "metadata": {"context_kind": "news", "year": "2006"}},
        {"content": "Undated drought brief", "metadata": {"context_kind": "news"}},
        {"content": "Recent 2024 update", "metadata": {"context_kind": "news", "year": "2024"}},
    ]
    packed = typed_context_pack(items, contract).items
    assert [i["content"] for i in packed] == [
        "Recent 2024 update",
        "Undated drought brief",
        "Old report from 2006",
    ]


@pytest.mark.parametrize("facet", list(synthetic_numeric_facets(limit=50)))
def test_compile_produces_measure_job_geo(facet: dict) -> None:
    query = facet_to_query(facet)
    dec = decompose_query(query)
    hit = resolve_measure(query, dec)
    task_mode = resolve_task_mode(query, dec)
    contract = compile_turn_contract(
        query, dec, measure_hit=hit, task_mode_hint=task_mode,
    )
    contract = resolve_capability(contract)
    assert contract.measure_id or contract.serve_status in ("clarify", "unsupported_measure")
    assert contract.job in (
        "fact", "trend", "rank", "compare", "list", "outlook", "diagnose", "brief", "clarify",
    )
    if facet["geo_grain"] == "admin2" and facet["measure_id"] in ("employment_share", "rainfall"):
        assert contract.serve_status == "unsupported_grain"


def test_employment_sex_breakdown_compiles() -> None:
    query = "What share of employment in agriculture by men and women in Uganda?"
    dec = decompose_query(query)
    hit = resolve_measure(query, dec)
    contract = compile_turn_contract(query, dec, measure_hit=hit, task_mode_hint="fact_lookup")
    assert contract.measure_id == "employment_share"
    assert "sex" in contract.breakdown
    contract = resolve_capability(contract)
    assert contract.serve_status == "served"
    assert contract.sql_plan.get("template") == "employment_share_by_sex"
    assert contract.vector_policy == "companion"


def test_yam_africa_ngo_declared_ipc_locks_production() -> None:
    query = "how many countries produce yam in africa"
    dec = {
        "entities": ["food_security_ipc", "yam", "Africa", "nutrition"],
        "domains": ["nutrition", "markets"],
        "primary_measures": ["food_security_ipc"],
        "geo_scope": "continent",
        "africa_panel": True,
        "job": "list",
    }
    ipc_hit = MeasureHit(MEASURES["food_security_ipc"], score=20, matched_alias="food_security_ipc")
    cleaned = sanitize_decomposition_for_bq(dec, primary_measures=["food_security_ipc"], query=query)
    assert cleaned["primary_measures"][0] == "production"
    assert "food_security_ipc" not in {str(e).lower() for e in (cleaned.get("entities") or [])}
    turn = resolve_capability(
        compile_turn_contract(query, cleaned, measure_hit=ipc_hit, task_mode_hint="fact_lookup")
    )
    assert turn.measure_id == "production"
    assert turn.serve_status == "served"
    assert turn.skip_vector_retrieval is False


def test_unsupported_measure_keeps_narrative_for_generate() -> None:
    contract = TurnContract(
        measure_id="food_security_ipc",
        geo_grain="africa",
        job="list",
        serve_status="unsupported_measure",
        skip_bq=True,
        skip_vector_retrieval=False,
        vector_policy="companion",
    )
    items = [
        {
            "content": "Yam is grown in West and Central Africa.",
            "metadata": {"context_kind": "academic"},
        }
    ]
    assert contract.should_retrieve_vector()
    packed = typed_context_pack(items, contract).items
    assert packed
    plan = build_generation_plan(
        "how many countries produce yam in africa",
        reranked_context=packed,
        turn_contract=contract.to_dict(),
        task_mode="fact_lookup",
    )
    assert plan.answer_shape != "gap_ack"


def test_producer_prices_not_production() -> None:
    query = "producer prices of maize in Kenya"
    dec = {"entities": ["maize", "Kenya"], "primary_measures": ["market_price"]}
    turn = compile_turn_contract(query, dec, task_mode_hint="fact_lookup")
    assert turn.measure_id == "market_price"


def test_should_retrieve_vector_numeric_policy_none() -> None:
    contract = TurnContract(
        measure_id="production",
        job="fact",
        serve_status="served",
        vector_policy="none",
    )
    assert contract.should_retrieve_vector() is True


def test_gold_traces_smoke() -> None:
    if not _GOLD_PATH.is_file():
        pytest.skip("gold_traces.yaml missing")
    data = yaml.safe_load(_GOLD_PATH.read_text(encoding="utf-8")) or {}
    traces = data.get("traces") or []
    assert len(traces) >= 5
    for trace in traces:
        query = str(trace.get("query") or "")
        dec = decompose_query(query)
        hit = resolve_measure(query, dec)
        task_mode = resolve_task_mode(query, dec)
        contract = resolve_capability(
            compile_turn_contract(query, dec, measure_hit=hit, task_mode_hint=task_mode)
        )
        if trace.get("expect_measure"):
            assert contract.measure_id == trace["expect_measure"]
        if trace.get("expect_job"):
            assert contract.job == trace["expect_job"]
        if trace.get("expect_serve_status"):
            assert contract.serve_status == trace["expect_serve_status"]
        if trace.get("expect_template"):
            assert contract.sql_plan.get("template") == trace["expect_template"]
        if trace.get("expect_vector_policy"):
            assert contract.vector_policy == trace["expect_vector_policy"]
        if trace.get("expect_output_type"):
            from ml.rag.chatbot.output_format import output_type_from_contract

            assert output_type_from_contract(contract) == trace["expect_output_type"]
