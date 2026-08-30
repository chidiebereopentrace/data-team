"""Property tests for typed turn contract invariants."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml.rag.chatbot.agri_measure_ontology import resolve_measure
from ml.rag.chatbot.capability_registry import resolve_capability
from ml.rag.chatbot.corpus_catalog import select_corpora
from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.query_decomposer import decompose_query
from ml.rag.chatbot.task_mode import resolve_task_mode
from ml.rag.chatbot.time_retrieval import time_kwargs_from_contract
from ml.rag.chatbot.typed_pack import should_zero_pack, typed_context_pack
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


def test_employment_unsupported_grain_zero_vector() -> None:
    contract = resolve_capability(
        TurnContract(measure_id="employment_share", geo_grain="admin2", job="list")
    )
    assert contract.vector_policy == "none"
    assert not contract.should_retrieve_vector()
    items = [{"content": "Policy essay", "metadata": {"context_kind": "policy"}}]
    assert typed_context_pack(items, contract) == []


def test_companion_pack_includes_narrative_when_bq_present() -> None:
    contract = resolve_capability(
        TurnContract(measure_id="production", geo_grain="country", job="fact")
    )
    items = [
        {"content": "[Structured data] 1000 tonnes", "metadata": {"context_kind": "bigquery", "status": "ok"}},
        {"content": "Drought affected yields", "metadata": {"context_kind": "news"}},
    ]
    packed = typed_context_pack(items, contract, bq_failed=False)
    assert any("[Structured data]" in str(i.get("content") or "") for i in packed)
    assert any("Drought" in str(i.get("content") or "") for i in packed)


def test_help_social_skip_retrieval_via_contract_job() -> None:
    for job in ("help", "social"):
        sel = select_corpora({}, contract_job=job, vector_policy="none")
        assert sel.active == []


def test_unsupported_grain_zero_pack() -> None:
    contract = TurnContract(
        measure_id="employment_share",
        geo_grain="admin2",
        serve_status="unsupported_grain",
        job="list",
        vector_policy="none",
    )
    items = [{"content": "[Structured data] x", "metadata": {"context_kind": "bigquery"}}]
    assert should_zero_pack(contract)
    assert typed_context_pack(items, contract) == []


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
    packed = typed_context_pack(items, contract, bq_failed=True)
    assert len(packed) >= 1
    assert any("News about rice" in it.get("content", "") for it in packed)


def test_bq_timeout_zero_pack() -> None:
    contract = TurnContract(measure_id="production", job="fact", serve_status="served")
    items = [
        {
            "content": "timeout",
            "metadata": {"context_kind": "bigquery", "status": "bq_timeout"},
        }
    ]
    assert should_zero_pack(contract, context_items=items)
    assert typed_context_pack(items, contract) == []


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


def test_hard_filter_drops_out_of_window_vector() -> None:
    contract = TurnContract(
        measure_id="production",
        job="fact",
        serve_status="served",
        vector_policy="companion",
        time_spec=TimeSpec(start="2024-01-01", end="2024-12-31", hard_filter=True),
    )
    items = [
        {"content": "[Structured data] 100", "metadata": {"context_kind": "bigquery"}},
        {"content": "Old report from 2006", "metadata": {"context_kind": "news", "year": "2006"}},
        {"content": "Recent 2024 update", "metadata": {"context_kind": "news", "year": "2024"}},
    ]
    packed = typed_context_pack(items, contract, bq_failed=False)
    assert any("2024" in str(i.get("content") or "") for i in packed)
    assert not any("2006" in str(i.get("content") or "") for i in packed)


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
