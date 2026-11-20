"""Captured QA failures: source selection must survive planning and execution."""
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from ml.rag.chatbot.document_evidence import requirement_document_candidates
from ml.rag.chatbot.evidence_requirements import fact_requirement_coverage
from ml.rag.chatbot.facet_compiler import compile_geo_grain, compile_turn_contract
from ml.rag.chatbot.graph import (
    node_bq_reason,
    node_bq_retrieve,
    node_decompose,
    node_generate,
    node_merge,
    node_rerank,
)
from ml.rag.chatbot.turn_contract import TurnContract
from ml.rag.chatbot.typed_pack import typed_context_pack
from ml.rag.retrievers.bq_retriever import BQRetriever

CASES = [
    (
        ("Assess how rainfall variability relates to maize production and yields in Zambia and Kenya "
         "from 2015 to 2024. Separate changes in harvested area from changes in yield."),
        {"job": "diagnose", "intent": "diagnostic", "entities": ["maize"],
         "geography": ["Zambia", "Kenya"], "geo_scope": "multi_country",
         "primary_measure_hints": ["climate", "production", "yield"]},
        {"agg_production_country_year"},
    ),
    (
        ("Examine whether agricultural productivity improvements in Nigeria, Ghana and Uganda "
         "between 2015 and 2024 coincided with changes in agricultural employment. "
         "Distinguish agriculture's share of total employment from women's share of agricultural employment."),
        {"job": "report", "intent": "descriptive", "entities": ["agriculture", "livelihoods"],
         "geography": ["Nigeria", "Ghana", "Uganda"], "geo_scope": "multi_country",
         "primary_measure_hints": ["employment_share", "yield"]},
        {"fct_employment"},
    ),
]


@pytest.mark.parametrize("query,facets,expected_tables", CASES)
def test_selected_warehouse_work_reaches_execution(monkeypatch, query, facets, expected_tables):
    facets = {**facets, "reference_date": "2026-11-18",
              "time_start": "2015-01-01", "time_end": "2024-12-31"}
    monkeypatch.setenv("RAG_SLOT_REASONER", "off")
    monkeypatch.setattr("ml.rag.chatbot.graph.metacog_enabled", lambda: False)
    monkeypatch.setattr("ml.rag.chatbot.graph.decompose_query", lambda *a, **kw: deepcopy(facets))
    state = {"query": query, **node_decompose({"query": query, "reference_date": "2026-11-18"})}
    assert state["source_selection"]["bigquery"]["selected"]
    state.update(node_bq_reason(state))
    assert not state["bq_sql_plan"].get("skip_bq"), state["turn_contract"]

    client = MagicMock()
    client.query.return_value.result.return_value = []
    monkeypatch.setenv("BQ_PROJECT", "test")
    monkeypatch.setattr(BQRetriever, "_get_client", lambda self: client)
    monkeypatch.setattr("ml.rag.retrievers.bq_retriever.dry_run_sql", lambda *a, **kw: None)
    result = node_bq_retrieve(state)
    sql = "\n".join(call.args[0] for call in client.query.call_args_list)
    assert client.query.called, result.get("bq_sql_debug")
    assert all(table in sql for table in expected_tables), sql


@pytest.mark.parametrize("scope", ["multi_country", ""])
def test_individual_countries_are_country_grain(scope):
    assert compile_geo_grain("Analyse Kenya and Zambia", {
        "geo_scope": scope, "geography": ["Kenya", "Zambia"],
    }) == "country"


def test_independent_indicator_scopes_survive_binding_and_coverage():
    query = (
        "Compare maize yields in Kenya and Uganda between 2018 and 2023. "
        "Separately, analyse agriculture's share of employment in Ghana from 2015 to 2024. "
        "Calculate yield changes in tonnes per hectare and employment changes in percentage points."
    )
    facets = {"job": "compare", "geo_scope": "multi_country", "entities": ["maize"],
              "geography": ["Kenya", "Uganda", "Ghana"], "reference_date": "2026-11-18"}
    turn = compile_turn_contract(query, facets)
    assert TurnContract.from_dict(turn.to_dict()) == turn
    yield_fact = next(fact for fact in turn.facts if fact.measure_id == "yield")
    employment = next(fact for fact in turn.facts if fact.measure_id == "employment_share")
    assert yield_fact.geo == ["Kenya", "Uganda"]
    assert employment.geo == ["Ghana"]
    assert (employment.time_spec.start, employment.time_spec.end) == ("2015-01-01", "2024-12-31")
    state = {"query": query, "decomposition": facets, "turn_contract": turn.to_dict()}
    plan = node_bq_reason(state)["bq_sql_plan"]
    assert set(plan["fact_tables"]) == {"yield", "employment_share"}
    assert plan["bind_contracts"]["yield"]["geo_literals"] == ["KEN", "UGA"]
    assert "GHA" in plan["bind_contracts"]["employment_share"]["geo_literals"]
    coverage = fact_requirement_coverage(turn, plan, [], [])
    assert {entry["geo"] for entry in coverage if entry["measure"] == "employment_share"} == {"Ghana"}
    assert all(entry["status"] == "not_queried" for entry in coverage)


def test_same_table_does_not_overwrite_different_requirements(monkeypatch):
    query = "Compare maize yields and maize production in Kenya and Uganda from 2018 to 2023."
    facets = {"job": "compare", "geography": ["Kenya", "Uganda"],
              "geo_scope": "multi_country", "entities": ["maize"], "reference_date": "2026-11-18"}
    turn = compile_turn_contract(query, facets)
    state = {"query": query, "decomposition": facets, "turn_contract": turn.to_dict()}
    state.update(node_bq_reason(state))
    assert state["bq_sql_plan"]["fact_tables"] == {
        "yield": "agg_production_country_year", "production": "agg_production_country_year",
    }
    client = MagicMock()
    client.query.return_value.result.return_value = []
    monkeypatch.setenv("BQ_PROJECT", "test")
    monkeypatch.setattr(BQRetriever, "_get_client", lambda self: client)
    monkeypatch.setattr("ml.rag.retrievers.bq_retriever.dry_run_sql", lambda *a, **kw: None)
    result = node_bq_retrieve(state)
    queries = [call.args[0] for call in client.query.call_args_list]
    assert len(queries) == 2, result.get("bq_sql_debug")
    assert any("yield_recomputed AS value" in sql for sql in queries)
    assert any("production_qty AS value" in sql for sql in queries)
    assert {row["requirement_id"] for row in result["bq_sql_debug"] if row.get("requirement_id")} == {"yield", "production"}


def test_other_report_cannot_answer_what_a_named_policy_says():
    query = "What does Kenya's National Climate Change Action Plan 2023–2027 say about agricultural adaptation?"
    turn = compile_turn_contract(query, {"geography": ["Kenya"], "evidence_topics": ["agricultural adaptation"]})
    passage = "Kenya's National Climate Change Action Plan 2023–2027 supports agricultural adaptation."
    items = [
        {"source": "policy", "content": passage, "metadata": {"title": title, "country_name": "Kenya"}}
        for title in [
            "RECIPES FOR SUCCESS 2: Policy Innovations to Achieve the Kampala Declaration Goals (2025)",
            "National Climate Change Action Plan 2018–2022",
            "National Climate Change Action Plan 2023–2027",
        ]
    ]
    accepted = requirement_document_candidates(turn, items)
    assert [item["metadata"]["title"] for item in accepted] == ["National Climate Change Action Plan 2023–2027"]


def test_population_context_does_not_substitute_hdi_for_food_security():
    query = "Compare food insecurity population counts and maize production in Kenya from 2019 to 2025."
    facets = {"job": "compare", "geography": ["Kenya"], "entities": ["maize"], "reference_date": "2026-11-18"}
    turn = compile_turn_contract(query, facets)
    plan = node_bq_reason({"query": query, "turn_contract": turn.to_dict(), "decomposition": facets})["bq_sql_plan"]
    assert "fct_hdi" not in plan["selected_tables"]
    assert plan["fact_tables"]["production"] == "agg_production_country_year"


@pytest.mark.parametrize("separator", [" ", "-", "‑"])
def test_broad_food_report_keeps_warehouse_search_with_different_punctuation(monkeypatch, separator):
    query = f"Prepare a comparative report on food{separator}price changes in Nigeria and Cameroon from 2020 to 2025."
    facets = {"job": "report", "entities": [], "geography": ["Nigeria", "Cameroon"],
              "geo_scope": "multi_country", "reference_date": "2026-11-18"}
    monkeypatch.setattr("ml.rag.chatbot.graph.decompose_query", lambda *a, **kw: deepcopy(facets))
    state = node_decompose({"query": query, "reference_date": "2026-11-18"})
    assert state["source_selection"]["bigquery"]["selected"]
    state = {"query": query, **state}
    state.update(node_bq_reason(state))
    assert not state["bq_sql_plan"].get("skip_bq")
    client = MagicMock()
    client.query.return_value.result.return_value = []
    monkeypatch.setenv("BQ_PROJECT", "test")
    monkeypatch.setattr(BQRetriever, "_get_client", lambda self: client)
    monkeypatch.setattr("ml.rag.retrievers.bq_retriever.dry_run_sql", lambda *a, **kw: None)
    result = node_bq_retrieve(state)
    assert client.query.called, result.get("bq_sql_debug")


def test_repeated_indicator_executes_each_local_scope_without_cross_contamination(monkeypatch):
    query = ("Compare maize yields in Kenya from 2018 to 2020. "
             "Separately compare maize yields in Ghana from 2022 to 2024.")
    facets = {"job": "compare", "geography": ["Kenya", "Ghana"], "entities": ["maize"],
              "geo_scope": "multi_country", "reference_date": "2026-11-18"}
    turn = compile_turn_contract(query, facets)
    state = {"query": query, "decomposition": facets, "turn_contract": turn.to_dict()}
    state.update(node_bq_reason(state))
    assert set(state["bq_sql_plan"]["fact_tables"]) == {"yield_1", "yield_2"}
    client = MagicMock()
    client.query.return_value.result.return_value = []
    monkeypatch.setenv("BQ_PROJECT", "test")
    monkeypatch.setattr(BQRetriever, "_get_client", lambda self: client)
    monkeypatch.setattr("ml.rag.retrievers.bq_retriever.dry_run_sql", lambda *a, **kw: None)
    result = node_bq_retrieve(state)
    sql = {row["requirement_id"]: row["sql"] for row in result["bq_sql_debug"] if row.get("requirement_id")}
    assert set(sql) == {"yield_1", "yield_2"}
    assert "KEN" in sql["yield_1"] and "GHA" not in sql["yield_1"]
    assert "2018" in sql["yield_1"] and "2024" not in sql["yield_1"]
    assert "GHA" in sql["yield_2"] and "KEN" not in sql["yield_2"]
    assert "2022" in sql["yield_2"] and "2018" not in sql["yield_2"]


def test_passage_top_k_cannot_drop_bound_observations():
    query = "Compare maize yields in Kenya and Uganda from 2018 to 2023."
    turn = compile_turn_contract(query, {"job": "compare", "geography": ["Kenya", "Uganda"], "entities": ["maize"]})
    rows = [
        {"source": "bigquery", "content": f"Maize yield in {country} in {year}: 2 t/ha.",
         "metadata": {"status": "ok", "requirement_id": "yield", "table_id": "agg_production_country_year",
                      "country_name": country, "year": year, "raw_row": {"value": 2, "year": year}}}
        for country in ["Kenya", "Uganda"] for year in range(2018, 2024)
    ]
    packed = typed_context_pack(rows, turn, top_k=4)
    assert len(packed.items) == 12
    assert not packed.dropped.get("over_top_k")


def test_complete_pipeline_keeps_local_period_and_renders_facts_without_model(monkeypatch):
    query = ("Compare maize yields in Kenya and Uganda between 2018 and 2023. "
             "Separately, analyse agriculture's share of employment in Ghana from 2015 to 2024. "
             "Use research to discuss possible explanations.")
    facets = {"job": "compare", "geography": ["Kenya", "Uganda", "Ghana"], "entities": ["maize"],
              "geo_scope": "multi_country", "reference_date": "2026-11-18"}
    turn = compile_turn_contract(query, facets)
    assert turn.documents
    state = {"query": query, "task_mode": "analytical", "decomposition": facets,
             "turn_contract": turn.to_dict(), "rerank_top_k": 4}
    state.update(node_bq_reason(state))

    def execute(sql, **kwargs):
        if "fct_employment" in sql:
            rows = [{"country_iso3": "GHA", "year": year, "value": 35 + (year - 2015), "unit": "%",
                     "indicator": "Share of employment in agriculture in total employment - ILO Modelled Estimates",
                     "sex": "Total", "organisation_name": "ILOSTAT"} for year in range(2015, 2024)]
        else:
            rows = [{"country_iso3": country, "year": year, "value": 1.5 + (year - 2018) / 10,
                     "product_name": "Maize (corn)", "organisation_name": "FAOSTAT"}
                    for country in ["KEN", "UGA"] for year in range(2018, 2024)]
        job = MagicMock()
        job.result.return_value = rows
        return job

    client = MagicMock()
    client.query.side_effect = execute
    monkeypatch.setenv("BQ_PROJECT", "test")
    monkeypatch.setattr(BQRetriever, "_get_client", lambda self: client)
    monkeypatch.setattr("ml.rag.retrievers.bq_retriever.dry_run_sql", lambda *a, **kw: None)
    model = MagicMock(return_value=None)
    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", model)
    state.update(node_bq_retrieve(state))
    assert len(state["bq_results"]) == 21
    state.update(node_merge(state))
    state.update(node_rerank(state))
    assert len(state["reranked_context"]) == 21
    result = node_generate(state)
    employment = [row for row in result["fact_requirement_coverage"] if row["measure"] == "employment_share"]
    assert employment[0]["observed_years"] == list(range(2015, 2024))
    assert employment[0]["missing_years"] == [2024]
    assert "2015: 35.00 %" in result["answer"] and "2023: 43.00 %" in result["answer"]
    assert "+8.00 percentage points" in result["answer"]
    assert "2018: 1.50 t/ha" in result["answer"] and "+0.50 t/ha" in result["answer"]
    assert "What this value is" not in result["answer"]
    assert len(result["citations"]) == 21
    model.assert_not_called()
