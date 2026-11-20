from datetime import date
from unittest.mock import MagicMock

import pytest

from ml.rag.chatbot.compile_sql_from_bind import compile_operation
from ml.rag.chatbot.document_evidence import document_requirement_coverage
from ml.rag.chatbot.document_evidence import requirement_document_candidates
from ml.rag.chatbot.evidence_requirements import fact_requirement_coverage, matches_bound_fact
from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.graph import (
    _document_search_texts,
    _retrieve_academic_papers,
    node_bq_reason,
    node_bq_retrieve,
    node_coverage_retry,
    node_decompose,
    node_generate,
    node_merge,
    node_parallel_retrieve,
    node_rerank,
)
from ml.rag.chatbot.turn_contract import TurnContract
from ml.rag.retrievers.bq_retriever import BQRetriever


QUESTION = (
    "Compare drought, maize yields, and food insecurity in Kenya in 2018–2019 "
    "versus 2022–2023. What do assessments say about the relationship?"
)
FACETS = {
    "entities": ["maize"],
    "geography": ["Kenya"],
    "geo_scope": "country",
    "job": "compare",
    "reference_date": "2025-01-01",
    "domains": ["Agricultural Food Systems & Value Chain"],
    "evidence_topics": ["Assessments of drought effects on maize yields and food insecurity in Kenya"],
}


@pytest.mark.parametrize("job", ["trend", "report", "fact"])
def test_food_price_report_retrieves_without_a_named_commodity(monkeypatch, job):
    query = "give me a robust detailed aanlytics of food price changes in nigeria and cameroun since 2020 till now"
    facets = {
        "intent": "descriptive", "job": job, "entities": [],
        "geography": ["Nigeria", "Cameroon"], "primary_measures": ["market_price"],
        "time_start": "2020-01-01", "time_end": "2026-11-18", "reference_date": "2026-11-18",
    }
    monkeypatch.setattr("ml.rag.chatbot.graph.decompose_query", lambda *a, **kw: dict(facets))
    state = node_decompose({"query": query, "reference_date": "2026-11-18"})
    assert state["task_mode"] == "analytical", state.get("turn_contract")
    assert state["source_selection"]["bigquery"]["selected"]
    assert state["source_selection"]["qdrant"]["selected"]
    planned = node_bq_reason({"query": query, **state})
    plan = planned["bq_sql_plan"]
    assert set(plan["selected_tables"]) <= {"fct_prices", "agg_prices_country_month"}, plan
    client = MagicMock()
    client.query.return_value.result.return_value = []
    monkeypatch.setenv("BQ_PROJECT", "test")
    monkeypatch.setattr(BQRetriever, "_get_client", lambda self: client)
    monkeypatch.setattr("ml.rag.retrievers.bq_retriever.dry_run_sql", lambda *a, **kw: None)
    retrieved = node_bq_retrieve({"query": query, **state, **planned})
    sql = " ".join(call.args[0] for call in client.query.call_args_list)
    assert client.query.call_count >= 1, retrieved.get("bq_sql_debug")
    assert "NGA" in sql and "CMR" in sql
    assert "2020" in sql and "2026" in sql


def test_explicit_requirements_survive_language_enrichment_and_serialization():
    original = "Comparez la sécheresse, le rendement du maïs et l’insécurité alimentaire au Kenya."
    contract = compile_turn_contract(QUESTION, FACETS, original_question=original, answer_lang="fr")
    restored = TurnContract.from_dict(contract.to_dict())
    assert restored == contract
    assert restored.original_question == original
    assert restored.answer_lang == "fr"
    assert restored.geo == ["Kenya"]
    assert restored.job == "compare"
    assert {fact.measure_id for fact in restored.facts} == {"climate", "yield", "food_security_ipc"}
    assert [(w.start.isoformat(), w.end.isoformat()) for w in restored.time_spec.windows] == [
        ("2018-01-01", "2019-12-31"), ("2022-01-01", "2023-12-31"),
    ]


def test_investment_comparison_keeps_both_indicators_and_focused_policy_evidence():
    query = (
        "Compare maize yields and agriculture’s share of total employment in Kenya and Uganda in 2018 and 2023. "
        "Using cited policy evaluations or research, assess whether mechanisation or farm input policies "
        "help explain the changes. Distinguish measured policy effects from plausible explanations."
    )
    facets = {**FACETS, "geography": ["Kenya", "Uganda"], "evidence_topics": []}
    turn = compile_turn_contract(query, facets)
    assert [(w.start.year, w.end.year) for w in turn.time_spec.windows] == [(2018, 2018), (2023, 2023)]
    planned = node_bq_reason({"query": query, "decomposition": facets, "turn_contract": turn.to_dict()})["bq_sql_plan"]
    assert planned["fact_tables"] == {"yield": "agg_production_country_year", "employment_share": "fct_employment"}
    assert all(" OR " in bind["time_sql"] for bind in planned["bind_contracts"].values())

    policy = "A 2020 evaluation in Kenya found that mechanisation improved maize yields."
    unrelated = "In Kenya, a 2020 agricultural investment committee discussed plausible explanations."
    candidates = requirement_document_candidates(turn, [
        {"source": "policy", "content": policy}, {"source": "policy", "content": unrelated},
    ])
    assert [item["content"] for item in candidates] == [policy]
    assert candidates[0]["metadata"]["document_requirements"]


def test_comparative_answer_keeps_verified_endpoints_when_model_omits_them(monkeypatch):
    query = ("Compare maize yields and agricultural employment share in Kenya and Uganda in 2018 and 2023. "
             "Using research, explain whether mechanisation affected the changes.")
    facets = {**FACETS, "geography": ["Kenya", "Uganda"], "evidence_topics": []}
    turn = compile_turn_contract(query, facets)
    planned = node_bq_reason({"query": query, "decomposition": facets, "turn_contract": turn.to_dict()})["bq_sql_plan"]
    rows = []
    for measure, table in planned["fact_tables"].items():
        for country, iso in (("Kenya", "KEN"), ("Uganda", "UGA")):
            for year, delta in ((2018, 0), (2023, 2)):
                value = 31 + delta if measure == "employment_share" else 1.25 + delta
                organisation = "ILOSTAT" if measure == "employment_share" else "FAOSTAT"
                rows.append({"source": "bigquery", "content": f"{country} {measure} in {year} was {value}.",
                             "metadata": {"table_id": table, "status": "ok", "country_name": country,
                                          "year": year, "organisation_name": organisation,
                                          "source_id": measure, "source_url": f"https://example.org/{table}",
                                          "value_semantics": {"measure_value": value},
                                          "raw_row": {"country_iso3": iso, "year": year, "value": value,
                                                      "organisation_name": organisation}}})
    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", lambda *a, **kw: "Employment figures are missing [1].")
    result = node_generate({"query": query, "decomposition": facets, "task_mode": "analytical", "answer_lang": "en",
                            "turn_contract": turn.to_dict(), "bq_sql_plan": planned,
                            "bq_results": rows, "reranked_context": rows})
    assert "Employment figures are missing" not in result["answer"]
    assert "33.00 %" in result["answer"] and "+2.00 percentage points" in result["answer"]
    assert "+2.00 t/ha" in result["answer"]
    assert len(result["citations"]) == 2  # Source/indicator citations retain all eight observations in the answer.


def test_warehouse_and_document_plans_follow_the_same_turn(monkeypatch):
    contract = compile_turn_contract(QUESTION, FACETS)
    state = {
        "query": QUESTION, "decomposition": FACETS, "turn_contract": contract.to_dict(),
        "user_query": QUESTION, "detail_rewrite": "generic rewrite",
    }
    plan = node_bq_reason(state)["bq_sql_plan"]
    # Drought and general food insecurity lack a proven comparable series for this question.
    assert plan["fact_tables"] == {"yield": "agg_production_country_year"}
    for bind in plan["bind_contracts"].values():
        time_sql = bind["time_sql"]
        assert " OR " in time_sql
        assert all(str(year) in time_sql for year in (2018, 2019, 2022, 2023))
        assert "BETWEEN 2018 AND 2023" not in time_sql
        compiled = compile_operation(bind, job=contract.job)
        assert compiled and compiled.sql
        assert time_sql.removeprefix("AND ") in compiled.sql
        assert "yield_recomputed" in compiled.sql
        assert "production_qty" not in compiled.sql
        assert "product_name = 'Maize (corn)'" in compiled.sql
    assert _document_search_texts(state) == [QUESTION, FACETS["evidence_topics"][0]]
    vector = MagicMock()
    vector.return_value.retrieve.return_value = []
    monkeypatch.setattr("ml.rag.chatbot.graph.VectorRetriever", vector)
    _retrieve_academic_papers(state)
    searches = vector.return_value.retrieve.call_args_list
    assert any(FACETS["evidence_topics"][0] in call.args for call in searches)
    assert all(call.kwargs.get("published_at_to") is None for call in searches)


def test_decompose_keeps_comparison_operation_through_advisory_routing(monkeypatch):
    monkeypatch.setattr("ml.rag.chatbot.graph.decompose_query", lambda *args, **kwargs: dict(FACETS))
    monkeypatch.setattr("ml.rag.chatbot.graph.metacog_enabled", lambda: True)
    advisory = MagicMock()
    monkeypatch.setattr("ml.rag.chatbot.graph.build_metacog_plan", advisory)
    state = node_decompose({"query": QUESTION})
    assert state["route_candidate"] == "full_rag"
    assert state["turn_contract"]["job"] == "compare"
    advisory.assert_not_called()


def test_rows_from_wrong_places_or_periods_cannot_fill_a_gap():
    turn = compile_turn_contract(QUESTION, FACETS)
    rows = [
        {"metadata": {"table_id": "fct_yield", "raw_row": {"country_iso3": "KEN", "year": 2019, "value": 2.0}}},
        {"metadata": {"table_id": "fct_yield", "raw_row": {"country_iso3": "KEN", "year": 2020, "value": 2.0}}},
        {"metadata": {"table_id": "fct_yield", "raw_row": {"country_iso3": "UGA", "year": 2023, "value": 2.0}}},
        {"metadata": {"table_id": "fct_yield", "year": 2023}},
    ]
    plan = {"fact_tables": {"yield": "fct_yield"}, "bind_contracts": {"fct_yield": {"time_column": "year"}}}
    coverage = fact_requirement_coverage(turn, plan, rows, [{"status": "ok", "table_id": "fct_yield"}])
    yield_coverage = [entry for entry in coverage if entry["measure"] == "yield"]
    assert [entry["missing_years"] for entry in yield_coverage] == [[2018], [2022, 2023]]
    assert [entry["status"] for entry in yield_coverage] == ["no_match", "no_match"]
    assert {entry["status"] for entry in coverage if entry["measure"] != "yield"} == {"unsupported"}


def test_comparison_executes_validated_yield_sql_and_keeps_the_missing_period(monkeypatch):
    turn = compile_turn_contract(QUESTION, FACETS)
    state = {"query": QUESTION, "decomposition": FACETS, "turn_contract": turn.to_dict()}
    client = MagicMock()
    client.query.return_value.result.return_value = [
        {"value": 2.1, "year": 2019, "country_iso3": "KEN", "product_name": "Maize (corn)"},
    ]
    monkeypatch.setenv("BQ_PROJECT", "test")
    monkeypatch.setattr(BQRetriever, "_get_client", lambda self: client)
    monkeypatch.setattr("ml.rag.retrievers.bq_retriever.dry_run_sql", lambda *args, **kwargs: None)
    plan = node_bq_reason(state)
    result = node_bq_retrieve({**state, **plan})
    assert client.query.call_count == 1, result["bq_sql_debug"]
    submitted = client.query.call_args.args[0]
    assert "yield_recomputed" in submitted and " OR " in submitted
    coverage = fact_requirement_coverage(turn, plan["bq_sql_plan"], result["bq_results"], result["bq_sql_debug"])
    assert [entry["missing_years"] for entry in coverage if entry["measure"] == "yield"] == [[2018], [2022, 2023]]
    assert node_coverage_retry({
        **state, **plan, "reranked_context": [], "bq_results": [],
    }) == {}


def test_mixed_comparison_binds_both_series_and_rejects_an_unrelated_passage(monkeypatch):
    query = (
        "Compare maize yields and agricultural employment share in Kenya in 2018 versus 2023. "
        "Explain the effects of mechanisation on these indicators."
    )
    topic = "Effects of mechanisation on maize yield and agricultural employment in Kenya"
    facets = {**FACETS, "evidence_topics": [topic]}
    turn = compile_turn_contract(query, facets)
    state = {"query": query, "decomposition": facets, "turn_contract": turn.to_dict()}
    client = MagicMock()
    employment_indicator = "Share of employment in agriculture in total employment - ILO Modelled Estimates"

    def run_query(sql, **kwargs):
        job = MagicMock()
        if "fct_employment" in sql:
            assert employment_indicator in sql
            assert "sex = 'Total'" in sql and "unit = '%'" in sql
            assert "year BETWEEN 2018 AND 2018" in sql
            assert "Maize (corn)" not in sql and "Agricultural Use" not in sql
            extra = {
                "indicator": employment_indicator, "sex": "Total", "unit": "%",
                "as_of_date": date(2026, 1, 1), "as_of_date_basis": "loaded_at",
            }
        else:
            assert "yield_recomputed" in sql and "Maize (corn)" in sql
            extra = {"product_name": "Maize (corn)"}
        assert "2018" in sql and "2023" in sql and " OR " in sql
        job.result.return_value = [
            {"country_iso3": "KEN", "year": year, "value": 2.1, "organisation_name": "Statistical agency", **extra}
            for year in (2018, 2023)
        ]
        return job

    client.query.side_effect = run_query
    monkeypatch.setenv("BQ_PROJECT", "test")
    monkeypatch.setattr(BQRetriever, "_get_client", lambda self: client)
    monkeypatch.setattr("ml.rag.retrievers.bq_retriever.dry_run_sql", lambda *args, **kwargs: None)
    plan = node_bq_reason(state)
    assert plan["bq_sql_plan"]["fact_tables"] == {
        "yield": "agg_production_country_year", "employment_share": "fct_employment",
    }
    outcome = node_bq_retrieve({**state, **plan})
    assert client.query.call_count == 2, outcome["bq_sql_debug"]
    coverage = fact_requirement_coverage(turn, plan["bq_sql_plan"], outcome["bq_results"], outcome["bq_sql_debug"])
    assert len(coverage) == 4
    assert all(entry["status"] == "satisfied" for entry in coverage)
    semantics = [item["metadata"]["value_semantics"] for item in outcome["bq_results"]]
    assert {item["unit"] for item in semantics} == {"t/ha", "%"}
    assert {item["period"] for item in semantics} == {"2018", "2023"}
    assert all(item["grain"] and item["organisation_name"] for item in semantics)

    passages = [
        {"content": "In Kenya, mechanisation increased maize yield and reduced agricultural employment in 2023.",
         "metadata": {"title": "Farm survey", "country_name": "Kenya", "published_at": "2024-08-01"}},
        {"content": "Kenya's 2023 statistical workbook lists mechanisation, maize yield and agricultural employment as headings.",
         "metadata": {"title": "Catalogue", "country_name": "Kenya"}},
        {"content": "Kenya recorded maize yield of 2.4 t/ha in 2023.",
         "metadata": {"title": "Official release", "country_name": "Kenya"}},
    ]
    calls = []

    def rank(question, items, **kwargs):
        calls.append(question)
        return items

    monkeypatch.setattr("ml.rag.chatbot.graph.rerank", rank)
    monkeypatch.setattr("ml.rag.chatbot.graph._retrieve_public_reports", lambda _: passages)
    for name in ("news", "ota", "academic_papers"):
        monkeypatch.setattr(f"ml.rag.chatbot.graph._retrieve_{name}", lambda _: [])
    vector_result = node_parallel_retrieve(state)
    assert set(vector_result["corpus_selection"]["active"]) == {"ota", "news", "public_reports", "academic_papers"}
    evidence = {**state, **plan, **outcome, **vector_result}
    ranked = node_rerank({**evidence, **node_merge(evidence)})["reranked_context"]
    assert calls == [topic]
    assert document_requirement_coverage(turn, ranked) == [
        {"question": topic, "status": "candidate_available", "count": 1},
    ]
    by_title = {item["metadata"].get("title"): item["metadata"] for item in ranked}
    assert "Catalogue" not in by_title
    assert by_title["Official release"]["document_fact_candidates"] == ["yield"]
    assert not by_title["Official release"]["document_requirements"]
    assert document_requirement_coverage(
        turn, [], corpus_results={"public_reports": {"status": "timeout"}},
    )[0]["status"] == "document_search_timed_out"


def test_comparison_admits_only_the_bound_crop_place_and_grain():
    query = "Compare maize yield in Kenya and Uganda in 2023."
    facets = {
        "job": "compare", "geography": ["Kenya", "Uganda"], "entities": ["maize"],
        "evidence_topics": ["maize yield in Kenya and Uganda"],
    }
    turn = compile_turn_contract(query, facets)
    assert not turn.documents
    planned = node_bq_reason({"query": query, "decomposition": facets, "turn_contract": turn.to_dict()})["bq_sql_plan"]
    table = planned["fact_tables"]["yield"]
    assert table == "agg_production_country_year"
    raw = {"product_name": "Maize (corn)", "country_iso3": "KEN", "year": 2023, "value": 0}

    def row(**overrides):
        return {"metadata": {"table_id": table, "raw_row": {**raw, **overrides}}}

    assert matches_bound_fact(row(), turn, planned)
    for invalid in (row(product_name="Rice"), row(year=2022), row(admin_1="Nairobi"), row(value=None)):
        assert not matches_bound_fact(invalid, turn, planned)
    coverage = fact_requirement_coverage(turn, planned, [row()], [{"table_id": table, "status": "timeout"}])
    assert [(entry["geo"], entry["status"]) for entry in coverage] == [("Kenya", "satisfied"), ("Uganda", "timed_out")]
