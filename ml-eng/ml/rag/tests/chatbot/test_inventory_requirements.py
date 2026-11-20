"""Broad member discovery must survive the real warehouse-to-answer path."""
from unittest.mock import MagicMock

import pytest

from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.evidence_requirements import fact_requirement_coverage
from ml.rag.chatbot.graph import node_bq_reason, node_bq_retrieve, node_generate, node_rerank
from ml.rag.retrievers.bq_retriever import BQRetriever

QUESTION = "give me a table itemizing the crop production in nigeria in 2020"
EXPANDED = (
    "Provide a table listing the crop production quantities in Nigeria for the year 2020, "
    "including major crops such as cassava, maize, rice, yam, sorghum, millet, groundnut, "
    "cotton, sugarcane, and palm oil, with production values in metric tons for each crop "
    "and region where available."
)
FACETS = {
    "intent": "descriptive", "geo_scope": "country", "geography": ["Nigeria"],
    "entities": ["production", "cassava", "maize", "rice", "yam", "cotton"],
    "domains": ["crop"], "primary_measures": ["production"],
    "time_start": "2020-01-01", "time_end": "2020-12-31", "reference_date": "2026-11-19",
    "time_windows": [{"start": "2020-01-01", "end": "2020-12-31", "grain": "year",
                      "phrase": "in 2020", "source": "query", "strict": True}],
    "fact_requirements": [{"measure_id": "production", "question": EXPANDED}],
}


@pytest.mark.parametrize("count", [13, 110])
def test_expanded_inventory_returns_all_products_with_verified_production(monkeypatch, count):
    turn = compile_turn_contract(EXPANDED, FACETS, original_question=QUESTION)
    state = {"query": EXPANDED, "user_query": QUESTION, "decomposition": FACETS,
             "task_mode": "fact_lookup", "answer_lang": "en", "turn_contract": turn.to_dict()}
    planned = node_bq_reason(state)
    assert not planned["bq_sql_plan"]["skip_bq"], planned["bq_sql_plan"]
    client = MagicMock()
    client.query.return_value.result.return_value = [
        {"value": 1000 + i, "year": 2020, "country_iso3": "NGA", "country_name": "Nigeria",
         "product_name": f"Crop {i:02d}", "organisation_name": "FAOSTAT"}
        for i in range(count)
    ]
    monkeypatch.setenv("BQ_PROJECT", "test")
    monkeypatch.setattr(BQRetriever, "_get_client", lambda self: client)
    monkeypatch.setattr("ml.rag.retrievers.bq_retriever.dry_run_sql", lambda *a, **kw: None)
    retrieved = node_bq_retrieve({**state, **planned})
    assert client.query.call_count == 1, retrieved["bq_sql_debug"]
    sql = client.query.call_args.args[0]
    assert "production_qty AS value" in sql and "yield_recomputed AS value" not in sql
    assert "NGA" in sql and "2020" in sql
    assert "product_name IN" not in sql and "product_name =" not in sql
    assert "cpcv2" in sql and "LIKE '01%'" in sql
    assert len(retrieved["bq_results"]) == count
    monkeypatch.setattr("ml.rag.chatbot.graph.rerank", lambda q, items, **kw: items)
    packed = node_rerank({**state, **planned, **retrieved, "rerank_top_k": 4,
                          "merged_context": retrieved["bq_results"]})
    assert len(packed["reranked_context"]) == count
    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", lambda *a, **kw: None)
    answer = node_generate({**state, **planned, **retrieved, **packed})
    for i in range(count):
        assert f"Crop {i:02d}" in answer["answer"]
        assert f"{1000+i:,.2f}" in answer["answer"]
    assert "2020" in answer["answer"] and "Nigeria" in answer["answer"]
    assert "|" in answer["answer"] and " t " in answer["answer"]
    assert len(answer["citations"]) == count
    assert {gap["status"] for gap in answer["fact_requirement_coverage"]} == {"satisfied"}


def test_unexecuted_inventory_cannot_be_answered_from_unrelated_reports(monkeypatch):
    turn = compile_turn_contract(EXPANDED, FACETS, original_question=QUESTION)
    state = {"query": EXPANDED, "user_query": QUESTION, "decomposition": FACETS,
             "task_mode": "fact_lookup", "answer_lang": "en", "turn_contract": turn.to_dict()}
    planned = node_bq_reason(state)
    irrelevant = {"source": "public_report", "content": "Nigeria recorded 100 COVID cases in 2020.",
                  "metadata": {"country_name": "Nigeria", "source_id": "covid", "publication_year": 2020}}
    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", lambda *a, **kw: "There is no crop production data [1].")
    result = node_generate({**state, **planned, "reranked_context": [irrelevant], "bq_results": []})
    assert "planned query did not run" in result["answer"]
    assert "There is no crop production data" not in result["answer"]
    assert not result["citations"]


@pytest.mark.parametrize("question", [
    "Give me a table of maize and rice production in Nigeria in 2020.",
    "Give me a table listing palm oil production in Nigeria in 2020.",
])
def test_explicit_product_sets_compile_catalog_filters(question):
    facets = {**FACETS, "fact_requirements": [{"measure_id": "production", "question": question}]}
    turn = compile_turn_contract(question, facets)
    plan = node_bq_reason({"query": question, "decomposition": facets, "turn_contract": turn.to_dict()})["bq_sql_plan"]
    assert not plan["skip_bq"]
    bind = plan["bind_contracts"]["production"]
    assert bind["product_literals"]
    assert bind["member_selection"] == "explicit"
    if "maize and rice" in question:
        assert len(bind["product_literals"]) == 2


@pytest.mark.parametrize(("question", "measure"), [
    (QUESTION, "production"),
    ("Give me a table itemizing agricultural imports in Nigeria in 2020.", "trade"),
])
def test_broad_inventories_need_no_model_invented_product_list(question, measure):
    facets = {**FACETS, "entities": [], "primary_measures": [measure],
              "fact_requirements": [{"measure_id": measure, "question": question}]}
    turn = compile_turn_contract(question, facets)
    assert turn.job != "clarify"
    plan = node_bq_reason({"query": question, "decomposition": facets, "turn_contract": turn.to_dict()})["bq_sql_plan"]
    assert not plan["skip_bq"]
    assert plan["bind_contracts"][measure]["member_selection"] == "discover"


def test_expansion_cannot_replace_explicit_place_period_or_national_grain():
    expanded = EXPANDED.replace("and region where available", "by state or region where available in Kenya in 2022")
    facets = {**FACETS, "job": "breakdown", "geography": ["Nigeria", "Kenya"],
              "fact_requirements": [{"measure_id": "production", "question": expanded}]}
    turn = compile_turn_contract(expanded, facets, original_question=QUESTION)
    plan = node_bq_reason({"query": expanded, "decomposition": facets, "turn_contract": turn.to_dict()})["bq_sql_plan"]
    assert plan["plan_source"] == "turn_requirements"
    bind = plan["bind_contracts"]["production"]
    assert bind["request_scope"]["geography"] == ["Nigeria"]
    assert bind["request_scope"]["time_start"] == "2020-01-01"
    assert bind["request_scope"]["time_end"] == "2020-12-31"
    assert not turn.breakdown


def test_missing_product_cannot_be_hidden_by_a_sibling_observation():
    question = "Give me a table of maize and rice production in Nigeria in 2020."
    facets = {**FACETS, "fact_requirements": [{"measure_id": "production", "question": question}]}
    turn = compile_turn_contract(question, facets)
    plan = node_bq_reason({"query": question, "decomposition": facets, "turn_contract": turn.to_dict()})["bq_sql_plan"]
    members = plan["bind_contracts"]["production"]["product_literals"]
    rows = [{"source": "bigquery", "content": "Verified quantity",
             "metadata": {"table_id": "agg_production_country_year", "requirement_id": "production", "status": "ok",
                          "raw_row": {"value": 10, "product_name": members[0], "country_iso3": "NGA", "year": 2020}}}]
    coverage = fact_requirement_coverage(turn, plan, rows, [{"status": "ok", "requirement_id": "production"}])
    assert {entry["member"]: entry["status"] for entry in coverage} == {members[0]: "satisfied", members[1]: "no_match"}


def test_relevant_reported_quantity_can_answer_when_warehouse_is_unavailable(monkeypatch):
    turn = compile_turn_contract(EXPANDED, FACETS, original_question=QUESTION)
    state = {"query": EXPANDED, "decomposition": FACETS, "task_mode": "fact_lookup", "answer_lang": "en",
             "turn_contract": turn.to_dict()}
    planned = node_bq_reason(state)
    report = {"source": "public_report", "content": "Nigeria's cassava production in 2020 was 1000 tonnes.",
              "metadata": {"country_name": "Nigeria", "source_id": "production-report", "article_title": "Production report",
                           "source_url": "https://example.org/production", "publication_year": 2021}}
    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", lambda *a, **kw: "Nigeria's cassava production in 2020 was 1000 tonnes, according to the report [1].")
    result = node_generate({**state, **planned, "reranked_context": [report], "bq_results": [],
                            "bq_sql_debug": [{"requirement_id": "production", "status": "timeout"}]})
    assert "1000 tonnes" in result["answer"]
    assert len(result["citations"]) == 1 and result["citations"][0]["kind"] != "structured_data"
    assert {entry["status"] for entry in result["fact_requirement_coverage"]} == {"timed_out"}


def test_cereal_group_uses_catalog_classification_instead_of_a_sample_basket():
    question = "Give me a table itemizing cereal production in Nigeria in 2020."
    facets = {**FACETS, "entities": ["cereals"], "fact_requirements": [{"measure_id": "production", "question": question}]}
    turn = compile_turn_contract(question, facets)
    plan = node_bq_reason({"query": question, "decomposition": facets, "turn_contract": turn.to_dict()})["bq_sql_plan"]
    assert not plan["skip_bq"]
    bind = plan["bind_contracts"]["production"]
    assert bind["product_group_prefix"] == "011"
    assert "LIKE '011%'" in bind["required_filters_sql"]
    assert not bind["product_literals"]


def test_listing_all_crops_does_not_bind_the_single_crops_aggregate():
    question = "Give me a table listing all crops produced in Nigeria in 2020."
    facets = {**FACETS, "entities": [], "fact_requirements": [{"measure_id": "production", "question": question}]}
    turn = compile_turn_contract(question, facets)
    plan = node_bq_reason({"query": question, "decomposition": facets, "turn_contract": turn.to_dict()})["bq_sql_plan"]
    bind = plan["bind_contracts"]["production"]
    assert bind["member_selection"] == "discover"
    assert bind["product_group_prefix"] == "01"
    assert not bind["product_literals"]
