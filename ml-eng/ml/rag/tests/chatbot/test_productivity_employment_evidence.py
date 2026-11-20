"""Preserve all requested evidence and use compact warehouse source labels."""
from ml.rag.chatbot.evidence_requirements import plan_fact_requirements
from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.generator import _format_source_citation, _registry_from_context_items, referenced_citations
from ml.rag.chatbot.document_evidence import requirement_document_candidates
from ml.rag.chatbot.graph import node_rerank


QUESTION = (
    "Examine whether agricultural productivity improvements in Nigeria, Ghana and Uganda "
    "between 2015 and 2024 coincided with changes in agricultural employment and women’s participation. "
    "Distinguish agriculture’s share of total employment from women’s share of agricultural employment. "
    "Report sex-disaggregated findings only where the required denominators are available. "
    "Use household studies or evaluations to discuss livelihood implications, and do not equate "
    "declining agricultural employment shares with declining farmer incomes."
)
FACETS = {
    "intent": "descriptive", "job": "report", "geo_scope": "multi_country",
    "entities": ["livelihoods", "agriculture"], "geography": ["Nigeria", "Ghana", "Uganda"],
    "time_start": "2015-01-01", "time_end": "2024-12-31", "reference_date": "2026-11-20",
    "primary_measures": ["employment_share", "production", "yield"],
    "evidence_topics": ["agricultural productivity improvements", "livelihood implications"],
}


def employment_row(year):
    row = {"country_name": "Ghana", "year": year, "value": 35.2, "unit": "%", "metric": "employment_share"}
    return {"source": "bigquery", "content": str(row), "metadata": {
        **row, "raw_row": row, "table_id": "fct_employment", "source_name": "FAOSTAT",
        "requirement_id": "employment_share", "value_semantics": {"measure_label": "Agricultural employment share", "unit": "%"},
    }}


def test_warehouse_citation_labels_name_source_and_indicator_without_row_dump():
    label = _format_source_citation(employment_row(2015))
    assert label == "FAOSTAT — Agricultural employment share"


def test_rows_from_one_source_indicator_share_one_public_citation():
    registry = _registry_from_context_items([employment_row(2015), employment_row(2016)])
    citations = referenced_citations("2015 [1]; 2016 [2]", registry)
    assert len(citations) == 1
    assert citations[0]["text"] == "FAOSTAT — Agricultural employment share"


def test_productivity_employment_question_plans_the_available_sector_productivity_indicator():
    contract = compile_turn_contract(QUESTION, FACETS)
    plan = plan_fact_requirements(contract, query=QUESTION, facets=FACETS)
    assert plan is not None
    tables = set(plan["selected_tables"])
    assert "fct_employment" in tables
    assert "labour_productivity" in plan["fact_tables"], plan.get("fact_tables")
    assert "yield" not in plan["fact_tables"], "Sector productivity must not silently become crop yield"
    assert len(plan["query_intents"]) >= 2
    filters = dict(plan["bind_contracts"]["labour_productivity"]["measure_filters"])
    assert filters == {"indicator": "Agriculture value added per worker (constant 2015 US$)", "sex": "Total", "unit": "USD"}


def household_study():
    return {"source": "academic", "score": 0.9, "content": (
        "The lower productivity in female-managed farms is an indication of the differences in the factors "
        "of agricultural production between genders. The production capacity of farm households in Nigeria "
        "is limited by their poor access to land, new technology and credit facilities. Women face greater "
        "vulnerabilities because of limited access to inputs and unpaid domestic work (Phillip et al. 2009). "
        "The study examines how input utilization influences the productivity of labor among men and women "
        "in Nigeria and how policies could reduce poverty in agricultural households."
    ), "metadata": {"geo_country_primary": "Nigeria", "publication_year": "2018",
        "article_title": "Input Utilization and Agricultural Labor Productivity", "doi": "10.1007/978-3-319-76222-7"}}


def test_historical_references_do_not_disqualify_an_in_period_household_study():
    turn = compile_turn_contract(QUESTION, FACETS)
    accepted = requirement_document_candidates(turn, [household_study()])
    assert accepted, "A 2018 study must not be excluded because it cites a 2009 paper"
    assert accepted[0]["metadata"]["document_requirements"]


def test_rerank_keeps_requested_explanatory_studies_alongside_warehouse_rows(monkeypatch):
    seen = []
    def rank(query, documents, **kwargs):
        seen.extend(documents)
        return [{**doc, "_rerank_score": 0.9} for doc in documents]
    monkeypatch.setattr("ml.rag.chatbot.graph.rerank", rank)
    turn = compile_turn_contract(QUESTION, FACETS)
    result = node_rerank({"query": QUESTION, "decomposition": FACETS, "task_mode": "analytical",
        "turn_contract": turn.to_dict(), "corpus_selection": {"active": ["academic", "public_reports"]},
        "merged_context": [*[employment_row(year) for year in range(2015, 2024)], household_study()]})
    assert seen and all(item["source"] != "bigquery" for item in seen)
    assert any(item["source"] == "academic" for item in result["reranked_context"])
    assert any((item.get("metadata") or {}).get("document_requirements") for item in result["reranked_context"])


def test_public_citation_grouping_remaps_markers_without_losing_observations():
    from ml.rag.chatbot.generator import _finalize_generation_result
    registry = _registry_from_context_items([employment_row(2015), employment_row(2016)])
    result = _finalize_generation_result(
        "Ghana’s employment share was 35.2% in 2015 [1] and 35.2% in 2016 [2].", registry,
        query="Compare Ghana agricultural employment shares in 2015 and 2016", inline_citations=True, model_output=False,
    )
    assert len(result.citations) == 1 and len(registry) == 2
    assert "2015" in result.answer and "2016" in result.answer and "35.2" in result.answer
    assert "[2]" not in result.answer


def test_different_warehouse_indicators_and_documents_keep_distinct_citations():
    from ml.rag.chatbot.citation_pack import consolidate_warehouse_citations
    prose, citations = consolidate_warehouse_citations("Share [1], productivity [2], study [3], another year [4].", [
        {"id": 1, "kind": "structured_data", "text": "FAOSTAT — Employment share"},
        {"id": 2, "kind": "structured_data", "text": "FAOSTAT — Labour productivity"},
        {"id": 3, "kind": "academic", "text": "Household study"},
        {"id": 4, "kind": "structured_data", "text": "FAOSTAT — Employment share"},
    ])
    assert [c["id"] for c in citations] == [1, 2, 3]
    assert "[2]" in prose and "[3]" in prose and "[4]" not in prose


def test_productivity_disambiguates_land_and_labour_measures():
    from ml.rag.chatbot.agri_measure_ontology import explicit_measure_ids
    assert "labour_productivity" in explicit_measure_ids("Agricultural productivity in Nigeria since 2015")
    assert "yield" not in explicit_measure_ids("Agricultural productivity in Nigeria since 2015")
    assert "yield" in explicit_measure_ids("Maize productivity in Nigeria since 2015")
    assert "labour_productivity" not in explicit_measure_ids("Maize productivity in Nigeria since 2015")
    assert {"yield", "labour_productivity"} <= set(explicit_measure_ids("Compare maize yield and labour productivity"))


def test_verified_report_context_reserves_space_for_the_requested_study(monkeypatch):
    from ml.rag.chatbot.generator import generate
    turn = compile_turn_contract(QUESTION, FACETS)
    plan = plan_fact_requirements(turn, query=QUESTION, facets=FACETS)
    rows = []
    for country in ("Ghana", "Nigeria", "Uganda"):
        for year in range(2015, 2024):
            for measure, indicator, unit in (
                ("employment_share", "Share of employment in agriculture in total employment - ILO Modelled Estimates", "%"),
                ("labour_productivity", "Agriculture value added per worker (constant 2015 US$)", "USD"),
            ):
                row = employment_row(year)
                raw = {"country_name": country, "year": year, "value": 35.2, "unit": unit, "sex": "Total", "indicator": indicator}
                row["metadata"].update(raw_row=raw, requirement_id=measure,
                    value_semantics={"measure_label": indicator, "unit": unit})
                row["content"] = str(raw) + " " + "Verified warehouse observation with units and provenance. " * 8
                rows.append(row)
    study = household_study()
    essential = "Quantile regression accounts for plot, household and village characteristics in Nigeria."
    study["content"] += " Methods and sampling limitations for the household panel." * 20 + " " + essential
    monkeypatch.setenv("RAG_GENERATE_CONTEXT_MAX_CHARS", "12000")
    def model(messages, **kwargs):
        context = messages[-1]["content"]
        assert essential in context, "Warehouse row headers must not crowd the requested study out of the model context"
        return "The Nigeria study discusses household input constraints [55]."
    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", model)
    result = generate(QUESTION, [*rows, study], comparison_plan=plan, turn_contract=turn.to_dict(),
        decomposition=FACETS, task_mode="analytical", answer_lang="en", require_document_citations=True)
    assert any(citation["kind"] == "academic" for citation in result.citations)
