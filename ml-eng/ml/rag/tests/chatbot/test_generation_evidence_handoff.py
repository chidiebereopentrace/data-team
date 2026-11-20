from datetime import date

import pytest

from ml.rag.chatbot.bq_context_enrich import enrich_bq_results
from ml.rag.chatbot.facet_compiler import compile_turn_contract
from ml.rag.chatbot.generator import _build_context_block, drop_price_off_topic, note_time_fallback
from ml.rag.chatbot.graph import node_bq_retrieve, node_generate, node_rerank
from ml.rag.retrievers.bq_retriever import BQRetriever


@pytest.mark.parametrize("model_status", ["complete", "timeout", "length"])
def test_food_price_report_keeps_each_country_history_and_document_context(monkeypatch, model_status):
    monkeypatch.setenv("RAG_GENERATE_CONTEXT_MAX_CHARS", "8000")
    monkeypatch.setenv("RAG_GENERATE_MAX_TOKENS", "8192")
    query = "Detailed analysis of food price changes in Nigeria and Cameroon since 2020."
    dec = {"job": "trend", "geography": ["Nigeria", "Cameroon"], "entities": [],
           "primary_measures": ["market_price"], "time_start": "2020-01-01", "time_end": "2026-11-18"}
    turn = compile_turn_contract(query, dec)
    rows = enrich_bq_results([
        {"source": "bigquery", "content": "", "metadata": {
            "status": "ok", "template": "mart_price", "price_report": True,
            "coverage_note": (
                "Commodity coverage is a staple basket, not all foods or an official CPI basket. "
                "National food indices are separate series. Missing years stay missing. "
                "Current-year averages cover only recorded observations and are provisional."
            ),
            "sql": "SELECT value FROM `test.mart_dev.fct_prices`",
            "raw_row": {
                "country_iso3": country, "product_name": product, "requested_product": product, "price_type": "Retail",
                "price_source": "wfp", "unit": "KG", "currency": currency, "avg_price": 200,
                "first_year": 2020, "year": 2026, "first_price": 100, "change_pct": 100,
                "annual_prices": "2020: 100.00; 2023: 150.00; 2026: 200.00",
                "period_start": date(2020, 1, 1), "latest_date": date(2026, 10, 1),
            },
        }}
        for country, currency in (("NGA", "NGN"), ("CMR", "XAF"))
        for product in ("Maize", "Rice", "Cassava", "Millet", "Sorghum", "Yam", "Beans", "Plantain")
    ], query=query, decomposition=dec)
    assert sum(len(row["content"].encode("utf-8")) for row in rows) > 6000
    report = {"source": "news", "content": "Food prices in Cameroon were affected by transport disruptions.",
              "metadata": {"title": "Market update", "country_name": "Cameroon"}}
    state = {"query": query, "decomposition": dec, "task_mode": "analytical", "analytical_mode": True,
             "turn_contract": turn.to_dict(), "bq_results": rows, "merged_context": [*rows, report],
             "rerank_top_k": 14, "corpus_selection": {"active": ["news"]}}
    monkeypatch.setattr(BQRetriever, "retrieve", lambda *a, **kw: rows)
    retrieved = node_bq_retrieve({**state, "bq_sql_plan": {"selected_tables": ["fct_prices"]}})
    assert len(retrieved["bq_results"]) == 16
    ranked = node_rerank(state)
    assert len(ranked["reranked_context"]) == 17

    def model(messages, **kwargs):
        assert kwargs["max_tokens"] == 8192
        context = messages[-1]["content"]
        assert "NGN" in context and "XAF" in context
        assert all(value in context for value in ("100.00", "150.00", "200.00"))
        assert "transport disruptions" in context
        return "Transport disruptions affected food prices [17]." if model_status != "timeout" else None

    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", model)
    monkeypatch.setattr("ml.rag.chatbot.generator.answer_was_length_capped", lambda: model_status == "length")
    result = node_generate({**state, **ranked})
    assert "**Nigeria**" in result["answer"] and "**Cameroon**" in result["answer"]
    assert len(result["citations"]) == (2 if model_status == "complete" else 1)
    assert "What the retrieved sources say" not in result["answer"]
    assert "Ask about fewer countries" not in result["answer"]


def test_current_price_uses_verified_observation_without_a_model_call(monkeypatch):
    query = "kedu ego ole ana-ele osikapa na kenya ugbua?"
    planning_query = "What is the current market price of rice in Kenya?"
    dec = {
        "entities": ["rice"], "geography": ["Kenya"], "primary_measures": ["market_price"],
        "time_start": "2026-08-20", "time_end": "2026-11-18", "time_phrase": "current",
    }
    turn = compile_turn_contract(planning_query, dec, original_question=query, answer_lang="ig")
    rows = enrich_bq_results([{
        "source": "bigquery", "content": "Rice price", "metadata": {
            "status": "ok", "template": "mart_price",
            "sql": "SELECT AVG(value) FROM `proj.mart_dev.fct_prices` WHERE year = 2026",
            "raw_row": {
                "product_name": "Rice", "country_iso3": "KEN", "price_type": "Retail", "price_source": "wfp",
                "unit": "KG", "currency": "KES", "avg_price": 132.086,
                "period_start": date(2026, 10, 1), "latest_date": date(2026, 10, 1),
            },
        },
    }], query=planning_query, decomposition=dec)
    avocado = {"source": "news", "content": "Kenya's avocado prices went up.", "metadata": {
        "title": "Avocado exports", "published_at": "2026-05-22",
    }}
    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("A verified lookup can be rendered directly")

    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", unexpected_model_call)

    result = node_generate({
        "query": query, "user_query": query, "decomposition": dec, "task_mode": "fact_lookup",
        "answer_lang": "ig", "turn_contract": turn.to_dict(), "bq_results": rows,
        "reranked_context": [*rows, avocado], "retrieval_partial": True,
    })

    assert rows[0]["metadata"]["as_of_date"] == "2026-10-01"
    assert not dec.get("time_fallback")
    assert "132.09" in result["answer"] and "2026-10-01" in result["answer"]
    assert query not in result["answer"]
    assert result["answer_lang"] == "ig"
    assert [citation["kind"] for citation in result["citations"]] == ["structured_data"]


def test_unpacked_passages_cannot_be_cited():
    items = [
        {"source": "news", "content": "Text from the selected passage.", "metadata": {"title": "Packed"}},
        {"source": "news", "content": "", "metadata": {"title": "Never sent to the model"}},
    ]
    block, registry = _build_context_block(items, budget=600, chunk_cap=300)
    assert "Text from the selected passage." in block
    assert [ref.item for ref in registry] == [items[0]]


def test_price_word_cannot_satisfy_the_requested_crop():
    avocado = {"source": "news", "content": "Avocado prices in Kenya went up."}
    rice = {"source": "news", "content": "Retail rice prices in Kenya went up."}
    dec = {"entities": ["rice"], "geography": ["Kenya"], "primary_measures": ["market_price"]}
    assert drop_price_off_topic([avocado, rice], "What is the rice price in Kenya?", dec) == [rice]


def test_price_freshness_uses_observations_instead_of_document_publication():
    scope = {"time_start": "2026-08-20", "time_end": "2026-11-18", "time_phrase": "current"}
    article = {"source": "news", "metadata": {"published_at": "2026-08-18"}}
    for observed in ("2026-10-01", "2026-07-01"):
        decomposition = dict(scope)
        row = {"source": "bigquery", "metadata": {
            "as_of_date": "2026-01-01", "raw_row": {"latest_date": observed},
        }}
        note_time_fallback(decomposition, [article, row])
        if observed == "2026-10-01":
            assert "time_fallback" not in decomposition
        else:
            assert decomposition["time_fallback"]["used"] == {"start": observed, "end": observed}


def test_yield_comparison_keeps_both_countries_in_a_tight_pack(monkeypatch):
    query = "Compare maize yield in Kenya and Uganda in 2023."
    dec = {"job": "compare", "geography": ["Kenya", "Uganda"], "entities": ["maize"]}
    turn = compile_turn_contract(query, dec)
    table = "agg_production_country_year"
    plan = {"plan_source": "turn_requirements", "fact_tables": {"yield": table},
            "bind_contracts": {table: {"time_column": "year", "product_literals": ["Maize (corn)"]}}}

    def row(country, value):
        country_name = {"KEN": "Kenya", "UGA": "Uganda"}[country]
        return {"source": "bigquery", "content": f"{country} maize yield: {value} t/ha", "metadata": {
            "table_id": table, "status": "ok", "organisation_name": "FAOSTAT", "country_name": country_name,
            "value_semantics": {"measure_column": "value", "measure_value": value, "unit": "t/ha"},
            "raw_row": {
                "country_iso3": country, "country_name": country_name,
                "year": 2023, "value": value, "product_name": "Maize (corn)",
            },
        }}

    kenya, duplicate, uganda = row("KEN", 2.4), row("KEN", 2.4), row("UGA", 3.1)
    state = {
        "query": query, "task_mode": "analytical", "decomposition": dec, "turn_contract": turn.to_dict(),
        "bq_sql_plan": plan, "merged_context": [kenya, duplicate, uganda], "rerank_top_k": 2,
    }
    result = node_rerank(state)
    assert result["reranked_context"] == [kenya, uganda]

    def model(messages, **kwargs):
        user_prompt = messages[-1]["content"]
        assert "KEN maize yield" in user_prompt and "UGA maize yield" in user_prompt
        assert "Old regional report" not in user_prompt
        return "Maize yields were 2.4 t/ha in Kenya [1] and 3.1 t/ha in Uganda [2]."

    report = {"source": "public_report", "content": "Old regional report on farm labour.",
              "metadata": {"title": "Old regional report"}}
    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", model)
    answer = node_generate({**state, "bq_results": [kenya, uganda], "reranked_context": [kenya, uganda, report]})
    assert "2.40 t/ha" in answer["answer"] and "3.10 t/ha" in answer["answer"]
    assert [item["kind"] for item in answer["citations"]] == ["structured_data"]


def test_mixed_comparison_preserves_completed_indicators_when_a_source_times_out(monkeypatch):
    query = (
        "Compare Kenya's rainfall, maize production, and IPC food insecurity in 2018 versus 2023. "
        "What do assessments say about their relationship?"
    )
    topic = "Assessments of rainfall, maize production and IPC food insecurity in Kenya"
    dec = {"job": "compare", "geography": ["Kenya"], "entities": ["maize"], "evidence_topics": [topic]}
    turn = compile_turn_contract(query, dec)
    assert {fact.measure_id for fact in turn.facts} == {"climate", "production", "food_security_ipc"}
    tables = {"climate": "fct_climate", "production": "agg_production_country_year", "food_security_ipc": "fct_food_security"}
    plan = {"plan_source": "turn_requirements", "fact_tables": tables,
            "bind_contracts": {table: {"time_column": "year"} for table in tables.values()}}

    def row(measure, year, value, unit):
        return {"source": "bigquery", "content": f"Kenya {measure} in {year}: {value} {unit}", "metadata": {
            "table_id": tables[measure], "status": "ok", "source_id": f"{measure}:{year}",
            "organisation_name": "Statistical agency",
            "value_semantics": {"measure_column": "value", "measure_value": value, "unit": unit, "period": str(year)},
            "raw_row": {"country_iso3": "KEN", "year": year, "value": value, "unit": unit},
        }}

    rows = [row("climate", 2018, 720, "mm"), row("climate", 2023, 834, "mm"),
            row("production", 2018, 3900000, "tonnes"), row("production", 2023, 4200000, "tonnes")]

    def model(messages, **kwargs):
        model_input = "\n".join(message["content"] for message in messages)
        assert all(item["content"] in model_input for item in rows)
        assert "food_security_ipc" in model_input and "timed out" in model_input
        return "Rainfall and maize production are documented. The food-security and assessment searches timed out."

    monkeypatch.setattr("ml.rag.chatbot.generator._call_llama", model)
    result = node_generate({
        "query": query, "task_mode": "analytical", "decomposition": dec,
        "turn_contract": turn.to_dict(), "bq_sql_plan": plan, "bq_results": rows,
        "reranked_context": rows, "retrieval_partial": True,
        "bq_sql_debug": [{"table_id": table, "status": "timeout" if key == "food_security_ipc" else "ok"}
                         for key, table in tables.items()],
        "corpus_selection": {"results": {"public_reports": {"status": "timeout"}}},
    })

    assert "720.00 mm" in result["answer"] and "834.00 mm" in result["answer"]
    assert "3,900,000.00 tonnes" in result["answer"] and "4,200,000.00 tonnes" in result["answer"]
    assert {entry["measure"] for entry in result["comparison_gaps"]} == {"food_security_ipc"}
    assert {entry["status"] for entry in result["comparison_gaps"]} == {"timed_out"}
