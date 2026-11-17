from ml.rag.eval.run_ota_eval import run_case


def test_mixed_case_records_missing_documents_even_when_warehouse_returns_a_fact():
    case = {
        "id": "mixed",
        "locale": "en",
        "split": "development",
        "reference_date": "2026-11-08",
        "turns": [
            {
                "user": "What changed in Ghana, and what does the report say about why?",
                "expect": {"requested_evidence": {"mart_facts": True, "documents": True}},
                "catalogue_scope": {"indicator_ids": ["PROD-03", "NUT-07"]},
            },
        ],
    }

    def invoke(query, **kwargs):
        assert query == case["turns"][0]["user"]
        assert kwargs["reference_date"] == "2026-11-08"
        assert "catalogue_scope" not in kwargs
        return {
            "answer": "Here is the reported yield.",
            "bq_results": [{"source": "bigquery", "content": "yield: 3 t/ha", "metadata": {
                "value_semantics": {"measure_value": 3, "measure_column": "yield"},
            }}],
            "citations": [{"source": "bigquery", "text": "Warehouse series"}],
            "vector_results": [{"source": "vector", "content": " "}],
            "corpus_selection": {"results": {"policies": {"status": "timeout", "hits": 0}}},
        }

    result = run_case(case, invoke)[0]
    assert result["evidence"]["mart_facts"]["candidates"] == 1
    assert result["evidence"]["documents"]["candidates"] == 0
    assert result["evidence"]["documents"]["corpus_statuses"] == {"policies": "timeout"}
    assert result["missing_requested_evidence"] == ["documents"]
    assert result["ground_truth_score"] is None


def test_follow_up_replays_actual_reply_and_does_not_turn_an_execution_error_into_a_gap():
    def turn(user):
        return {
            "user": user,
            "expect": {"requested_evidence": {"mart_facts": True, "documents": False}},
            "catalogue_scope": {"indicator_ids": ["PROD-03"]},
        }

    case = {
        "id": "follow-up",
        "locale": "en",
        "split": "development",
        "reference_date": "2026-11-08",
        "turns": [turn("What was maize yield in Ghana?"), turn("And in 2024?")],
    }

    def invoke(query, **kwargs):
        if query.startswith("What"):
            return {"answer": "The last confirmed period was 2022.", "active_request": {"turn": 1}}
        assert kwargs["recent_turns"][-1] == {
            "role": "assistant", "content": "The last confirmed period was 2022.",
        }
        assert kwargs["active_request"] == {"turn": 1}
        raise TimeoutError("Provider deadline")

    results = run_case(case, invoke)
    assert len(results) == 2
    assert results[1]["error_kind"] == "TimeoutError"
    assert results[1]["missing_requested_evidence"] is None
