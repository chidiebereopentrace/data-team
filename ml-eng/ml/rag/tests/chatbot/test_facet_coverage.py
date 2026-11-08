"""A comparison is not covered by evidence for only one of the requested places."""
from __future__ import annotations

from ml.rag.chatbot.coverage_retry import should_coverage_retry, slots_missing_from_evidence
from ml.rag.chatbot.facet_coverage import build_facet_matrix


def test_nigeria_text_does_not_cover_ghana() -> None:
    state = {
        "decomposition": {"geography": ["Ghana", "Nigeria"], "entities": ["yam"]},
        "bq_results": [{"content": "Nigeria yam production was 1000 tonnes in 2023."}],
        "route_candidate": "full_rag",
        "coverage_retry": 0,
    }
    cells = {cell["name"]: cell["status"] for cell in build_facet_matrix(state)}
    assert cells["Nigeria"] == "supported"
    assert cells["Ghana"] == "absent"
    assert slots_missing_from_evidence(state) is True
    assert should_coverage_retry(state) is True
