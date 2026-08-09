"""Decomposition facets must be grounded in the user query text."""

from __future__ import annotations

from unittest import mock

from ml.rag.chatbot.query_decomposer import decompose_query


def test_igbo_yam_query_does_not_keep_hallucinated_nigeria() -> None:
    q = "kedu obodo kacha ako ji na mba africa"
    fake_llm = {
        "intent": "locate",
        "entities": ["Nigeria", "Africa"],
        "geography": ["Nigeria"],
        "domains": ["agriculture"],
        "time_start": "",
        "time_end": "",
    }
    with mock.patch(
        "ml.rag.chatbot.query_decomposer._call_llama_decompose",
        return_value=fake_llm,
    ):
        out = decompose_query(q)
    geo_l = [g.lower() for g in out.get("geography") or []]
    ent_l = [e.lower() for e in out.get("entities") or []]
    assert "nigeria" not in geo_l
    assert "nigeria" not in ent_l
    # Africa may remain as an entity (present in query) but is dropped from geo filters.
    assert "africa" in ent_l or "africa" not in geo_l
