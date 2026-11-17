"""Six first-class Qdrant corpus paths: retrieve wiring, soft-fail, merge kinds."""
from __future__ import annotations

from threading import Event
from typing import Any
from unittest import mock

import pytest

from ml.rag.chatbot import graph as graph_mod
from ml.rag.text_processors.chunking_config import COLLECTION_ALIASES, profile_for_collection


def _hit(doc_kind: str, content: str = "chunk") -> dict[str, Any]:
    return {
        "content": content,
        "score": 0.9,
        "metadata": {"doc_kind": doc_kind, "id": f"{doc_kind}-1"},
    }


def test_full_rag_fans_out_vector_and_bq() -> None:
    compiled = graph_mod.build_graph()
    edges = {(e.source, e.target) for e in compiled.get_graph().edges}
    assert ("parallel_retrieve", "bq_reason") not in edges
    assert ("retrieve_legs", "merge") in edges
    assert ("decompose", "retrieve_legs") in edges


def test_chunking_aliases_cover_six_collections() -> None:
    assert COLLECTION_ALIASES["news_data"] == "news"
    assert COLLECTION_ALIASES["academic_papers"] == "research"
    assert COLLECTION_ALIASES["policies"] == "research"
    assert COLLECTION_ALIASES["public_reports"] == "research"
    assert COLLECTION_ALIASES["formation"] == "research"
    assert COLLECTION_ALIASES["OTA_insights"] == "ota"
    assert COLLECTION_ALIASES["news_public_reports"] == "research"  # legacy only
    assert profile_for_collection("public_reports").qdrant_collection


@pytest.mark.parametrize("env, corpus", [
    ("QDRANT_COLLECTION_NEWS", "news"),
    ("QDRANT_COLLECTION_ACADEMIC_PAPERS", "research"),
    ("QDRANT_COLLECTION_POLICIES", "research"),
    ("QDRANT_COLLECTION_PUBLIC_REPORTS", "research"),
    ("QDRANT_COLLECTION_FORMATION", "research"),
    ("QDRANT_COLLECTION_OTA_INSIGHTS", "ota"),
])
def test_configured_versioned_collection_uses_corpus_profile(
    monkeypatch: pytest.MonkeyPatch, env: str, corpus: str
) -> None:
    collection = f"{env.removeprefix('QDRANT_COLLECTION_').lower()}_v2"
    monkeypatch.setenv(env, collection)
    assert profile_for_collection(collection).corpus == corpus


def test_parallel_retrieve_registers_six_paths() -> None:
    state: graph_mod.RAGGraphState = {"query": "maize Kenya"}
    calls: list[str] = []

    def _track(name: str, payload: list[dict[str, Any]]):
        def _fn(_state: graph_mod.RAGGraphState) -> list[dict[str, Any]]:
            calls.append(name)
            return payload

        return _fn

    with (
        mock.patch.object(graph_mod, "_retrieve_news", _track("news", [_hit("news_article")])),
        mock.patch.object(
            graph_mod,
            "_retrieve_academic_papers",
            _track("academic_papers", [_hit("academic_article")]),
        ),
        mock.patch.object(
            graph_mod, "_retrieve_policies", _track("policies", [_hit("policy_document")])
        ),
        mock.patch.object(
            graph_mod,
            "_retrieve_public_reports",
            _track("public_reports", [_hit("public_report")]),
        ),
        mock.patch.object(
            graph_mod,
            "_retrieve_formation",
            _track("formation", [_hit("agricultural_practise")]),
        ),
        mock.patch.object(graph_mod, "_retrieve_ota", _track("ota", [_hit("ota_insight")])),
        mock.patch.object(graph_mod, "_use_legacy_research_collection", return_value=False),
    ):
        out = graph_mod.node_parallel_retrieve(state)

    assert set(calls) == {
        "news",
        "academic_papers",
        "policies",
        "public_reports",
        "formation",
        "ota",
    }
    assert len(out["vector_news_results"]) == 1
    assert len(out["vector_academic_papers_results"]) == 1
    assert len(out["vector_policies_results"]) == 1
    assert len(out["vector_public_reports_results"]) == 1
    assert len(out["vector_formation_results"]) == 1
    assert len(out["vector_ota_results"]) == 1
    assert len(out["vector_results"]) == 6
    assert len(out["vector_academic_results"]) == 4  # deprecated concat of four dense corpora


@pytest.mark.parametrize(
    ("fn_name", "env_key", "default_coll", "doc_kind"),
    [
        ("_retrieve_academic_papers", "QDRANT_COLLECTION_ACADEMIC_PAPERS", "academic_papers", "academic_article"),
        ("_retrieve_policies", "QDRANT_COLLECTION_POLICIES", "policies", "policy_document"),
        ("_retrieve_public_reports", "QDRANT_COLLECTION_PUBLIC_REPORTS", "public_reports", "public_report"),
        ("_retrieve_formation", "QDRANT_COLLECTION_FORMATION", "formation", "agricultural_practise"),
    ],
)
def test_dense_retrieve_uses_collection_and_doc_kind(
    monkeypatch: pytest.MonkeyPatch,
    fn_name: str,
    env_key: str,
    default_coll: str,
    doc_kind: str,
) -> None:
    monkeypatch.delenv(env_key, raising=False)
    seen: dict[str, Any] = {}

    def _fake_retrieve(state, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return [_hit(doc_kind)]

    with mock.patch.object(graph_mod, "_vector_retrieve_for_corpus", side_effect=_fake_retrieve):
        fn = getattr(graph_mod, fn_name)
        out = fn({"query": "q", "decomposition": {}})

    assert seen["collection_env"] == env_key
    assert seen["default_collection"] == default_coll
    build = seen["build_kwargs"]
    kw = build({"query": "q"}, {})
    assert kw["doc_kind"] == doc_kind
    assert kw["vector_search_mode"] == "dense_named"
    assert out and out[0]["_context_kind"] in {
        "academic",
        "policy",
        "public_report",
        "formation",
    }


def test_one_failing_corpus_does_not_wipe_others() -> None:
    state: graph_mod.RAGGraphState = {"query": "q"}

    def _boom(_state: graph_mod.RAGGraphState) -> list[dict[str, Any]]:
        raise RuntimeError("collection missing")

    with (
        mock.patch.object(graph_mod, "_retrieve_news", return_value=[_hit("news_article")]),
        mock.patch.object(graph_mod, "_retrieve_academic_papers", side_effect=_boom),
        mock.patch.object(graph_mod, "_retrieve_policies", return_value=[_hit("policy_document")]),
        mock.patch.object(graph_mod, "_retrieve_public_reports", return_value=[]),
        mock.patch.object(graph_mod, "_retrieve_formation", return_value=[]),
        mock.patch.object(graph_mod, "_retrieve_ota", return_value=[_hit("ota_insight")]),
        mock.patch.object(graph_mod, "_use_legacy_research_collection", return_value=False),
    ):
        out = graph_mod.node_parallel_retrieve(state)

    assert len(out["vector_news_results"]) == 1
    assert out["vector_academic_papers_results"] == []
    assert len(out["vector_policies_results"]) == 1
    assert len(out["vector_ota_results"]) == 1
    assert out["retrieval_partial"] is True
    assert out["corpus_selection"]["results"]["academic_papers"]["status"] == "error"
    assert out["corpus_selection"]["results"]["public_reports"]["status"] == "empty"


def test_slow_corpus_does_not_hold_ready_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    release = Event()
    finished = Event()

    def slow(_state):
        release.wait(0.75)
        finished.set()
        return [_hit("academic_article")]

    monkeypatch.setenv("RAG_CORPUS_RETRIEVE_TIMEOUT_S", "0.15")
    monkeypatch.setattr(graph_mod, "_retrieve_academic_papers", slow)
    monkeypatch.setattr(graph_mod, "_retrieve_news", lambda state: [_hit("news_article")])
    monkeypatch.setattr(graph_mod, "_use_legacy_research_collection", lambda: False)
    for name in ("policies", "public_reports", "formation", "ota"):
        monkeypatch.setattr(graph_mod, f"_retrieve_{name}", lambda state: [])

    try:
        out = graph_mod.node_parallel_retrieve({"query": "maize Kenya"})
        assert not finished.is_set()
        assert out["retrieval_partial"] is True
        assert len(out["vector_news_results"]) == 1
        assert out["vector_academic_papers_results"] == []
    finally:
        release.set()


def test_merge_emits_distinct_context_kinds() -> None:
    state: graph_mod.RAGGraphState = {
        "bq_results": [],
        "vector_news_results": [graph_mod._tag_vector(_hit("news_article"), "news")],
        "vector_academic_papers_results": [
            graph_mod._tag_vector(_hit("academic_article"), "academic")
        ],
        "vector_policies_results": [graph_mod._tag_vector(_hit("policy_document"), "policy")],
        "vector_public_reports_results": [
            graph_mod._tag_vector(_hit("public_report"), "public_report")
        ],
        "vector_formation_results": [
            graph_mod._tag_vector(_hit("agricultural_practise"), "formation")
        ],
        "vector_ota_results": [graph_mod._tag_vector(_hit("ota_insight"), "ota_insight")],
    }
    merged = graph_mod.node_merge(state)["merged_context"]
    kinds = {m["_context_kind"] for m in merged}
    assert kinds == {
        "news",
        "academic",
        "policy",
        "public_report",
        "formation",
        "ota_insight",
    }
    assert any(m["content"].startswith("[News]") for m in merged)
    assert any(m["content"].startswith("[Academic") for m in merged)
    assert any(m["content"].startswith("[Policy") for m in merged)
    assert any(m["content"].startswith("[Public report") for m in merged)
    assert any(m["content"].startswith("[Formation") for m in merged)
    assert any(m["content"].startswith("[OTA") for m in merged)


def test_policy_hits_survive_a_later_query_view_timing_out(monkeypatch: pytest.MonkeyPatch) -> None:
    release = Event()
    seen = []
    hit = _hit("policy_document")

    def search(query, **kwargs):
        seen.append(query)
        if len(seen) == 1:
            return [hit]
        release.wait(1)
        return []

    monkeypatch.setenv("RAG_CORPUS_RETRIEVE_TIMEOUT_S", "0.15")
    monkeypatch.setattr(graph_mod, "_use_legacy_research_collection", lambda: False)
    for name in ("news", "academic_papers", "public_reports", "formation", "ota"):
        monkeypatch.setattr(graph_mod, f"_retrieve_{name}", lambda state: [])
    try:
        with mock.patch.object(graph_mod, "VectorRetriever") as retriever:
            retriever.return_value.last_index_health = None
            retriever.return_value.retrieve.side_effect = search
            out = graph_mod.node_parallel_retrieve({
                "query": "How did agricultural policies affect livelihoods in Ghana?",
                "decomposition": {"evidence_topics": ["farmer livelihoods"]},
            })
        assert out.get("retrieval_partial") is True
        assert [item["content"] for item in out["vector_policies_results"]] == [hit["content"]]
        assert out["corpus_selection"]["results"]["policies"]["status"] == "timeout"
    finally:
        release.set()


def test_dense_corpus_soft_fails_on_retrieve_error() -> None:
    with mock.patch.object(
        graph_mod, "_vector_retrieve_for_corpus", side_effect=RuntimeError("gone")
    ):
        out = graph_mod._retrieve_public_reports({"query": "q", "decomposition": {}})
    assert out == []
