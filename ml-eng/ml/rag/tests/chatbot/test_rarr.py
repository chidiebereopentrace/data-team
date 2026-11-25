"""RARR answer mode: draft, revise against the graph's evidence, and leave the default path alone."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from ml.rag.chatbot import graph as graph_mod
from ml.rag.chatbot import rarr


def _news_item() -> dict[str, Any]:
    return {
        "content": "Maize yields declined in Kenya in 2024 because of late rains.",
        "source": "news",
        "_context_kind": "news",
        "metadata": {
            "doc_kind": "news_article",
            "geo_scope": "country",
            "geo_countries": "Kenya",
            "geo_country_primary": "Kenya",
            "published_at": "2025-06-01",
            "document_id": "news-doc-1",
            "domains": "yield",
            "title": "Kenya maize season",
        },
    }


class _FakeGraph:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.calls = 0

    def invoke(self, initial: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.calls += 1
        return {**initial, **self.state}


_EVIDENCE_STATE = {
    "answer": "Maize yields declined in Kenya in 2024 [Source 1].",
    "citations": [{"id": 1, "kind": "news", "text": "Kenya maize season"}],
    "answer_outcome": "documents",
    "reranked_context": [_news_item()],
    "acf_band": "moderate",
    "acf_score": 60,
}


def _stub_llm(monkeypatch: pytest.MonkeyPatch, replies: dict[str, str]) -> list[str]:
    seen: list[str] = []

    def fake(messages: list[dict[str, str]], **kw: Any) -> str:
        seen.append(kw["purpose"])
        return replies.get(kw["purpose"], "")

    monkeypatch.setattr(rarr, "llm_chat_complete", fake)
    return seen


def test_statements_skip_headings_and_list_markers() -> None:
    text = "## Overview:\n1. Maize output rose 4% in 2023 [Source 1]. - Sorghum is drought tolerant.\n2023 rains were poor."
    assert rarr.statements(text) == [
        "Maize output rose 4% in 2023 [Source 1].",
        "Sorghum is drought tolerant.",
        "2023 rains were poor.",
    ]


def test_score_without_evidence_is_low_not_a_refusal() -> None:
    rows = rarr.statement_rows("Teff is grown mostly in Ethiopia. It tolerates waterlogged soils.", set())
    acf = rarr.score_revision(
        rows, [], query="where is teff grown", decomposition=None, reference_date=date(2026, 11, 25)
    )
    assert (acf.band, acf.score, len(rows)) == ("no_evidence", 0, 2)
    assert "model knowledge" in acf.explanation


def test_revise_cites_evidence_and_scales_score_by_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_llm(
        monkeypatch,
        {"rarr_revise": "Maize yields fell in Kenya in 2024 [Source 1]. Farmers often replant after late rains."},
    )
    revision = rarr.revise_with_evidence(
        "why did maize yields fall in Kenya",
        "Maize yields in Kenya rose in 2024. Farmers often replant after late rains.",
        dict(_EVIDENCE_STATE),
        reference_date=date(2026, 11, 25),
    )
    assert revision is not None
    assert [r["verified"] for r in revision.statements] == [True, False]
    assert revision.trace["revise_called"] is True
    assert [s["id"] for s in revision.trace["evidence_sources"]] == [1]
    assert [c["id"] for c in revision.citations] == [1]
    assert revision.acf.components["coverage"] == 0.5
    assert revision.acf.score == round(revision.acf.components["evidence_score"] * 0.5)


def test_revise_without_evidence_keeps_draft_and_scores_low(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_llm(monkeypatch, {})
    revision = rarr.revise_with_evidence(
        "what is fonio", "Fonio is a West African millet.", {"reranked_context": []}
    )
    assert revision is not None
    assert revision.answer == "Fonio is a West African millet.\n\n" + rarr.NO_EVIDENCE_NOTE
    assert revision.citations == []
    assert (revision.acf.band, revision.acf.score) == ("no_evidence", 0)
    assert seen == []


def test_default_mode_never_drafts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeGraph(dict(_EVIDENCE_STATE))
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: fake)
    seen = _stub_llm(monkeypatch, {"rarr_draft": "draft", "rarr_revise": "revised"})
    out = graph_mod.run_rag("why did maize yields fall in Kenya")
    assert seen == []
    assert out["answer"] == _EVIDENCE_STATE["answer"]
    assert "rarr" not in out and "answer_mode" not in out


def test_rarr_mode_replaces_answer_and_keeps_evidence_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeGraph(dict(_EVIDENCE_STATE))
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: fake)
    seen = _stub_llm(
        monkeypatch,
        {
            "rarr_draft": "Maize yields in Kenya rose in 2024.",
            "rarr_revise": "Maize yields fell in Kenya in 2024 [Source 1].",
        },
    )
    out = graph_mod.run_rag("why did maize yields fall in Kenya", answer_mode="rarr")
    assert fake.calls == 1
    assert sorted(seen) == ["rarr_draft", "rarr_revise"]
    assert out["answer"] == "Maize yields fell in Kenya in 2024 [Source 1]."
    trace = out["rarr"]
    assert trace["status"] == "revised"
    assert trace["evidence_answer"] == _EVIDENCE_STATE["answer"]
    assert trace["evidence_acf"]["acf_score"] == 60
    assert trace["draft"] == "Maize yields in Kenya rose in 2024."
    assert trace["revise_raw"] == "Maize yields fell in Kenya in 2024 [Source 1]."
    assert trace["draft_ms"] is not None and trace["graph_ms"] is not None
    assert [r["verified"] for r in trace["statements"]] == [True]
    assert out["answer_mode"] == "rarr"


def test_unknown_source_numbers_do_not_count_as_verified() -> None:
    rows = rarr.statement_rows("Yields fell sharply in 2024 [Source 7].", {1, 2})
    assert rows[0]["verified"] is False and rows[0]["unknown_sources"] == [7]


def test_rarr_mode_leaves_non_retrieval_routes_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeGraph({"answer": "I am ADZA.", "answer_outcome": ""})
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: fake)
    _stub_llm(monkeypatch, {"rarr_draft": "draft", "rarr_revise": "revised"})
    out = graph_mod.run_rag("who are you", answer_mode="rarr")
    assert out["answer"] == "I am ADZA."
    assert "answer_mode" not in out
    assert out["rarr"]["status"] == "skipped"


def test_rarr_mode_answers_clarify_route_from_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeGraph(
        {"answer": "I need a bit more detail.", "answer_outcome": "", "task_mode": "clarify"}
    )
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: fake)
    seen = _stub_llm(monkeypatch, {"rarr_draft": "Cucumbers in Lagos markets sell for about N1,500 per kg."})
    out = graph_mod.run_rag("whats the price of cucumber in nigeria now", answer_mode="rarr")
    assert seen == ["rarr_draft"]
    assert out["answer"].startswith("Cucumbers in Lagos markets")
    assert rarr.NO_EVIDENCE_NOTE in out["answer"]
    assert (out["acf_band"], out["acf_score"]) == ("no_evidence", 0)
    assert out["rarr"]["evidence_answer"] == "I need a bit more detail."


def test_rarr_mode_falls_back_when_draft_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeGraph(dict(_EVIDENCE_STATE))
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: fake)
    _stub_llm(monkeypatch, {"rarr_revise": "revised"})
    out = graph_mod.run_rag("why did maize yields fall in Kenya", answer_mode="rarr")
    assert out["answer"] == _EVIDENCE_STATE["answer"]
    assert out["rarr"]["status"] == "skipped"


def test_env_switch_turns_rarr_on_for_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ANSWER_MODE", "rarr")
    fake = _FakeGraph(dict(_EVIDENCE_STATE))
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: fake)
    _stub_llm(
        monkeypatch,
        {"rarr_draft": "Maize yields in Kenya rose in 2024.", "rarr_revise": "Maize yields fell in Kenya in 2024 [Source 1]."},
    )
    out = graph_mod.run_rag("why did maize yields fall in Kenya")
    assert out["answer_mode"] == "rarr"


def test_explicit_evidence_mode_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ANSWER_MODE", "rarr")
    fake = _FakeGraph(dict(_EVIDENCE_STATE))
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: fake)
    seen = _stub_llm(monkeypatch, {"rarr_draft": "draft", "rarr_revise": "revised"})
    out = graph_mod.run_rag("why did maize yields fall in Kenya", answer_mode="evidence")
    assert seen == [] and "rarr" not in out


def test_empty_revision_retries_then_keeps_evidence_answer_not_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rarr, "_RETRY_PAUSE_S", 0.0)
    fake = _FakeGraph(dict(_EVIDENCE_STATE))
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: fake)
    seen = _stub_llm(monkeypatch, {"rarr_draft": "Maize yields in Kenya rose in 2024."})
    out = graph_mod.run_rag("why did maize yields fall in Kenya", answer_mode="rarr")
    assert seen.count("rarr_revise") == 3
    assert out["answer"] == _EVIDENCE_STATE["answer"]
    assert out["rarr"]["status"] == "skipped"


def test_revision_retry_recovers_from_a_failed_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rarr, "_RETRY_PAUSE_S", 0.0)
    replies = iter(["", "Maize yields fell in Kenya in 2024 [Source 1]."])

    def fake(messages: list[dict[str, str]], **kw: Any) -> str:
        return "Maize yields in Kenya rose in 2024." if kw["purpose"] == "rarr_draft" else next(replies)

    monkeypatch.setattr(rarr, "llm_chat_complete", fake)
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: _FakeGraph(dict(_EVIDENCE_STATE)))
    out = graph_mod.run_rag("why did maize yields fall in Kenya", answer_mode="rarr")
    assert out["answer"] == "Maize yields fell in Kenya in 2024 [Source 1]."
    assert out["rarr"]["revise_attempts"] == 2


def test_revision_time_is_reserved_from_the_request_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_left: dict[str, float | None] = {}

    class _SlowGraph(_FakeGraph):
        def invoke(self, initial: dict[str, Any], config: Any = None) -> dict[str, Any]:
            seen_left["graph"] = graph_mod.remaining_seconds(None)
            return super().invoke(initial, config)

    def fake(messages: list[dict[str, str]], **kw: Any) -> str:
        if kw["purpose"] == "rarr_revise":
            seen_left["revise"] = rarr.remaining_seconds(None)
            return "Maize yields fell in Kenya in 2024 [Source 1]."
        return "Maize yields in Kenya rose in 2024."

    monkeypatch.setenv("RAG_REQUEST_TIMEOUT_S", "100")
    monkeypatch.setenv("RAG_RARR_REVISE_RESERVE_S", "40")
    monkeypatch.setattr(rarr, "llm_chat_complete", fake)
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: _SlowGraph(dict(_EVIDENCE_STATE)))
    graph_mod.run_rag("why did maize yields fall in Kenya", answer_mode="rarr")
    assert 55 < seen_left["graph"] <= 60
    assert 95 < seen_left["revise"] <= 100


def test_revision_skipped_when_request_budget_is_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_REQUEST_TIMEOUT_S", "0.5")
    fake = _FakeGraph(dict(_EVIDENCE_STATE))
    monkeypatch.setattr(graph_mod, "_get_compiled_graph", lambda: fake)
    seen = _stub_llm(monkeypatch, {"rarr_draft": "draft text here", "rarr_revise": "revised [Source 1]."})
    out = graph_mod.run_rag("why did maize yields fall in Kenya", answer_mode="rarr")
    assert "rarr_revise" not in seen
    assert out["answer"] == _EVIDENCE_STATE["answer"]
    assert "left for revision" in out["rarr"]["reason"]

