"""Tests for corpus catalog heuristic gate/boost."""
from __future__ import annotations

from ml.rag.chatbot.corpus_catalog import (
    ALL_CORPUS_KEYS,
    select_corpora,
)


def test_router_off_returns_all(monkeypatch) -> None:
    monkeypatch.setenv("RAG_CORPUS_ROUTER", "off")
    sel = select_corpora({}, query="how to plant maize")
    assert sel.active == list(ALL_CORPUS_KEYS)
    assert sel.rationale == "router_off"


def test_formation_cues_boost_formation(monkeypatch) -> None:
    monkeypatch.delenv("RAG_CORPUS_ROUTER", raising=False)
    sel = select_corpora({}, query="how to plant maize for smallholder farmers")
    assert "formation" in sel.active
    assert sel.boosts["formation"] > sel.boosts["news"]
    assert "formation_cues" in sel.rationale


def test_policy_cues_boost_policies(monkeypatch) -> None:
    monkeypatch.delenv("RAG_CORPUS_ROUTER", raising=False)
    sel = select_corpora({}, query="what is the fertilizer subsidy policy in Kenya")
    assert "policies" in sel.active
    assert "policy_cues" in sel.rationale
    assert sel.boosts["policies"] >= 0.1


def test_farmers_soft_skips_academic(monkeypatch) -> None:
    monkeypatch.delenv("RAG_CORPUS_ROUTER", raising=False)
    sel = select_corpora(
        {"intent": "descriptive"},
        plan_type="Farmers",
        query="maize prices in Ghana",
    )
    assert "academic_papers" not in sel.active
    assert "formation" in sel.active
    assert "news" in sel.active
    assert len(sel.active) >= 3
    assert len(ALL_CORPUS_KEYS) - len(sel.active) <= 3


def test_never_empty(monkeypatch) -> None:
    monkeypatch.delenv("RAG_CORPUS_ROUTER", raising=False)
    sel = select_corpora({}, query="")
    assert sel.active
    assert set(sel.active) == set(ALL_CORPUS_KEYS)


def test_investment_cues_boost_ota(monkeypatch) -> None:
    monkeypatch.delenv("RAG_CORPUS_ROUTER", raising=False)
    sel = select_corpora(
        {"intent": "descriptive"},
        query="which country is best for agricultural investments in africa",
    )
    assert "ota" in sel.active
    assert "investment_decision_cues" in sel.rationale
    assert sel.boosts["ota"] >= 0.15


def test_integrated_plan_boosts_ota(monkeypatch) -> None:
    monkeypatch.delenv("RAG_CORPUS_ROUTER", raising=False)
    sel = select_corpora({}, plan_type="Integrated", query="maize outlook")
    assert "ota" in sel.active
    assert "plan_ota_boost" in sel.rationale
