"""Tests for chat-memory relevance gating."""

from __future__ import annotations

from ml.rag.chatbot.generator import _resolve_memory_block
from ml.rag.chatbot.memory_relevance import memory_relevant_for_query


def test_empty_memory_is_relevant() -> None:
    assert memory_relevant_for_query("maize yields Kenya", "", [], None) is True


def test_unrelated_nigeria_memory_not_relevant_to_igbo_yam() -> None:
    summary = "User discussed Nigeria fertilizer and maize markets in Abuja."
    recent = [
        {"role": "user", "content": "Tell me about Nigeria maize prices"},
        {"role": "assistant", "content": "Nigeria maize prices varied in Abuja markets."},
    ]
    q = "kedu obodo kacha ako ji na mba africa"
    dec = {"entities": ["Africa"], "geography": [], "domains": ["agriculture"]}
    assert memory_relevant_for_query(q, summary, recent, dec) is False
    block = _resolve_memory_block(
        query=q,
        conversation_summary=summary,
        recent_turns=recent,
        decomposition=dec,
    )
    assert block == ""


def test_related_memory_is_kept() -> None:
    summary = "User asked about yam production across West Africa."
    recent = [
        {"role": "user", "content": "Which countries grow the most yam?"},
        {"role": "assistant", "content": "Nigeria and Ghana are major yam producers."},
    ]
    dec = {"entities": ["yam"], "geography": [], "domains": []}
    assert memory_relevant_for_query(
        "yam production in africa",
        summary,
        recent,
        dec,
    ) is True
