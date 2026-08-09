"""Decide whether session chat memory should be injected for the current query."""

from __future__ import annotations

import re
from typing import Any

from ml.rag.chat_history import normalize_messages

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "to",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "with",
        "from",
        "as",
        "by",
        "at",
        "it",
        "its",
        "user",
        "assistant",
        "about",
        "what",
        "which",
        "when",
        "where",
        "how",
        "why",
        "na",
        "mba",
    }
)

# Too broad to count as topic continuity by themselves.
_BROAD_FACETS = frozenset(
    {
        "africa",
        "african",
        "global",
        "worldwide",
        "international",
        "agriculture",
        "agricultural",
    }
)


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) >= 3 and t.lower() not in _STOP
    }


def _memory_text(summary: str, recent_turns: list[dict[str, str]] | None) -> str:
    parts: list[str] = []
    s = (summary or "").strip()
    if s:
        parts.append(s)
    for m in normalize_messages(recent_turns):
        parts.append(m.get("content") or "")
    return "\n".join(parts)


def _facet_values(decomposition: dict[str, Any] | None, key: str) -> set[str]:
    if not isinstance(decomposition, dict):
        return set()
    raw = decomposition.get(key)
    if not isinstance(raw, list):
        return set()
    return {str(x).strip().lower() for x in raw if str(x).strip()}


def memory_relevant_for_query(
    query: str,
    summary: str = "",
    recent_turns: list[dict[str, str]] | None = None,
    decomposition: dict[str, Any] | None = None,
) -> bool:
    """
    Return True when prior chat memory should be injected for this query.

    Empty memory is treated as relevant (no-op). Unrelated prior topics return False.
    """
    mem = _memory_text(summary, recent_turns)
    if not mem.strip():
        return True

    q = (query or "").strip()
    if not q:
        return False

    mem_l = mem.lower()
    # Shared geography / entities from decomposition appearing in memory.
    for val in _facet_values(decomposition, "geography") | _facet_values(decomposition, "entities"):
        if len(val) >= 3 and val not in _BROAD_FACETS and val in mem_l:
            return True

    q_toks = _tokens(q)
    m_toks = _tokens(mem)
    if not q_toks:
        return False
    overlap = q_toks & m_toks
    # Need at least 2 shared content tokens, or ≥30% of query content tokens.
    if len(overlap) >= 2:
        return True
    if len(overlap) >= 1 and (len(overlap) / len(q_toks)) >= 0.3:
        return True
    return False


__all__ = ["memory_relevant_for_query"]
