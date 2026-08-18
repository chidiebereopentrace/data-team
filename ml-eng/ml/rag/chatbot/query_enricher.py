"""Merge elliptical / anaphoric follow-ups with prior conversation topic before decompose."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chat_history import normalize_messages
from ml.rag.chatbot.assistant_identity import is_meta_query
from ml.rag.chatbot.product_knowledge import is_product_query
from ml.rag.chatbot.query_gate import is_greeting_query

_ANAPHORA_RE = re.compile(
    r"\b("
    r"that\s+country|that\s+nation|that\s+place|those\s+countries|"
    r"the\s+same\s+crop|that\s+crop|same\s+(?:crop|commodity|country)|"
    r"there|for\s+that|about\s+that|and\s+for\s+\w+"
    r")\b",
    re.IGNORECASE,
)

_ELLIPTICAL_RE = re.compile(
    r"^\s*("
    r"country\s+is\s+\w+|"
    r"(?:the\s+)?country(?:\s+is|:)?\s+\w+|"
    r"what\s+about\s+[\w\s]+|"
    r"how\s+about\s+[\w\s]+|"
    r"same\s+for\s+[\w\s]+|"
    r"and\s+(?:in\s+)?[\w\s]{2,40}\??|"
    r"for\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\??"
    r")\s*$",
    re.IGNORECASE,
)

_SHORT_FOLLOWUP_RE = re.compile(
    r"^\s*("
    r"yes|no|ok|okay|"
    r"niger|kenya|nigeria|ethiopia|somalia|uganda|senegal|zambia|"
    r"maize|rice|cassava|wheat|coffee"
    r")\s*\.?\s*$",
    re.IGNORECASE,
)


def _prior_user_texts(summary: str, recent_turns: list[dict[str, Any]] | None) -> list[str]:
    texts: list[str] = []
    for m in normalize_messages(recent_turns):
        if str(m.get("role") or "").lower() == "user":
            c = str(m.get("content") or "").strip()
            if c:
                texts.append(c)
    s = (summary or "").strip()
    if s:
        texts.append(s)
    return texts


def _looks_elliptical(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if is_greeting_query(q) or is_meta_query(q) or is_product_query(q, None):
        return False
    if _ANAPHORA_RE.search(q):
        return True
    if _ELLIPTICAL_RE.match(q):
        return True
    if len(q.split()) <= 6 and _SHORT_FOLLOWUP_RE.match(q):
        return True
    if len(q.split()) <= 4 and not re.search(
        r"\b(what|which|how|why|when|show|give|compare|analy)\b", q, re.IGNORECASE
    ):
        return True
    return False


def enrich_query_with_memory(
    query: str,
    *,
    conversation_summary: str | None = None,
    recent_turns: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Return enriched query string when the user turn is elliptical / anaphoric.

    Keys: enriched_query, original_query, enriched (bool), prior_topic (str|None)
    """
    original = (query or "").strip()
    priors = _prior_user_texts(conversation_summary or "", recent_turns)
    prior_topic = priors[-1] if priors else None
    if not original or not prior_topic:
        return {
            "enriched_query": original,
            "original_query": original,
            "enriched": False,
            "prior_topic": prior_topic,
        }
    if is_greeting_query(original) or is_meta_query(original) or is_product_query(original, None):
        return {
            "enriched_query": original,
            "original_query": original,
            "enriched": False,
            "prior_topic": prior_topic,
        }
    if not _looks_elliptical(original):
        return {
            "enriched_query": original,
            "original_query": original,
            "enriched": False,
            "prior_topic": prior_topic,
        }
    prior_tokens = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", prior_topic)}
    cur_tokens = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", original)}
    if prior_tokens and len(prior_tokens & cur_tokens) >= max(2, len(prior_tokens) // 3):
        return {
            "enriched_query": original,
            "original_query": original,
            "enriched": False,
            "prior_topic": prior_topic,
        }

    enriched = f"{prior_topic.rstrip('.!?')}. Follow-up: {original}"
    return {
        "enriched_query": enriched,
        "original_query": original,
        "enriched": True,
        "prior_topic": prior_topic,
    }


__all__ = ["enrich_query_with_memory"]
