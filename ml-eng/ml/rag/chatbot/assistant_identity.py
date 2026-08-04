"""
Assistant identity and meta-query handling for Ask ADZA / OpenTrace.

Detects identity and product questions so the RAG graph can short-circuit retrieval
and return clean, chat-UI-ready answers. Supports hybrid mode (static canonical answers
+ LLM persona for variants) controlled by RAG_META_RESPONSES.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    is_english_answer_lang,
    language_instruction,
)
from ml.rag.chatbot.plan_policy import instruction_for_category, plan_generation_addendum

# Canonical identity-only patterns (product/OpenTrace questions use product_knowledge.py)
_META_PATTERNS: tuple[str, ...] = (
    r"\bwho are you\b",
    r"\bwhat(?:'s| is) your name\b",
    r"\bwhat are you\b",
    r"\bwhat do you do\b",
    r"\bwhat are you doing\b",
    # French
    r"\bqui\s+(?:es[- ]tu|êtes[- ]vous|etes[- ]vous)\b",
    r"\bquel\s+est\s+(?:ton|votre)\s+nom\b",
    r"\bqu['’]est[- ]ce\s+que\s+tu\s+fais\b",
    # Swahili
    r"\bwewe\s+ni\s+nani\b",
    r"\bjina\s+lako\s+nani\b",
    r"\buna[- ]?fanya\s+nini\b",
    # Nigerian Pidgin
    r"\bwho\s+you\s+be\b",
    r"\bwetin\s+be\s+your\s+name\b",
    r"\bwetin\s+you\s+dey\s+do\b",
    # Igbo / Yoruba (high-frequency identity)
    r"\bonye\s+[ịi]\s+b[ụu]\b",
    r"\bta\s+ni\s+[ẹe]\b",
)

_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)


def is_meta_query(query: str) -> bool:
    """Return True if the query is about the assistant identity or OpenTrace product."""
    if not query or not query.strip():
        return False
    return bool(_META_RE.search(query))


def classify_meta_query(query: str) -> str | None:
    """
    Classify a meta query into a bucket for static answer selection or LLM routing.
    Returns one of: identity, name, role, or None.
    """
    if not query:
        return None
    q = query.lower()
    if (
        re.search(r"\bwho are you\b", q)
        or re.search(r"\bwhat are you\b", q)
        or re.search(r"\bqui\s+(?:es[- ]tu|êtes[- ]vous|etes[- ]vous)\b", q)
        or re.search(r"\bwewe\s+ni\s+nani\b", q)
        or re.search(r"\bwho\s+you\s+be\b", q)
        or re.search(r"\bonye\s+[ịi]\s+b[ụu]\b", q)
        or re.search(r"\bta\s+ni\s+[ẹe]\b", q)
    ):
        return "identity"
    if (
        re.search(r"\bwhat(?:'s| is) your name\b", q)
        or re.search(r"\bquel\s+est\s+(?:ton|votre)\s+nom\b", q)
        or re.search(r"\bjina\s+lako\s+nani\b", q)
        or re.search(r"\bwetin\s+be\s+your\s+name\b", q)
    ):
        return "name"
    if (
        re.search(r"\bwhat do you do\b|\bwhat are you doing\b", q)
        or re.search(r"\bqu['’]est[- ]ce\s+que\s+tu\s+fais\b", q)
        or re.search(r"\buna[- ]?fanya\s+nini\b", q)
        or re.search(r"\bwetin\s+you\s+dey\s+do\b", q)
    ):
        return "role"
    return None


# Static canonical answers (sourced from OpenTrace deck, chat-UI ready, no retrieval)
_STATIC_ANSWERS: dict[str, str | None] = {
    "identity": (
        "I am Ask ADZA, OpenTrace Africa's AI-powered advisory interface. "
        "I help you explore agricultural intelligence across Africa using plain-language questions. "
        "My answers are grounded in OpenTrace's integrated datasets — news, research, policy sources, "
        "and structured agricultural data — with transparency about confidence and limits. "
        "I am not a general-purpose chatbot; I am built for decision-makers who need evidence-based agricultural insight."
    ),
    "name": (
        "I am Ask ADZA, the natural-language interface for OpenTrace Africa. "
        "OpenTrace Africa builds Africa's agricultural intelligence layer."
    ),
    "role": (
        "I answer questions about African agriculture, food security, climate, markets, and related development data "
        "by pulling together evidence from OpenTrace's federated intelligence layer. "
        "I summarise what the sources show, cite where possible, and flag when evidence is partial or uncertain. "
        "For data-heavy questions I may also use structured OpenTrace datasets; I do not invent statistics or present projections as facts."
    ),
}


def static_meta_answer(bucket: str | None, query: str = "") -> str | None:
    """Return a static answer for canonical buckets when available."""
    if not bucket:
        return None
    text = _STATIC_ANSWERS.get(bucket)
    if text is None:
        return None
    return text.strip()


# Persona prompt for LLM path (hybrid/llm modes) on non-canonical identity questions
META_SYSTEM_PROMPT = (
    "You are Ask ADZA, the natural-language interface for OpenTrace Africa. "
    "OpenTrace Africa builds Africa's agricultural intelligence layer. "
    "You help users explore agricultural intelligence across Africa using plain-language questions. "
    "For questions about OpenTrace's mission, pillars (OFIA, ACF, Ask ADZA), partnerships, or product positioning, "
    "those are handled by a separate product knowledge path — focus here on who you are and what you do as an assistant. "
    "Websites: opentrace.africa, askadza.africa. Contact: contact@opentrace.africa."
)


def _env_mode() -> str:
    """Return the active meta response mode: hybrid (default), static, or llm."""
    raw = os.environ.get("RAG_META_RESPONSES", "hybrid").strip().lower()
    if raw in ("static", "llm"):
        return raw
    return "hybrid"


def generate_meta_answer(query: str, **kwargs: Any) -> str:
    """
    Produce a meta answer for identity questions.

    Hybrid mode (default): static canonical answers when available; otherwise LLM via persona prompt.
    Non-English queries skip static English answers and use the LLM + language mirror instruction.
    Respects RAG_META_RESPONSES env var.
    Optionally appends category tone and plan-tier guidance when provided.
    """
    bucket = classify_meta_query(query)
    mode = _env_mode()
    lang = detect_answer_language(query)

    # Static path — English only (canonical copy is English)
    if mode in ("hybrid", "static") and is_english_answer_lang(lang):
        static = static_meta_answer(bucket, query)
        if static:
            category = kwargs.get("category") or ""
            tone = instruction_for_category(category) if category else ""
            if tone:
                static = static.rstrip() + "\n\n" + tone
            return _append_footer(static)

    # LLM path (hybrid for variants / non-English, or llm mode)
    from ml.rag.chatbot.generator import _call_llama, _resolve_memory_block  # local import to avoid circular

    memory_block = _resolve_memory_block(**kwargs)
    category = (kwargs.get("category") or "").strip()
    plan_type = (kwargs.get("plan_type") or "").strip()
    tone = instruction_for_category(category) if category else ""
    plan_addendum = plan_generation_addendum(plan_type) if plan_type else ""

    system = META_SYSTEM_PROMPT + "\n\n" + language_instruction(lang)
    if tone:
        system = system + "\n\n" + tone
    if plan_addendum:
        system = system + "\n\n" + plan_addendum

    user = (memory_block.strip() + "\n\n" if memory_block.strip() else "") + f"Question: {query}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = _call_llama(messages, purpose="generate_meta")
    answer = raw.strip() if raw else "I am Ask ADZA, the OpenTrace agricultural advisory assistant."
    return _append_footer(answer)


# Static footer appended to every meta answer (product requirement)
META_ANSWER_FOOTER = "\n\nSource: OpenTrace Africa © 2026 | opentrace.africa"


def _append_footer(text: str) -> str:
    if not text:
        return text
    if "opentrace.africa" in text.lower():
        return text
    return text.rstrip() + META_ANSWER_FOOTER


__all__ = [
    "is_meta_query",
    "classify_meta_query",
    "static_meta_answer",
    "generate_meta_answer",
    "META_SYSTEM_PROMPT",
    "META_ANSWER_FOOTER",
]