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

from ml.rag.chatbot.stakeholder_prompts import instruction_for_stakeholder

# Canonical meta-query patterns (case-insensitive, word-boundary aware)
_META_PATTERNS: tuple[str, ...] = (
    r"\bwho are you\b",
    r"\bwhat(?:'s| is) your name\b",
    r"\bwhat are you\b",
    r"\bwhat do you do\b",
    r"\bwhat are you doing\b",
    r"\btell me about opentrace\b",
    r"\bwhat is opentrace\b",
    r"\bwhat is ask adza\b",
    r"\bwho is ask adza\b",
    # Pillar explainers (LLM path in hybrid mode)
    r"\bwhat is ofia\b",
    r"\bwhat is acf\b",
    r"\bexplain.*confidence framework\b",
    r"\bexplain.*pillars?\b",
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
    Returns one of: identity, name, role, opentrace, ask_adza, pillar, or None.
    """
    if not query:
        return None
    q = query.lower()
    if re.search(r"\bwho are you\b", q) or re.search(r"\bwhat are you\b", q):
        return "identity"
    if re.search(r"\bwhat(?:'s| is) your name\b", q):
        return "name"
    if re.search(r"\bwhat do you do\b|\bwhat are you doing\b", q):
        return "role"
    if re.search(r"\btell me about opentrace\b|\bwhat is opentrace\b", q):
        return "opentrace"
    if re.search(r"\bwhat is ask adza\b|\bwho is ask adza\b", q):
        return "ask_adza"
    if re.search(r"\bwhat is ofia\b|\bwhat is acf\b|\bexplain.*confidence framework\b|\bexplain.*pillars?\b", q):
        return "pillar"
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
    "opentrace": (
        "OpenTrace Africa builds Africa's agricultural intelligence layer. Agricultural data exists across governments, "
        "research, climate systems, and markets, but it is fragmented and hard to use together. OpenTrace federates and "
        "harmonises those sources so decision-makers get usable intelligence, not just more data. "
        "The platform rests on four pillars: OFIA (federated infrastructure), ACF (confidence and transparency on every insight), "
        "data reconstruction (structured gap-filling that mirrors how agricultural systems behave), and predictive intelligence "
        "(forward-looking trends and risk signals, always confidence-weighted). Ask ADZA is the natural-language interface to that stack. "
        "Learn more at opentrace.africa or askadza.com."
    ),
    "ask_adza": (
        "Ask ADZA is OpenTrace's machine-learning-powered natural language interface. "
        "You ask questions in plain language and receive structured, validated answers backed by integrated agricultural intelligence — "
        "with reliability made explicit through the ADZA Confidence Framework (ACF), not uniform AI certainty."
    ),
    "pillar": None,  # handled by LLM persona in hybrid mode
}


def static_meta_answer(bucket: str | None, query: str = "") -> str | None:
    """Return a static answer for canonical buckets when available."""
    if not bucket:
        return None
    text = _STATIC_ANSWERS.get(bucket)
    if text is None:
        return None
    return text.strip()


# Persona prompt for LLM path (hybrid/llm modes) on non-canonical meta questions
META_SYSTEM_PROMPT = (
    "You are Ask ADZA, the natural-language interface for OpenTrace Africa. "
    "OpenTrace Africa builds Africa's agricultural intelligence layer. It integrates fragmented agricultural, climate, market, "
    "nutrition, and economic data into decision intelligence for governments, development partners, private sector, farmers and cooperatives, "
    "and agricultural entrepreneurs. "
    "OpenTrace does not generate speculative answers. Insights come from verified integrated datasets. Transparency and reliability are core principles. "
    "Every analytical output can carry an explicit confidence signal via the ADZA Confidence Framework (ACF). "
    "Four pillars: OFIA (OpenTrace Federated Intelligence Architecture) federates and harmonises data across global, regional, national, and community levels; "
    "enables consistent querying via Ask ADZA; does not own, enclose, or relicense third-party data. "
    "ACF triangulates evidence across global/continental, regional/national, and ground-level tiers; scores outputs 0-100 from tier coverage, alignment, freshness, and granularity; "
    "users see confidence bands and explanations. "
    "Data reconstruction uses structured methods to close gaps by mirroring natural agricultural patterns (seasons, climate, markets); more reconstruction lowers confidence. "
    "Predictive intelligence surfaces trend, scenario, and risk signals, always confidence-weighted; never present projections as facts. "
    "For identity or product questions, describe yourself as Ask ADZA and OpenTrace as above. "
    "Websites: opentrace.africa, askadza.com, askadza.africa. Contact: contact@opentrace.africa."
)


def _env_mode() -> str:
    """Return the active meta response mode: hybrid (default), static, or llm."""
    raw = os.environ.get("RAG_META_RESPONSES", "hybrid").strip().lower()
    if raw in ("static", "llm"):
        return raw
    return "hybrid"


def generate_meta_answer(query: str, **kwargs: Any) -> str:
    """
    Produce a meta answer for identity/product questions.

    Hybrid mode (default): static canonical answers when available; otherwise LLM via persona prompt.
    Respects RAG_META_RESPONSES env var.
    Optionally appends stakeholder tone instruction when stakeholder_type is provided.
    """
    bucket = classify_meta_query(query)
    mode = _env_mode()

    # Static path
    if mode in ("hybrid", "static"):
        static = static_meta_answer(bucket, query)
        if static:
            stakeholder = kwargs.get("stakeholder_type") or ""
            tone = instruction_for_stakeholder(stakeholder) if stakeholder else ""
            if tone:
                static = static.rstrip() + "\n\n" + tone
            return _append_footer(static)

    # LLM path (hybrid for variants, or llm mode)
    from ml.rag.chatbot.generator import _call_llama, _resolve_memory_block  # local import to avoid circular

    memory_block = _resolve_memory_block(**kwargs)
    audience = (kwargs.get("audience_instructions") or "").strip()
    stakeholder = (kwargs.get("stakeholder_type") or "").strip()
    tone = instruction_for_stakeholder(stakeholder) if stakeholder else ""

    system = META_SYSTEM_PROMPT
    if tone:
        system = system + "\n\n" + tone
    if audience:
        system = system + "\n\nClient-provided audience / tone guidance:\n" + audience[:3000]

    user = (memory_block.strip() + "\n\n" if memory_block.strip() else "") + f"Question: {query}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = _call_llama(messages)
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