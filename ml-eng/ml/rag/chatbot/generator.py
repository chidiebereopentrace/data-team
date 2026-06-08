"""
Generator node: takes query + reranked context and produces the final answer (LLM).

Uses the configured chat backend (OpenRouter, LM Studio, or Hugging Face router) via ``llm_chat``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from ml.rag.llm_chat import llm_chat_complete, llm_configured, llm_default_timeout_s, llm_model_id

from ml.rag.chat_history import normalize_messages, truncate_chat_history
from ml.rag.chat_memory import (
    build_memory_prompt_block,
    default_summary_max_chars,
    default_verbatim_max_chars,
)
from ml.rag.text_processors.preprocess.bibliographic_metadata import format_academic_citation


def _build_prompt(
    query: str,
    context_block: str,
    decomposition: dict[str, Any] | None = None,
    memory_block: str = "",
    audience_instructions: str = "",
) -> list[dict[str, str]]:
    system = (
        "You are an agricultural advisory assistant for OpenTrace stakeholders (government, NGOs, "
        "agribusiness, finance, farmers). Write clear prose for decision-makers — not a database console. "
        "Answer ONLY using facts in the Context below from retrieved OpenTrace sources: [News], [Academic], "
        "[Policy], [Public report], and [Structured data] (tabular facts already extracted from OpenTrace data). "
        "For [Academic] snippets, cite the source when metadata is present (authors, year, title, journal, DOI). "
        "Cite specific numbers or regions when the context supports them. "
        "If the context does not fully answer the question, say so and summarize what the sources do show. "
        "Do not invent citations, statistics, or datasets. "
        "Never output SQL, query code, table DDL, pipeline steps, or instructions to run BigQuery. "
        "Never mention bigquery-public-data or other external warehouses. "
        "Do not suggest example queries the user should run."
    )
    intent_tone = ""
    if decomposition:
        intent = str(decomposition.get("intent") or "").strip().lower()
        if intent == "predictive":
            intent_tone = (
                " The user's primary intent is forward-looking: clearly separate what the context shows "
                "from speculation; state uncertainty and limits of the data; avoid presenting guesses as facts."
            )
        elif intent == "diagnostic":
            intent_tone = (
                " The user's primary intent asks why or what drives outcomes: do not claim causation "
                "unless the context explicitly supports it; distinguish correlation from causation."
            )
    if intent_tone:
        system = system + intent_tone

    if audience_instructions:
        system = system + "\n\nClient-provided audience / tone guidance (follow where it does not conflict with the grounding rules above):\n" + audience_instructions[:3000]

    mb = (memory_block.strip() + "\n\n") if memory_block.strip() else ""
    user = f"{mb}Context:\n{context_block}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _resolve_memory_block(**kwargs: Any) -> str:
    conv_summary = kwargs.get("conversation_summary")
    recent_turns = kwargs.get("recent_turns")
    raw_history = kwargs.get("chat_history")
    block = ""
    if isinstance(recent_turns, list) or isinstance(conv_summary, str):
        s = (conv_summary if isinstance(conv_summary, str) else "").strip()
        rt = normalize_messages(recent_turns if isinstance(recent_turns, list) else None)
        block = build_memory_prompt_block(s, rt)
    elif isinstance(raw_history, list) and raw_history:
        rt = truncate_chat_history(raw_history)
        block = build_memory_prompt_block("", rt)
    cap = default_verbatim_max_chars() + default_summary_max_chars()
    if len(block) > cap:
        block = block[-cap:]
    return block


def _strip_sql_from_answer(text: str) -> str:
    """Remove SQL blocks the model may emit despite instructions (advisory UI only)."""
    if not text:
        return text
    out = re.sub(r"```(?:sql)?\s*[\s\S]*?```", "", text, flags=re.IGNORECASE)
    lines = []
    for line in out.splitlines():
        s = line.strip()
        if re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP)\b", s, re.IGNORECASE):
            continue
        if "bigquery-public-data" in line.lower():
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned or text.strip()


def _clean_answer(text: str) -> str:
    """Remove Llama chat template echoes and other non-answer artifacts from LLM output."""
    if not text:
        return text
    # Remove everything before the last [/INST] if the model echoed the prompt
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    # Remove leading Context:/Question: blocks the model may have echoed
    text = re.sub(r"^(Context:|Question:).*$", "", text, flags=re.MULTILINE | re.IGNORECASE).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _strip_sql_from_answer(text) or text.strip()


def _append_structured_citations(answer: str, context_items: list[dict[str, Any]]) -> str:
    """Append a clean Citations block when usable metadata exists in context_items."""
    if not context_items:
        return answer

    seen: set[str] = set()
    lines: list[str] = []

    for item in context_items:
        meta = item.get("metadata") or {}
        if not isinstance(meta, dict):
            continue

        kind = str(meta.get("doc_kind") or item.get("_context_kind") or "").lower()
        label = ""
        text = ""

        if kind in ("academic_article", "policy_document", "public_report"):
            cite = format_academic_citation(meta)
            if cite and cite not in seen:
                seen.add(cite)
                label = "[Academic]" if "academic" in kind else "[Policy/Public]"
                text = f"{label} {cite}"
        elif kind == "news_article":
            title = str(meta.get("title") or meta.get("source_file") or "").strip()
            src = str(meta.get("source") or meta.get("publisher") or "").strip()
            date = str(meta.get("published_at") or meta.get("date") or "").strip()[:10]
            if title:
                entry = f"{title} — {src}" if src else title
                if date:
                    entry += f" ({date})"
                if entry not in seen:
                    seen.add(entry)
                    text = f"[News] {entry}"
        elif kind in ("ota_insight", "ota_metric"):
            name = str(meta.get("metric_text") or meta.get("title") or meta.get("label") or "").strip()
            if name:
                entry = f"{name} — OpenTrace OTA"
                if entry not in seen:
                    seen.add(entry)
                    text = f"[OTA Insight] {entry}"

        if text:
            lines.append(text)

    if not lines:
        return answer

    citations_block = "\n\nCitations\n" + "\n".join(f"- {line}" for line in lines)
    return (answer.rstrip() + citations_block).strip()


def _call_llama(messages: list[dict[str, str]]) -> str:
    """Call configured LLM backend; never raises on HTTP errors."""
    gen_timeout = float(os.environ.get("RAG_GENERATE_TIMEOUT_S", "0") or 0) or llm_default_timeout_s()
    max_toks = int(os.environ.get("RAG_GENERATE_MAX_TOKENS", "1024") or 1024)
    # Default 0.5 gives noticeably more natural advisory prose than the old 0.3 while staying grounded.
    # Lower (0.2–0.3) for very conservative output; higher (0.55–0.65) only with stronger models.
    temperature = float(os.environ.get("RAG_GENERATE_TEMPERATURE", "0.5") or 0.5)
    return llm_chat_complete(
        messages,
        model=llm_model_id(),
        max_tokens=max_toks,
        temperature=temperature,
        timeout_s=gen_timeout,
    )


def generate(
    query: str,
    context_items: list[dict[str, Any]],
    **kwargs: Any,
) -> str:
    """
    Produce an answer from query and context.

    - If an LLM backend is configured (``RAG_LLM_BASE_URL`` or ``HF_API_TOKEN``), calls chat completions.
    - Otherwise, falls back to a simple debug-style answer that echoes the context.
    """
    decomposition = kwargs.get("decomposition")
    if not isinstance(decomposition, dict):
        decomposition = None

    memory_block = _resolve_memory_block(**kwargs)

    # Client-provided (AskADZA UI owns user profile / tone). Never auto-apply
    # the static stakeholder_prompts mapping here for the initial handoff.
    audience_instructions = (
        kwargs.get("audience_instructions")
        or kwargs.get("stakeholder_type")
        or ""
    ).strip()

    if not context_items:
        allow_ungrounded = os.environ.get("RAG_ALLOW_UNGROUNDED", "").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        )
        if allow_ungrounded:
            messages = _build_prompt(
                query,
                context_block="[No external context]",
                decomposition=decomposition,
                memory_block=memory_block,
                audience_instructions=audience_instructions,
            )
            llama_answer = _call_llama(messages)
            if llama_answer:
                cleaned = _clean_answer(llama_answer)
                return _append_structured_citations(cleaned, context_items)
        return (
            "I couldn't find relevant OpenTrace sources (news, research, policy, public reports, "
            "or structured agricultural data) for this question. Try naming a specific country or crop, "
            "or confirm the knowledge bases are loaded."
        )

    content_key = "content" if any("content" in c for c in context_items) else "text"
    ctx_budget = 6000
    if memory_block:
        ctx_budget = max(2000, 6000 - len(memory_block))
    context_block = "\n\n".join(
        (c.get(content_key) or c.get("text", str(c)))[:2000] for c in context_items
    )[:ctx_budget]

    messages = _build_prompt(
        query, context_block,
        decomposition=decomposition,
        memory_block=memory_block,
        audience_instructions=audience_instructions,
    )
    llama_answer = _call_llama(messages)
    if llama_answer:
        cleaned = _clean_answer(llama_answer)
        return _append_structured_citations(cleaned, context_items)

    if llm_configured():
        hint = (
            "[LLM generation failed — local server may have timed out or the request was cancelled. "
            f"Try RAG_LLM_TIMEOUT_S=300, RAG_GENERATE_MAX_TOKENS=1024, and RAG_LLM_RERANK=off. "
            f"Model id must match LM Studio: {llm_model_id()!r}. Showing retrieved context only.]\n\n"
        )
    else:
        hint = (
            "[LLM unavailable — set RAG_LLM_BASE_URL + RAG_LLM_API_KEY (OpenRouter, LM Studio, etc.) "
            "or HF_API_TOKEN for the Hugging Face router. Showing retrieved context only.]\n\n"
        )
    return hint + f"Context:\n{context_block[:3000]}\n\nQuery: {query}"
