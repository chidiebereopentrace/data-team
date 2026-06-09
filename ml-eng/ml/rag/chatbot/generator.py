"""
Generator node: takes query + reranked context and produces the final answer (LLM).

Uses the configured chat backend (OpenRouter, LM Studio, or Hugging Face router) via ``llm_chat``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from ml.rag.llm_chat import llm_chat_complete, llm_configured, llm_default_timeout_s, llm_model_id

from ml.rag.chat_history import normalize_messages, truncate_chat_history
from ml.rag.chat_memory import (
    build_memory_prompt_block,
    default_summary_max_chars,
    default_verbatim_max_chars,
)
from ml.rag.text_processors.preprocess.bibliographic_metadata import format_academic_citation

_BQ_TABLE_RE = re.compile(
    r"FROM\s+`?(?:[\w-]+\.)*([\w-]+)`?",
    re.IGNORECASE,
)
_SOURCE_REF_RE = re.compile(
    r"\[Source\s+(\d+)\]"
    r"|\[(?!\s*Source\s)(\d+)\]"
    r"|\bSource\s+(\d+)\b",
    re.IGNORECASE,
)
_VERBOSE_INLINE_REF_RE = re.compile(r"\[Source\s+(\d+)\s*\|[^\]]*\]", re.IGNORECASE)
_MODEL_SOURCES_APPENDIX_RE = re.compile(
    r"\n+(?:Sources|References|Bibliography)\s*:?\s*\n[\s\S]*\Z",
    re.IGNORECASE,
)
_BQ_FAILURE_MARKERS = (
    "[bq execution error",
    "[bq validation failed",
)
_BQ_ROW_HINT_KEYS = ("country", "product", "fnid", "planting_year", "harvest_year", "year", "region")
_BQ_PUBLIC_LABEL = "OpenTrace agricultural data"

_BQ_MIN_CHARS = 800


@dataclass(frozen=True)
class SourceRef:
    """A numbered context chunk passed to the LLM, with a preformatted citation line."""

    source_id: int
    item: dict[str, Any]
    citation_line: str


@dataclass(frozen=True)
class GenerationResult:
    """Structured generator output for API responses."""

    answer: str
    citations: list[dict[str, Any]]


def _generate_max_tokens() -> int:
    return int(os.environ.get("RAG_GENERATE_MAX_TOKENS", "2048") or 2048)


def _context_max_chars(memory_block: str = "") -> int:
    base = int(os.environ.get("RAG_GENERATE_CONTEXT_MAX_CHARS", "12000") or 12000)
    if memory_block.strip():
        return max(4000, base - len(memory_block))
    return base


def _chunk_max_chars() -> int:
    return int(os.environ.get("RAG_GENERATE_CHUNK_MAX_CHARS", "3000") or 3000)


def _citations_mode() -> str:
    raw = os.environ.get("RAG_CITATIONS_MODE", "referenced").strip().lower()
    if raw == "all":
        return "all"
    return "referenced"


def _append_sources_to_answer() -> bool:
    return os.environ.get("RAG_APPEND_SOURCES_TO_ANSWER", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def _source_kind(item: dict[str, Any]) -> str:
    meta = _item_metadata(item)
    kind = str(
        item.get("_context_kind")
        or (meta.get("doc_kind") if meta else "")
        or item.get("source")
        or ""
    ).lower()
    return kind


def _bq_table_from_meta(meta: dict[str, Any]) -> str:
    sql = str(meta.get("sql") or "")
    m = _BQ_TABLE_RE.search(sql)
    if m:
        return m.group(1)
    return ""


def _bq_row_hint(meta: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in _BQ_ROW_HINT_KEYS:
        val = meta.get(key)
        if val is not None and str(val).strip():
            parts.append(f"{key}={val}")
    return ", ".join(parts[:6])


def _item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("metadata")
    return raw if isinstance(raw, dict) else {}


def is_usable_context_item(item: dict[str, Any]) -> bool:
    """Return False for BQ failure/debug chunks that must not reach users."""
    meta = _item_metadata(item)
    if meta.get("validation_failed") or meta.get("execution_error"):
        return False
    raw = str(item.get("content") or item.get("text") or "").strip().lower()
    return not any(marker in raw for marker in _BQ_FAILURE_MARKERS)


def filter_context_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop BQ error and validation-failure chunks before generation."""
    return [item for item in items if is_usable_context_item(item)]


def _format_source_citation(item: dict[str, Any]) -> str | None:
    """Build a citation line for a context item (all source kinds)."""
    meta = _item_metadata(item)
    kind = _source_kind(item)

    if kind == "bigquery":
        if not is_usable_context_item(item):
            return None
        hint = _bq_row_hint(meta)
        if hint:
            return f"[Structured data] {_BQ_PUBLIC_LABEL} ({hint})"
        return f"[Structured data] {_BQ_PUBLIC_LABEL}"

    if kind in ("academic_article", "academic"):
        cite = format_academic_citation(meta)
        return f"[Academic] {cite}" if cite else None

    if kind in ("policy_document", "public_report", "policy", "public_report"):
        cite = format_academic_citation(meta)
        return f"[Policy/Public] {cite}" if cite else None

    if kind in ("news_article", "news"):
        title = str(meta.get("title") or meta.get("source_file") or "").strip()
        src = str(meta.get("source") or meta.get("publisher") or "").strip()
        date = str(meta.get("published_at") or meta.get("date") or "").strip()[:10]
        if title:
            entry = f"{title} — {src}" if src else title
            if date:
                entry += f" ({date})"
            return f"[News] {entry}"
        return None

    if kind in ("ota_insight", "ota_metric"):
        name = str(meta.get("metric_text") or meta.get("title") or meta.get("label") or "").strip()
        if name:
            return f"[OTA Insight] {name} — OpenTrace OTA"
        return None

    if kind == "web_wikipedia":
        title = str(meta.get("title") or "Wikipedia").strip()
        url = str(meta.get("url") or "").strip()
        return f"[Wikipedia] {title} — {url}" if url else f"[Wikipedia] {title}"

    if kind == "web_search":
        title = str(meta.get("title") or "Web source").strip()
        url = str(meta.get("url") or "").strip()
        return f"[Web] {title} — {url}" if url else f"[Web] {title}"

    return None


def _source_kind_display(kind: str) -> str:
    if kind == "bigquery":
        return "Structured data"
    if kind in ("academic_article", "academic"):
        return "Academic"
    if kind in ("policy_document", "public_report", "policy"):
        return "Policy/Public"
    if kind in ("news_article", "news"):
        return "News"
    if kind in ("ota_insight", "ota_metric"):
        return "OTA Insight"
    if kind == "web_wikipedia":
        return "Wikipedia"
    if kind == "web_search":
        return "Web search"
    return kind or "Context"


def _source_header_label(item: dict[str, Any], source_id: int) -> str:
    """Minimal context header — bibliographic detail lives in the Sources block only."""
    kind = _source_kind(item)
    return f"[Source {source_id}]\nType: {_source_kind_display(kind)}"


def _chunk_allocations(
    items: list[dict[str, Any]],
    budget: int,
    chunk_cap: int,
) -> list[int]:
    """Rank-weighted per-chunk char budgets; BQ chunks get a minimum floor."""
    n = len(items)
    if n == 0 or budget <= 0:
        return []

    weights = [max(1, n - i) for i in range(n)]
    total_weight = sum(weights)
    alloc = [min(chunk_cap, max(1, int(budget * w / total_weight))) for w in weights]

    for i, item in enumerate(items):
        if _source_kind(item) == "bigquery":
            alloc[i] = max(alloc[i], min(_BQ_MIN_CHARS, chunk_cap))

    while sum(alloc) > budget:
        idx = max(range(n), key=lambda j: alloc[j])
        if alloc[idx] <= 1:
            break
        alloc[idx] -= 1

    return alloc


def _build_context_block(
    context_items: list[dict[str, Any]],
    budget: int,
    chunk_cap: int,
) -> tuple[str, list[SourceRef]]:
    """
    Pack reranked chunks into a numbered, labeled context string and source registry.
    """
    if not context_items:
        return "", []

    content_key = "content" if any("content" in c for c in context_items) else "text"
    allocations = _chunk_allocations(context_items, budget, chunk_cap)

    parts: list[str] = []
    registry: list[SourceRef] = []
    used = 0

    for idx, (item, limit) in enumerate(zip(context_items, allocations, strict=False)):
        if used >= budget:
            break
        source_id = idx + 1
        raw = str(item.get(content_key) or item.get("text") or "")
        body = raw.strip()

        remaining = budget - used
        take = min(limit, chunk_cap, remaining)
        if take <= 0:
            continue

        header = _source_header_label(item, source_id)
        body_trunc = body[: max(0, take - len(header) - 2)]
        if not body_trunc.strip():
            continue

        block = f"{header}\n{body_trunc.strip()}"
        parts.append(block)
        used += len(block) + 2

        cite_line = _format_source_citation(item)
        if cite_line:
            registry.append(SourceRef(source_id=source_id, item=item, citation_line=cite_line))

    return "\n\n".join(parts), registry


def _strip_model_sources_appendix(text: str) -> str:
    """Remove model-written Sources/References blocks that poison footnote extraction."""
    if not text:
        return text
    return _MODEL_SOURCES_APPENDIX_RE.sub("", text).strip()


def _answer_for_citation_extraction(answer: str) -> str:
    """Prose-only text used to detect which footnote numbers the model actually cited."""
    return _normalize_inline_citations(_strip_model_sources_appendix(answer))


def _normalize_inline_citations(text: str) -> str:
    """Collapse verbose model citations to Wikipedia-style footnote numbers [N]."""
    if not text:
        return text
    text = _VERBOSE_INLINE_REF_RE.sub(lambda m: f"[{m.group(1)}]", text)
    text = re.sub(r"\[Source\s+(\d+)\]", r"[\1]", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\[)\bSource\s+(\d+)\b(?!\])", r"[\1]", text, flags=re.IGNORECASE)
    return text


def extract_referenced_source_ids(answer: str) -> set[int]:
    """Return source IDs cited inline in answer prose ([N], [Source N], or Source N)."""
    normalized = _answer_for_citation_extraction(answer)
    ids: set[int] = set()
    for m in _SOURCE_REF_RE.finditer(normalized):
        for g in m.groups():
            if g:
                ids.add(int(g))
                break
    return ids


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
        "Answer ONLY using facts in the Context below from numbered OpenTrace sources "
        "(each chunk is labeled [Source N] with a Type line). "
        "Synthesize evidence across all numbered sources — do not rely on a single snippet. "
        "When stating a specific fact, number, or claim from the context, cite it inline as a bare "
        "footnote number like [1] or [5] — the same N as [Source N] in the Context. "
        "Never put author names, titles, source types, or context headers in the answer body; "
        "do not use (Author et al., year) citations — use footnote numbers only. "
        "Do not output a Sources, References, or Bibliography section; the system appends one. "
        "Example: 'Rice yields rose in the north.[3]' — not 'From [Source 3 | Academic | Author (2019)]'. "
        "Sources labeled Type: Wikipedia or Type: Web search are supplemental external context — "
        "prefer OpenTrace news, research, and structured data when available; treat web sources as "
        "partial background and state limits when relying on them. "
        "When the context supports it, write a substantive multi-paragraph answer "
        "(roughly 4–8 paragraphs for complex questions). "
        "Structure complex answers: (1) direct answer, (2) supporting evidence by theme, region, or time period, "
        "(3) brief limits or gaps. "
        "For compare or trend questions, organize by region or time period when the context provides that breakdown. "
        "Include specific numbers, dates, and regions when present in the context. "
        "Lead with a direct answer — never open with 'The context provided...', 'Unfortunately...', "
        "'Based on the context...', or similar meta-commentary. "
        "If evidence is partial, state limits briefly after the substantive answer. "
        "Do not invent sources, cite Source IDs not in the Context, or invent statistics. "
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
        elif intent in ("compare", "decision_support"):
            intent_tone = (
                " Prefer a fuller synthesis when multiple sources agree or contrast; "
                "organize evidence so decision-makers can compare regions, time periods, or drivers."
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
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    text = re.sub(r"^(Context:|Question:).*$", "", text, flags=re.MULTILINE | re.IGNORECASE).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_sql_from_answer(text) or text.strip()
    return _strip_model_sources_appendix(text)


def _citation_kind_normalized(kind: str) -> str:
    k = kind.lower()
    if k == "bigquery":
        return "structured_data"
    if k in ("news_article",):
        return "news"
    if k in ("academic_article",):
        return "academic"
    if k in ("policy_document", "public_report"):
        return "policy"
    if k in ("ota_insight", "ota_metric"):
        return "ota"
    return k or "context"


def _citation_url(kind: str, meta: dict[str, Any]) -> str | None:
    k = kind.lower()
    if k in ("web_wikipedia", "web_search"):
        url = str(meta.get("url") or "").strip()
        return url or None
    if k in ("news_article", "news"):
        for key in ("url", "link", "source_url"):
            url = str(meta.get(key) or "").strip()
            if url.startswith("http"):
                return url
    if k in ("academic_article", "academic", "policy_document", "policy", "public_report"):
        doi = str(meta.get("doi") or "").strip()
        if doi.startswith("http"):
            return doi
        if doi:
            return f"https://doi.org/{doi.lstrip('doi:').strip()}"
    return None


def _source_ref_to_citation_dict(ref: SourceRef) -> dict[str, Any]:
    meta = _item_metadata(ref.item)
    kind = _source_kind(ref.item)
    return {
        "id": ref.source_id,
        "kind": _citation_kind_normalized(kind),
        "text": ref.citation_line,
        "url": _citation_url(kind, meta),
    }


def _referenced_source_refs(answer: str, source_registry: list[SourceRef]) -> list[SourceRef]:
    if not source_registry:
        return []
    mode = _citations_mode()
    if mode == "referenced":
        cited_ids = extract_referenced_source_ids(answer)
        if not cited_ids:
            return []
        return [r for r in source_registry if r.source_id in cited_ids]
    return list(source_registry)


def referenced_citations(answer: str, source_registry: list[SourceRef]) -> list[dict[str, Any]]:
    """Build structured citation objects for referenced (or all) sources."""
    prose = _strip_model_sources_appendix(answer)
    refs = _referenced_source_refs(prose, source_registry)
    return [_source_ref_to_citation_dict(r) for r in refs]


def _append_structured_citations(answer: str, source_registry: list[SourceRef]) -> str:
    """Append Sources block for referenced (or all) sources with bibliographic detail."""
    answer = _strip_model_sources_appendix(answer)
    cites = referenced_citations(answer, source_registry)
    if not cites:
        return answer
    lines = [f"{c['id']}. {c['text']}" for c in cites]
    block = "\n\nSources\n" + "\n".join(lines)
    return (answer.rstrip() + block).strip()


def _finalize_generation_result(
    answer: str,
    source_registry: list[SourceRef],
) -> GenerationResult:
    prose = _strip_model_sources_appendix(answer)
    citations = referenced_citations(prose, source_registry)
    if _append_sources_to_answer() and citations:
        lines = [f"{c['id']}. {c['text']}" for c in citations]
        prose = (prose.rstrip() + "\n\nSources\n" + "\n".join(lines)).strip()
    return GenerationResult(answer=prose, citations=citations)


def _call_llama(messages: list[dict[str, str]]) -> str:
    """Call configured LLM backend; never raises on HTTP errors."""
    gen_timeout = float(os.environ.get("RAG_GENERATE_TIMEOUT_S", "0") or 0) or llm_default_timeout_s()
    max_toks = _generate_max_tokens()
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
) -> GenerationResult:
    """
    Produce an answer from query and context.

    - If an LLM backend is configured (``RAG_LLM_BASE_URL`` or ``HF_API_TOKEN``), calls chat completions.
    - Otherwise, falls back to a simple debug-style answer that echoes the context.
    """
    decomposition = kwargs.get("decomposition")
    if not isinstance(decomposition, dict):
        decomposition = None

    memory_block = _resolve_memory_block(**kwargs)

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
                cleaned = _normalize_inline_citations(_clean_answer(llama_answer))
                return _finalize_generation_result(cleaned, [])
        return GenerationResult(
            answer=(
                "I couldn't find relevant OpenTrace sources (news, research, policy, public reports, "
                "or structured agricultural data) for this question. Try naming a specific country or crop, "
                "or confirm the knowledge bases are loaded."
            ),
            citations=[],
        )

    usable_context = filter_context_items(context_items)
    ctx_budget = _context_max_chars(memory_block)
    chunk_cap = _chunk_max_chars()
    context_block, source_registry = _build_context_block(usable_context, ctx_budget, chunk_cap)

    messages = _build_prompt(
        query,
        context_block,
        decomposition=decomposition,
        memory_block=memory_block,
        audience_instructions=audience_instructions,
    )
    llama_answer = _call_llama(messages)
    if llama_answer:
        cleaned = _normalize_inline_citations(_clean_answer(llama_answer))
        return _finalize_generation_result(cleaned, source_registry)

    if llm_configured():
        hint = (
            "[LLM generation failed — OpenRouter or configured API may have timed out or the request was cancelled. "
            f"Try RAG_LLM_TIMEOUT_S=300, RAG_GENERATE_TIMEOUT_S=300, RAG_GENERATE_MAX_TOKENS={_generate_max_tokens()}, "
            f"and RAG_LLM_RERANK=off. Model: {llm_model_id()!r}. Showing retrieved context only.]\n\n"
        )
    else:
        hint = (
            "[LLM unavailable — set RAG_LLM_BASE_URL + RAG_LLM_API_KEY (OpenRouter, etc.) "
            "or HF_API_TOKEN for the Hugging Face router. Showing retrieved context only.]\n\n"
        )
    return GenerationResult(
        answer=hint + f"Context:\n{context_block[:3000]}\n\nQuery: {query}",
        citations=[],
    )
