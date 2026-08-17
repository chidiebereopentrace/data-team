"""
Generator node: takes query + reranked context and produces the final answer (LLM).

Uses the configured chat backend (OpenRouter, LM Studio, or Hugging Face router) via ``llm_chat``.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from ml.rag.llm_chat import llm_chat_complete, llm_configured, llm_default_timeout_s, llm_model_id
from ml.rag.chatbot.plan_policy import model_for_plan

from ml.rag.chat_history import normalize_messages, truncate_chat_history
from ml.rag.chat_memory import (
    build_memory_prompt_block,
    default_summary_max_chars,
    default_verbatim_max_chars,
)
from ml.rag.chatbot.acf_scoring import ACFResult, no_evidence_acf, score_cited_evidence
from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    is_english_answer_lang,
    language_instruction,
)
from ml.rag.chatbot.context_diversity import normalize_context_kind
from ml.rag.chatbot.export_intent import want_inline_citations
from ml.rag.chatbot.geo_regions import is_zone_label
from ml.rag.chatbot.memory_relevance import memory_relevant_for_query
from ml.rag.chatbot.plan_policy import instruction_for_category, plan_generation_addendum
from ml.rag.observability import observed_span, trace_elapsed_ms, update_current_span_metadata
from ml.rag.text_processors.preprocess.bibliographic_metadata import format_academic_citation

_DOC_TABLE_FIGURE_RE = re.compile(
    r"\b(?:tables?|figures?|figs?\.?|appendices|appendix)\s+"
    r"(?:[A-Z]?\d+(?:\.\d+)*|[IVXLC]+)\b",
    re.IGNORECASE,
)

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
# Sprint 1, Week 3 (direct-answer-first): deterministic backstop for the prompt rule
# in _build_prompt. Even with the "no preamble" system instruction, the model
# occasionally still opens with an academic/meta-commentary hedge (the corpus is
# heavily academic). This regex matches ONLY clear connective preambles at the very
# start of the answer so we can strip them and let the substantive answer lead.
# It deliberately does NOT touch content-bearing openings (e.g. "This study examines").
_PREAMBLE_OPENER_RE = re.compile(
    r"^\s*(?:"
    r"based on the (?:provided |given )?context|"
    r"according to the (?:provided |given )?context|"
    r"the (?:provided |given )?context(?: above| provided)?"
    r"(?: clearly)?(?: shows| indicates| suggests| states| reveals| highlights| provides| mentions)|"
    r"it is important to note|"
    r"it is worth noting|"
    r"it should be noted|"
    r"the evidence suggests|"
    r"unfortunately"
    r")\b[\s,:;\u2014-]*(?:that\s+)?",
    re.IGNORECASE,
)
_MAX_PREAMBLE_UNWIND = 3

_BQ_FAILURE_MARKERS = (
    "[bq execution error",
    "[bq validation failed",
    "[bq no_project",
    "[bq no_valid_sql",
)
_BQ_ROW_HINT_KEYS = (
    "country",
    "country_name",
    "product",
    "product_name",
    "element",
    "fnid",
    "planting_year",
    "harvest_year",
    "year",
    "region",
)
_BQ_PUBLIC_LABEL = "OpenTrace agricultural data"

_RANKING_QUERY_RE = re.compile(
    r"\b("
    r"highest|lowest|top\s+\d+|bottom\s+\d+|"
    r"which\s+(?:\w+\s+){0,3}countr(?:y|ies)|"
    r"rank(?:ing|ed)?|most\s+(?:produced|production)|"
    r"produces?\s+the\s+most|the\s+most\s+\w+|"
    r"least\s+(?:produced|production)|"
    r"largest|smallest|biggest"
    r")\b",
    re.IGNORECASE,
)

_NUMERIC_QUANTITY_RE = re.compile(
    r"\b("
    r"how\s+much|how\s+many|what\s+is\s+the|what\s+was\s+the|"
    r"total|amount|volume|tonnes?|tons?|"
    r"yield|production|output|price|prices|gdp|hdi|"
    r"percent|percentage|\%|rate|rates|average|avg|mean|"
    r"trend|changed\s+by|increase|decrease|growth|decline|"
    r"population\s+count|people\s+in|phase\s+3"
    r")\b",
    re.IGNORECASE,
)

_QUALITATIVE_ONLY_RE = re.compile(
    r"\b("
    r"why|explain|what\s+does\s+the\s+policy|impact\s+of|"
    r"what\s+drove|overview|background|describe\s+the\s+policy"
    r")\b",
    re.IGNORECASE,
)

_NUMERIC_ENTITY_TERMS = frozenset(
    {
        "production",
        "yield",
        "price",
        "prices",
        "gdp",
        "hdi",
        "tonnes",
        "volume",
        "food security",
        "market price",
        "population",
        "ipc",
        "fews",
        "maize",
        "rice",
        "wheat",
        "millet",
        "sorghum",
        "cassava",
    }
)

_COMPARATIVE_BQ_RE = re.compile(
    r"\b("
    r"what\s+drove|drivers?|factors?\s+behind|compared\s+to|relative\s+to|"
    r"benchmark|backdrop|context\s+for|versus|\bvs\.?"
    r")\b",
    re.IGNORECASE,
)

_PURE_NARRATIVE_RE = re.compile(
    r"\b("
    r"what\s+does\s+the\s+policy\s+say|summarize\s+the\s+brief|"
    r"summarise\s+the\s+brief|policy\s+brief\s+say"
    r")\b",
    re.IGNORECASE,
)

_COMPARATIVE_BQ_INTENTS = frozenset({"compare", "diagnostic", "decision_support"})

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
    acf: ACFResult | None = None


def _generate_max_tokens() -> int:
    return int(os.environ.get("RAG_GENERATE_MAX_TOKENS", "2048") or 2048)


def _context_max_chars(memory_block: str = "", *, soft_cap: bool = False) -> int:
    base = int(os.environ.get("RAG_GENERATE_CONTEXT_MAX_CHARS", "12000") or 12000)
    if soft_cap:
        try:
            halved = int(os.environ.get("RAG_GENERATE_CONTEXT_MAX_CHARS_NO_BQ", str(max(4000, base // 2))) or base // 2)
        except ValueError:
            halved = max(4000, base // 2)
        base = min(base, max(4000, halved))
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
    status = str(meta.get("status") or "").strip().lower()
    if status in {"no_project", "no_valid_sql", "validation_failed", "execution_error"}:
        return False
    raw = str(item.get("content") or item.get("text") or "").strip().lower()
    return not any(marker in raw for marker in _BQ_FAILURE_MARKERS)


def is_ranking_numeric_query(query: str) -> bool:
    """True for highest/lowest/which-country style ranking questions."""
    return bool(_RANKING_QUERY_RE.search(query or ""))


def is_numeric_data_query(
    query: str,
    decomposition: dict[str, Any] | None = None,
) -> bool:
    """True when the question asks for quantitative/numeric facts (superset of ranking)."""
    if is_ranking_numeric_query(query):
        return True
    q = query or ""
    if _QUALITATIVE_ONLY_RE.search(q) and not _NUMERIC_QUANTITY_RE.search(q):
        return False
    if _NUMERIC_QUANTITY_RE.search(q):
        return True
    if isinstance(decomposition, dict):
        entities = decomposition.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                text = str(entity or "").strip().lower()
                if not text:
                    continue
                if any(term in text for term in _NUMERIC_ENTITY_TERMS):
                    return True
    return False


def is_comparative_bq_query(
    query: str,
    decomposition: dict[str, Any] | None = None,
) -> bool:
    """True when BQ should support comparative/diagnostic synthesis (not authoritative numeric)."""
    if is_numeric_data_query(query, decomposition):
        return False
    q = query or ""
    if _PURE_NARRATIVE_RE.search(q):
        return False
    if _COMPARATIVE_BQ_RE.search(q):
        return True
    if not isinstance(decomposition, dict):
        return False
    intent = str(decomposition.get("intent") or "").strip().lower()
    if intent in _COMPARATIVE_BQ_INTENTS:
        return True
    if intent in ("diagnostic", "decision_support"):
        entities = decomposition.get("entities")
        geography = decomposition.get("geography")
        entity_text = " ".join(str(e) for e in entities).lower() if isinstance(entities, list) else ""
        geo_text = " ".join(str(g) for g in geography).lower() if isinstance(geography, list) else ""
        has_topic = bool(entity_text.strip())
        has_geo = bool(geo_text.strip())
        if has_topic and has_geo:
            return True
    return False


def should_elevate_bq_context(
    query: str,
    decomposition: dict[str, Any] | None,
    *,
    usable_bq: bool,
) -> bool:
    """Pin/boost BQ when numeric or comparative analysis can use structured rows."""
    if not usable_bq:
        return False
    return is_numeric_data_query(query, decomposition) or is_comparative_bq_query(
        query, decomposition
    )


def pin_bq_context_first(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Place BigQuery structured items before narrative sources (order preserved within groups)."""
    bq_items = [item for item in items if _source_kind(item) == "bigquery"]
    other_items = [item for item in items if _source_kind(item) != "bigquery"]
    return bq_items + other_items


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


def _source_header_label(
    item: dict[str, Any],
    source_id: int,
    *,
    cite_line: str | None = None,
) -> str:
    """Context header with type and optional bibliographic citation line for prose attribution."""
    kind = _source_kind(item)
    header = f"[Source {source_id}]\nType: {_source_kind_display(kind)}"
    if cite_line:
        header += f"\nCitation: {cite_line}"
    return header


def dedupe_bq_context_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate BigQuery rows (same metadata ``source_id``) before packing."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        if _source_kind(item) != "bigquery":
            out.append(item)
            continue
        meta = _item_metadata(item)
        key = str(meta.get("source_id") or "").strip()
        if not key:
            out.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _registry_from_context_items(items: list[dict[str, Any]]) -> list[SourceRef]:
    """Build a citation registry from citable context items (packing-independent)."""
    registry: list[SourceRef] = []
    for idx, item in enumerate(items):
        cite_line = _format_source_citation(item)
        if cite_line:
            registry.append(
                SourceRef(source_id=idx + 1, item=item, citation_line=cite_line)
            )
    return registry


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
        shrinkable = [j for j in range(n) if alloc[j] > 1]
        if not shrinkable:
            break
        non_bq = [j for j in shrinkable if _source_kind(items[j]) != "bigquery"]
        idx = max(non_bq if non_bq else shrinkable, key=lambda j: alloc[j])
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
        source_id = idx + 1
        cite_line = _format_source_citation(item)
        if cite_line:
            registry.append(SourceRef(source_id=source_id, item=item, citation_line=cite_line))

        if used >= budget:
            continue

        raw = str(item.get(content_key) or item.get("text") or "")
        body = raw.strip()

        remaining = budget - used
        take = min(limit, chunk_cap, remaining)
        if take <= 0:
            continue

        header = _source_header_label(item, source_id, cite_line=cite_line)
        body_trunc = body[: max(0, take - len(header) - 2)]
        if not body_trunc.strip():
            continue

        block = f"{header}\n{body_trunc.strip()}"
        parts.append(block)
        used += len(block) + 2

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


def _strip_invalid_citation_markers(answer: str, source_registry: list[SourceRef]) -> str:
    """Remove [N] / [Source N] footnotes that reference missing registry IDs."""
    if not answer or not source_registry:
        return answer
    valid = {ref.source_id for ref in source_registry}

    def _keep_bracket(m: re.Match[str]) -> str:
        try:
            n = int(m.group(1))
        except (TypeError, ValueError):
            return ""
        return m.group(0) if n in valid else ""

    text = re.sub(r"\[(?!\s*Source\s)(\d+)\]", _keep_bracket, answer)
    text = re.sub(r"\[Source\s+(\d+)\]", _keep_bracket, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\[)\bSource\s+(\d+)\b(?!\])", _keep_bracket, text, flags=re.IGNORECASE)
    # Collapse orphaned punctuation/spacing left by removed markers.
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    return text.strip()


def _strip_all_inline_citation_markers(answer: str) -> str:
    """Remove every [N] / [Source N] marker when inline footnotes are disabled."""
    if not answer:
        return answer
    text = re.sub(r"\[(?!\s*Source\s)\d+\]", "", answer)
    text = re.sub(r"\[Source\s+\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\[)\bSource\s+\d+\b(?!\])", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    return text.strip()


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
    category: str = "",
    plan_type: str = "",
    answer_lang: str | None = None,
    structured_bq_unavailable: bool = False,
    structured_bq_numeric_available: bool = False,
    structured_bq_comparative_available: bool = False,
    analytical_mode: bool = False,
    task_mode: str = "chat",
    measure_id: str | None = None,
    recency_tier: str | None = None,
    context_source_kinds: list[str] | None = None,
    inline_citations: bool = False,
    generation_plan: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    lang = (answer_lang or "").strip() or detect_answer_language(query)
    mode = (task_mode or ("analytical" if analytical_mode else "chat")).strip().lower()
    if inline_citations:
        cite_rules = (
            "When stating a specific fact, number, or claim from the context, name the source in readable "
            "prose using the Citation line (author or organization, title or outlet, and year when present), "
            "then add the matching footnote number [N] — the same N as [Source N]. "
            "Never cite with a bare number alone (wrong: 'According to [6]'; right: 'According to Branca et al. "
            "(2012), agriculture's GDP share rose.[6]'). "
            "Never refer to in-document labels from the context such as 'Table 6.1', 'Figure 2', "
            "'Fig. 3', or 'Appendix A' — paraphrase the finding and cite with the Citation line + [N] only. "
        )
    else:
        cite_rules = (
            "When stating a specific fact, number, or claim from the context, you may name the source in "
            "readable prose using the Citation line (author or organization, title or outlet, and year when "
            "present). Do NOT insert [N], [Source N], or bare Source N footnote markers — the system returns "
            "structured citations separately for the client. "
            "Never refer to in-document labels from the context such as 'Table 6.1', 'Figure 2', "
            "'Fig. 3', or 'Appendix A' — paraphrase the finding instead. "
        )
    system = (
        "You are an agricultural business-intelligence assistant for OpenTrace stakeholders "
        "(government, NGOs, agribusiness, finance, farmers). "
        "OpenTrace / Ask ADZA is scoped to African agriculture, food systems, climate, "
        "markets, and related policy — unless the user explicitly names a non-African place. "
        "For unscoped which-country or continental rankings, answer from African OpenTrace "
        "evidence (especially structured data). Do not crown a non-African country from "
        "Wikipedia or general knowledge when Africa-scoped evidence is missing. "
        # Lead with the answer. This instruction must come first because LLMs weight
        # early-prompt content most heavily; burying it lower causes the model to
        # default to thesis-style preambles (testing R8: 'lead with a clear answer').
        "Your first sentence is the direct answer to the user's question — no preamble, "
        "no 'According to the context...', no 'It is important to note...', no scene-setting. "
        "After the direct answer, support it with the relevant evidence in plain prose. "
        "Write clear, decisive prose for decision-makers — not a database console, not an academic paper. "
        "Answer ONLY using facts in the Context below from numbered OpenTrace sources "
        "(each chunk is labeled [Source N] with Type and Citation lines when available). "
        "Synthesize evidence across all numbered sources — do not rely on a single snippet. "
        + cite_rules
        + "Do not paste raw context headers, Type lines, Citation lines, or pipe-delimited "
        "[Source N | ...] strings into the answer. "
        "Do not output a Sources, References, or Bibliography section; the system appends one. "
        "Never echo BigQuery or SQL table identifiers (e.g. stg_*, bronze table names) in the prose. "
        "Sources labeled Type: Wikipedia or Type: Web search are supplemental external context — "
        "prefer OpenTrace news, research, and structured data when available; treat web sources as "
        "partial background and state limits when relying on them. "
        # Length matches the question. The previous fixed '4–8 paragraphs' floor was the
        # single biggest cause of thesis-style answers — the model padded with hedges
        # and restatements to hit the target even on simple lookups. Drop the floor;
        # let the question determine the length.
        "Length matches the question: 2–4 sentences for a simple lookup, 2–4 short paragraphs "
        "for a complex synthesis. Never pad with filler, restatements of the question, or hedges to fill space. "
        "For complex questions structure the answer as: (1) direct answer first, "
        "(2) supporting evidence by theme, region, or time period, (3) brief limits or gaps. "
        "For compare or trend questions, organize by region or time period when the context provides that breakdown. "
        "Include specific numbers, dates, and regions when present in the context. "
        # Anti-thesis openings. Listed explicitly because the model otherwise reaches
        # for these by default (the corpus is heavily academic). Tested phrases from
        # internal R6/R8 reports.
        "Forbidden openings — never start with: 'The context provided', 'Based on the context', "
        "'Unfortunately', 'It is important to note', 'It is worth noting', 'This study examines', "
        "'The evidence suggests', 'In recent years', 'Across the literature', or any similar academic / "
        "meta-commentary opener. Start with the substantive answer instead. "
        + language_instruction(lang, inline_citations=inline_citations)
        + " "
        "If evidence is partial, state limits briefly after the substantive answer. "
        "Do not invent sources, cite Source IDs not in the Context, or invent statistics. "
        "Never output SQL, query code, table DDL, pipeline steps, or instructions to run BigQuery. "
        "Never mention bigquery-public-data or other external warehouses. "
        "Do not suggest example queries the user should run."
    )
    kinds = [str(k).strip().lower() for k in (context_source_kinds or []) if str(k).strip()]
    unique_kinds = list(dict.fromkeys(kinds))
    if len(unique_kinds) >= 2:
        system = (
            system
            + "\n\nCROSS-DOMAIN SYNTHESIS: Context includes multiple source Types ("
            + ", ".join(unique_kinds)
            + "). Domains reinforce each other — weave them together. "
            "Use structured/BigQuery rows for levels, trends, and comparisons when present; "
            "use news, policy, public reports, and research for mechanisms, events, and "
            "stakeholder context. Do not answer from only one Type when others are relevant. "
            "If structured data is missing, still synthesize across the narrative corpora "
            "that are present; admit the structured gap once and do not invent yields, IPC "
            "phases, or exact production totals."
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

    cat_tone = instruction_for_category(
        category or None,
        measure_id=measure_id,
        recency_tier=recency_tier,
    ) if (category or measure_id or recency_tier) else ""
    if cat_tone:
        system = system + "\n\n" + cat_tone
    plan_addendum = plan_generation_addendum(plan_type) if plan_type else ""
    if plan_addendum:
        system = system + "\n\n" + plan_addendum
    if generation_plan:
        from ml.rag.chatbot.generation_plan import generation_plan_addendum

        gen_plan_addendum = generation_plan_addendum(generation_plan)
        if gen_plan_addendum:
            system = system + "\n\n" + gen_plan_addendum
    if analytical_mode or mode == "analytical":
        bq_cite = (
            "Prefer OpenTrace structured data (Type: structured / BigQuery) for every quantitative "
            "claim — cite those [N] footnotes first. "
            if inline_citations
            else "Prefer OpenTrace structured data (Type: structured / BigQuery) for every quantitative "
            "claim — attribute in prose when needed; do not insert [N] footnotes. "
        )
        system = (
            system
            + "\n\nANALYTICAL REPORT MODE: Structure the answer with these markdown headings "
            "in order:\n"
            "## Executive summary\n"
            "## Regional overview\n"
            "## Country comparison\n"
            "## Major products\n"
            "## Data gaps\n"
            "## Conclusion\n"
            + bq_cite
            + "Do not invent production totals, rankings, or "
            "year values that are not in Context. If structured rows are thin, say so under Data gaps. "
            "Keep country lists aligned with the geography in Context; do not substitute unrelated "
            "countries from narrative PDFs."
        )
    elif mode == "fact_lookup":
        system = (
            system
            + "\n\nFACT LOOKUP MODE: Give a short direct answer (1–3 sentences) using structured "
            "data first when present. Lead with the number, country, crop, and year. "
            "Do not write a multi-section report."
        )
    elif mode == "briefing":
        system = (
            system
            + "\n\nBRIEFING MODE: Write a short situational briefing with 3–6 bullet points. "
            "Prefer the most recent news and OTA insights. End with one line on confidence limits."
        )
    elif mode == "research":
        system = (
            system
            + "\n\nRESEARCH SYNTHESIS MODE: Synthesize academic/policy evidence in plain "
            "stakeholder language. Prefer peer-reviewed and policy corpora; use BQ only for "
            "bibliographic/project facts. Do not invent citations."
        )
    elif mode == "data_export_only":
        system = (
            system
            + "\n\nDATA EXPORT MODE: Write only a 2–4 sentence caption summarizing the table/chart "
            "the user will download. No essay, no multi-section report, no long narrative."
        )
    # Keep category plainness/precision when answering in a named non-English language
    # (avoids English academic bleed on e.g. Igbo + Farmers).
    if category and cat_tone and not is_english_answer_lang(lang) and lang not in ("unknown", ""):
        system = (
            system
            + "\n\nAnswer in the user's language while keeping the category audience rules "
            "(plainness, framing, and jargon limits) — do not switch to English academic prose."
        )
    if structured_bq_unavailable and is_numeric_data_query(query, decomposition):
        system = (
            "CRITICAL: BigQuery structured data was attempted but returned no usable rows. "
            "Do NOT invent specific numeric facts (production totals, precise yield figures, "
            "prices, GDP, population counts, or ranked country answers) from news or policy "
            "text alone. If the question asks for a numeric answer from structured data, "
            "state clearly that OpenTrace structured data is unavailable for this query and "
            "avoid naming a specific number or country unless a structured-data source in "
            "Context explicitly provides it.\n\n"
        ) + system
    if structured_bq_numeric_available:
        system = (
            "CRITICAL: Context includes OpenTrace structured BigQuery data with explicit "
            "measure labels and units. For numeric answers (totals, yields, prices, GDP, "
            "population counts, rankings), use those structured rows as the authoritative "
            "source. Do not override them with policy, news, or web narrative that cites "
            "a different number, unit, or country.\n\n"
        ) + system
    elif structured_bq_comparative_available:
        system = (
            "Context includes OpenTrace structured data with measure labels and units. "
            "Use it for comparative statistics (production levels, yields, prices, trends, "
            "regional benchmarks). Use policy, research, and news sources for causal "
            "explanations and policy drivers. Do not invent numbers; cite structured data "
            "when stating quantitative comparisons.\n\n"
        ) + system

    mb = (memory_block.strip() + "\n\n") if memory_block.strip() else ""
    user = f"{mb}Context:\n{context_block}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _resolve_memory_block(**kwargs: Any) -> str:
    query = str(kwargs.get("query") or kwargs.get("_query") or "")
    dec = kwargs.get("decomposition") if isinstance(kwargs.get("decomposition"), dict) else None
    conv_summary = kwargs.get("conversation_summary")
    recent_turns = kwargs.get("recent_turns")
    raw_history = kwargs.get("chat_history")
    block = ""
    if isinstance(recent_turns, list) or isinstance(conv_summary, str):
        s = (conv_summary if isinstance(conv_summary, str) else "").strip()
        rt = normalize_messages(recent_turns if isinstance(recent_turns, list) else None)
        if not memory_relevant_for_query(query, s, rt, dec):
            return ""
        block = build_memory_prompt_block(s, rt)
    elif isinstance(raw_history, list) and raw_history:
        rt = truncate_chat_history(raw_history)
        hist_text = "\n".join(m.get("content") or "" for m in rt)
        if not memory_relevant_for_query(query, hist_text, rt, dec):
            return ""
        block = build_memory_prompt_block("", rt)
    cap = default_verbatim_max_chars() + default_summary_max_chars()
    if len(block) > cap:
        block = block[-cap:]
    return block


def _strip_doc_table_figure_labels(text: str) -> str:
    """Remove leftover Table/Figure/Appendix labels from model prose."""
    if not text:
        return text
    cleaned = _DOC_TABLE_FIGURE_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() or text.strip()


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


def _strip_preamble_openers(text: str) -> str:
    """
    Sprint 1, Week 3 (direct-answer-first): deterministic backstop for the "no
    preamble" prompt rule. If the model still opens with a connective/meta-commentary
    preamble ("Based on the context, ...", "It is important to note that ...", etc.),
    strip it so the substantive answer leads. Capitalises the new first character.

    Conservative by design:
      - Only rewrites the very START of the answer.
      - Only strips recognised preamble phrases (see _PREAMBLE_OPENER_RE); leaves
        content-bearing openings untouched.
      - Never returns empty: if stripping would leave nothing, the original is kept.
      - Unwinds at most _MAX_PREAMBLE_UNWIND stacked preambles.
    """
    if not text:
        return text
    out = text.lstrip()
    for _ in range(_MAX_PREAMBLE_UNWIND):
        m = _PREAMBLE_OPENER_RE.match(out)
        if not m:
            break
        stripped = out[m.end():].lstrip()
        if not stripped:
            # Preamble was the entire content — keep original rather than emptying.
            return text.strip()
        # Recapitalise the first alphabetic character of the remaining answer.
        out = stripped[0].upper() + stripped[1:] if stripped[0].isalpha() else stripped
    return out


def _clean_answer(text: str) -> str:
    """Remove Llama chat template echoes and other non-answer artifacts from LLM output."""
    if not text:
        return text
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    text = re.sub(r"^(Context:|Question:).*$", "", text, flags=re.MULTILINE | re.IGNORECASE).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_sql_from_answer(text) or text.strip()
    text = _strip_doc_table_figure_labels(text) or text.strip()
    text = _strip_model_sources_appendix(text)
    return _strip_preamble_openers(text)


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
    """Extract a clickable URL for a citation, trying multiple metadata fields.

    Sprint 1, Week 3: improved to check url/link/source_url for ALL source types
    (not just news), so academic papers with a direct URL also get clickable links
    even when DOI is missing.
    """
    k = kind.lower()

    # Web sources — url is always present
    if k in ("web_wikipedia", "web_search"):
        url = str(meta.get("url") or "").strip()
        return url or None

    # News — check multiple URL field names
    if k in ("news_article", "news"):
        for key in ("url", "link", "source_url"):
            url = str(meta.get(key) or "").strip()
            if url.startswith("http"):
                return url
        return None

    # Academic / policy / public report — DOI preferred, then direct URL fallback
    if k in ("academic_article", "academic", "policy_document", "policy", "public_report"):
        doi = str(meta.get("doi") or "").strip()
        if doi.startswith("http"):
            return doi
        if doi:
            return f"https://doi.org/{doi.lstrip('doi:').strip()}"
        # Fallback: check for direct url/link fields (some ingested records have these)
        for key in ("url", "link", "source_url"):
            url = str(meta.get(key) or "").strip()
            if url.startswith("http"):
                return url
        return None

    # OTA insights — check for url field
    if k in ("ota_insight", "ota_metric"):
        url = str(meta.get("url") or meta.get("source_url") or "").strip()
        return url if url.startswith("http") else None

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


def _referenced_source_refs(
    answer: str,
    source_registry: list[SourceRef],
    *,
    inline_citations: bool = True,
) -> list[SourceRef]:
    if not source_registry:
        return []
    if not inline_citations:
        # No inline markers expected — return all packed sources for the citation block.
        return list(source_registry)
    mode = _citations_mode()
    if mode == "referenced":
        cited_ids = extract_referenced_source_ids(answer)
        if not cited_ids:
            return []
        return [r for r in source_registry if r.source_id in cited_ids]
    return list(source_registry)


def referenced_citations(
    answer: str,
    source_registry: list[SourceRef],
    *,
    inline_citations: bool = True,
) -> list[dict[str, Any]]:
    """Build structured citation objects for referenced (or all) sources."""
    prose = _strip_model_sources_appendix(answer)
    refs = _referenced_source_refs(prose, source_registry, inline_citations=inline_citations)
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


def _decomposition_geo_hint(decomposition: dict[str, Any] | None) -> str:
    """Extract a short human-readable geo hint from the query decomposer output, if any."""
    if not isinstance(decomposition, dict):
        return ""
    for key in ("countries", "regions", "geography", "geo"):
        val = decomposition.get(key)
        if isinstance(val, list) and val:
            return ", ".join(str(x) for x in val if x)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _decomposition_time_hint(decomposition: dict[str, Any] | None) -> str:
    """Extract a short time-window hint from the query decomposer output, if any."""
    if not isinstance(decomposition, dict):
        return ""
    ts = str(decomposition.get("time_start") or decomposition.get("start_date") or "").strip()
    te = str(decomposition.get("time_end") or decomposition.get("end_date") or "").strip()
    if ts and te:
        return f"{ts} → {te}"
    return ts or te or str(decomposition.get("time_period") or "").strip()


def _no_data_fallback_message(
    query: str,
    decomposition: dict[str, Any] | None = None,
) -> str:
    """
    Sprint 1 (Jul 2026): structured gap-acknowledgement returned when we have no
    OpenTrace context for a query. Replaces the previous single-sentence fallback
    which testers found dev-facing and easy to mistake for a low-confidence answer.

    Format:
      - one-line direct statement
      - Gap block (query + any geo/time the decomposer extracted)
      - What would help block (concrete guidance)
      - explicit ACF marker so the confidence signal is never invisible on this path
    """
    q = (query or "").strip() or "your question"
    geo = _decomposition_geo_hint(decomposition)
    time_hint = _decomposition_time_hint(decomposition)

    gap_lines = [f"- Query: {q}"]
    if geo:
        gap_lines.append(f"- Geography: {geo}")
    if time_hint:
        gap_lines.append(f"- Time period: {time_hint}")

    help_lines = [
        "- Name a specific country or region (e.g. Kenya, West Africa).",
        "- Name a specific crop, commodity, or agricultural metric.",
        "- Narrow the time period (e.g. last 3 years, 2020–2024).",
    ]

    return (
        "I don't have OpenTrace data for this question.\n\n"
        "**Gap**\n"
        + "\n".join(gap_lines)
        + "\n\n**What would help**\n"
        + "\n".join(help_lines)
        + "\n\n**ACF: no evidence** — no OpenTrace sources (news, research, policy, "
        "public reports, or structured agricultural data) matched this query, so no "
        "confidence signal can be computed. This is a data gap, not a low-confidence answer."
    )


def _query_target_countries(decomposition: dict[str, Any] | None) -> list[str]:
    """Normalized list of countries the query is scoped to (zone labels excluded)."""
    if not isinstance(decomposition, dict):
        return []
    out: list[str] = []
    for key in ("countries", "geography", "regions", "geo"):
        val = decomposition.get(key)
        if isinstance(val, list):
            out.extend(str(x).strip() for x in val if str(x).strip())
        elif isinstance(val, str) and val.strip():
            out.append(val.strip())
    seen: set[str] = set()
    result: list[str] = []
    for c in out:
        if is_zone_label(c):
            continue
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            result.append(c)
    return result


def usable_context_after_geo_purity(
    items: list[dict[str, Any]] | None,
    decomposition: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Filter unusable items then drop geo-conflicting chunks (inspector / generate)."""
    usable = filter_context_items(list(items or []))
    return _drop_geo_conflicting(usable, decomposition)


def _geo_conflicts(item: dict[str, Any], allowed_lower: set[str]) -> bool:
    """
    True only when a chunk's geo metadata names specific countries and NONE of
    them match the query's target countries. Chunks with no geo metadata are kept
    (benefit of the doubt — e.g. structured BQ rows, global/continental references).
    """
    meta = _item_metadata(item)

    def _norm_list(s: str) -> set[str]:
        if not s:
            return set()
        parts = re.split(r"[;,/]", s)
        return {p.strip().lower() for p in parts if p.strip()}

    primary = str(meta.get("geo_country_primary") or meta.get("country") or "")
    blob = str(meta.get("geo_countries") or "")
    meta_countries = _norm_list(primary) | _norm_list(blob)
    if not meta_countries:
        return False  # no geo signal → keep
    return not (meta_countries & allowed_lower)


def _drop_geo_conflicting(
    items: list[dict[str, Any]],
    decomposition: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Sprint 1 (Jul 2026): source purity. Drop chunks whose geo metadata clearly
    names other countries than the query asked for (e.g. a Kenya-tagged chunk on a
    Senegal query). Conservative: chunks lacking geo metadata are always kept, so
    structured/global references are not accidentally removed. Toggle off with
    RAG_DROP_GEO_CONFLICTING_CONTEXT=off.
    """
    if os.environ.get("RAG_DROP_GEO_CONFLICTING_CONTEXT", "on").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return items
    countries = _query_target_countries(decomposition)
    if not countries:
        return items
    allowed_lower = {c.lower() for c in countries}
    return [it for it in items if not _geo_conflicts(it, allowed_lower)]


def _min_usable_context() -> int:
    """
    Minimum usable chunks required before we call the LLM. When the surviving
    context is at or below this count we return the structured gap message instead
    of letting the model pad thin/irrelevant context with general-knowledge prose.
    Default 0 (disabled) preserves prior behaviour; set RAG_MIN_USABLE_CONTEXT=1+
    to enforce.
    """
    try:
        return max(0, int(os.environ.get("RAG_MIN_USABLE_CONTEXT", "0") or 0))
    except ValueError:
        return 0


def _finalize_generation_result(
    answer: str,
    source_registry: list[SourceRef],
    *,
    query: str = "",
    decomposition: dict[str, Any] | None = None,
    inline_citations: bool = False,
) -> GenerationResult:
    """Attach structured citations and score ACF Path B on cited sources only."""
    t0 = time.perf_counter()
    with observed_span("citations", input_data={"registry_size": len(source_registry)}):
        prose = _strip_model_sources_appendix(answer)
        if inline_citations:
            prose = _strip_invalid_citation_markers(prose, source_registry)
        else:
            prose = _strip_all_inline_citation_markers(prose)
        citations = referenced_citations(
            prose, source_registry, inline_citations=inline_citations
        )
        if _append_sources_to_answer() and citations:
            lines = [f"{c['id']}. {c['text']}" for c in citations]
            prose = (prose.rstrip() + "\n\nSources\n" + "\n".join(lines)).strip()

        kind_counts: dict[str, int] = {}
        for c in citations:
            kind = str(c.get("kind") or "unknown")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        cited_ids: list[int] = []
        for c in citations[:20]:
            try:
                cited_ids.append(int(c["id"]))
            except (KeyError, TypeError, ValueError):
                continue

        cited_refs = _referenced_source_refs(
            prose, source_registry, inline_citations=inline_citations
        )
        if cited_refs:
            acf = score_cited_evidence(
                cited_refs,
                query=query,
                decomposition=decomposition,
            )
            acf_status = "scored"
        else:
            acf = no_evidence_acf()
            acf_status = "no_citations"

        update_current_span_metadata(
            {
                "citation_count": len(citations),
                "registry_size": len(source_registry),
                "citations_mode": _citations_mode(),
                "inline_citations": inline_citations,
                "cited_ids": cited_ids,
                "kind_counts": kind_counts,
                "latency_ms": trace_elapsed_ms(t0),
                "acf_status": acf_status,
                "acf_band": acf.band,
                "acf_band_label": acf.band_label,
                "acf_score": acf.score,
                "acf_explanation": acf.explanation,
                "acf_claim_level": acf.claim_level,
                "acf_question_type": acf.question_type,
                "acf_applied_ceiling": acf.applied_ceiling,
                "acf_config_version": acf.config_version,
            }
        )
        return GenerationResult(answer=prose, citations=citations, acf=acf)


def _call_llama(
    messages: list[dict[str, str]],
    *,
    purpose: str = "generate",
    model: str | None = None,
) -> str:
    """Call configured LLM backend; never raises on HTTP errors.

    model: override the global RAG_LLM_MODEL_ID (used for per-plan routing, ML-041).
    When None, falls back to llm_model_id() which reads RAG_LLM_MODEL_ID from env.
    """
    gen_timeout = float(os.environ.get("RAG_GENERATE_TIMEOUT_S", "0") or 0) or llm_default_timeout_s()
    max_toks = _generate_max_tokens()
    # Sprint 1 (Jul 2026): set to 0.7 so responses have natural variety across similar
    # queries. The "no synthesis on gaps" test-1 concern is handled by code-level
    # guardrails (empty context → `_no_data_fallback_message` without an LLM call;
    # `RAG_ALLOW_UNGROUNDED` path is prompt-hardened; `filter_context_items` drops
    # broken chunks), so temperature can stay in the natural-prose range.
    temperature = float(os.environ.get("RAG_GENERATE_TEMPERATURE", "0.7") or 0.7)
    return llm_chat_complete(
        messages,
        model=model or llm_model_id(),
        max_tokens=max_toks,
        temperature=temperature,
        timeout_s=gen_timeout,
        purpose=purpose or "generate",
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

    mem_kwargs = dict(kwargs)
    mem_kwargs["query"] = query
    mem_kwargs["decomposition"] = decomposition
    memory_block = _resolve_memory_block(**mem_kwargs)

    category = str(kwargs.get("category") or "").strip()
    plan_type = str(kwargs.get("plan_type") or "").strip()
    answer_lang = str(kwargs.get("answer_lang") or "").strip() or None
    structured_bq_unavailable = bool(kwargs.get("structured_bq_unavailable"))
    structured_bq_numeric_available = bool(kwargs.get("structured_bq_numeric_available"))
    structured_bq_comparative_available = bool(kwargs.get("structured_bq_comparative_available"))
    analytical_mode = bool(kwargs.get("analytical_mode"))
    task_mode = str(kwargs.get("task_mode") or ("analytical" if analytical_mode else "chat")).strip()
    measure_id = str(kwargs.get("measure_id") or "").strip() or None
    recency_tier = str(kwargs.get("recency_tier") or "").strip() or None
    export_intent = kwargs.get("export_intent")
    export_intent_s = str(export_intent).strip() if export_intent else None
    generation_plan = kwargs.get("generation_plan")
    if generation_plan is not None and not isinstance(generation_plan, dict):
        generation_plan = None
    inline_citations = want_inline_citations(
        query,
        task_mode=task_mode,
        export_intent=export_intent_s,
    )

    if structured_bq_unavailable and is_numeric_data_query(query, decomposition):
        usable_preview = filter_context_items(context_items or [])
        has_narrative = any(is_usable_context_item(item) for item in usable_preview)
        if not has_narrative:
            return GenerationResult(
                answer=_no_data_fallback_message(query, decomposition),
                citations=[],
                acf=no_evidence_acf(
                    explanation=(
                        "Structured BigQuery data was required for this numeric question "
                        "but returned no usable rows."
                    )
                ),
            )

    if not context_items:
        allow_ungrounded = os.environ.get("RAG_ALLOW_UNGROUNDED", "").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        )
        if allow_ungrounded:
            # Sprint 1 (Jul 2026): even when RAG_ALLOW_UNGROUNDED=on, do NOT let the
            # model synthesise academic prose from thin air. Hard-prepend a rule that
            # forces a structured gap acknowledgement when Context is empty.
            messages = _build_prompt(
                query,
                context_block="[No external context]",
                decomposition=decomposition,
                memory_block=memory_block,
                category=category,
                plan_type=plan_type,
                answer_lang=answer_lang,
                structured_bq_unavailable=structured_bq_unavailable,
                structured_bq_numeric_available=structured_bq_numeric_available,
                structured_bq_comparative_available=structured_bq_comparative_available,
                analytical_mode=analytical_mode,
                task_mode=task_mode,
                measure_id=measure_id,
                recency_tier=recency_tier,
                inline_citations=inline_citations,
                generation_plan=generation_plan,
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = (
                    "CRITICAL: The Context below is empty ('[No external context]'). "
                    "Do NOT synthesise, do NOT draw on general knowledge, do NOT produce "
                    "academic prose. Reply in exactly this shape:\n"
                    "  Line 1: 'I don't have OpenTrace data for this question.'\n"
                    "  Then a short 'What would help' list (3 bullets: country/region, "
                    "crop or commodity, time period).\n"
                    "  Then a final line: 'ACF: no evidence.'\n"
                    "No other prose. No hedging. No caveats beyond the ACF line.\n\n"
                ) + messages[0]["content"]
            llama_answer = _call_llama(messages, purpose="generate", model=model_for_plan(plan_type))
            if llama_answer:
                cleaned = _normalize_inline_citations(_clean_answer(llama_answer))
                return _finalize_generation_result(
                    cleaned,
                    [],
                    query=query,
                    decomposition=decomposition,
                    inline_citations=inline_citations,
                )
        # Default (RAG_ALLOW_UNGROUNDED off or LLM returned nothing): structured
        # gap message so testers can distinguish "no data" from "low confidence".
        return GenerationResult(
            answer=_no_data_fallback_message(query, decomposition),
            citations=[],
            acf=no_evidence_acf(),
        )

    usable_context = dedupe_bq_context_items(
        usable_context_after_geo_purity(context_items, decomposition)
    )

    # Sprint 1 (Jul 2026): if geo/error filtering leaves too little usable context,
    # return the structured gap message instead of letting the model pad thin or
    # irrelevant context with general-knowledge prose (test-1 finding).
    min_usable = _min_usable_context()
    if len(usable_context) <= min_usable:
        return GenerationResult(
            answer=_no_data_fallback_message(query, decomposition),
            citations=[],
            acf=no_evidence_acf(),
        )

    ctx_budget = _context_max_chars(
        memory_block,
        soft_cap=bool(
            structured_bq_unavailable and task_mode in ("chat", "briefing")
        ),
    )
    chunk_cap = _chunk_max_chars()
    context_block, source_registry = _build_context_block(usable_context, ctx_budget, chunk_cap)
    if not source_registry and usable_context:
        source_registry = _registry_from_context_items(usable_context)
    source_kinds = [normalize_context_kind(item) for item in usable_context]

    messages = _build_prompt(
        query,
        context_block,
        decomposition=decomposition,
        memory_block=memory_block,
        category=category,
        plan_type=plan_type,
        answer_lang=answer_lang,
        structured_bq_unavailable=structured_bq_unavailable,
        structured_bq_numeric_available=structured_bq_numeric_available,
        structured_bq_comparative_available=structured_bq_comparative_available,
        analytical_mode=analytical_mode,
        task_mode=task_mode,
        measure_id=measure_id,
        recency_tier=recency_tier,
        context_source_kinds=source_kinds,
        inline_citations=inline_citations,
        generation_plan=generation_plan,
    )
    llama_answer = _call_llama(messages, purpose="generate", model=model_for_plan(plan_type))
    if llama_answer:
        cleaned = _normalize_inline_citations(_clean_answer(llama_answer))
        return _finalize_generation_result(
            cleaned,
            source_registry,
            query=query,
            decomposition=decomposition,
            inline_citations=inline_citations,
        )

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
        acf=no_evidence_acf(
            explanation="Generation failed before citations could be scored."
        ),
    )
