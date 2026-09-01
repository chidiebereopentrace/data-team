"""
Generator node: takes query + reranked context and produces the final answer (LLM).

Uses the configured chat backend (OpenRouter, LM Studio, or Hugging Face router) via ``llm_chat``.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from ml.rag.llm_chat import llm_chat_complete, llm_configured, llm_default_timeout_s, llm_model_id
from ml.rag.chatbot.plan_policy import model_for_plan

from ml.rag.chat_history import normalize_messages, truncate_chat_history
from ml.rag.chat_memory import (
    build_memory_prompt_block,
    default_summary_max_chars,
    default_verbatim_max_chars,
)
from ml.rag.chatbot.acf_scoring import ACFResult, apply_bq_execute_ceiling, no_evidence_acf, score_cited_evidence
from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    insufficient_context_answer,
    is_english_answer_lang,
    language_instruction,
)
from ml.rag.chatbot.context_diversity import dedupe_context_items, normalize_context_kind
from ml.rag.chatbot.export_intent import want_inline_citations
from ml.rag.chatbot.geo_regions import is_zone_label
from ml.rag.chatbot.memory_relevance import memory_relevant_for_query
from ml.rag.chatbot.plan_policy import instruction_for_category, plan_generation_addendum
from ml.rag.chatbot.output_format import export_caption_instruction
from ml.rag.chatbot.stakeholder_prompts import prose_register_addendum
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
_SOURCE_MULTI_REF_RE = re.compile(r"\[Sources?\s+([\d,\sand]+)\]", re.IGNORECASE)
_PLAIN_MULTI_REF_RE = re.compile(r"\[(?!\s*Source\s)(\d+(?:\s*[,–-]\s*\d+)+)\]")
_PIPE_LABEL_REF_RE = re.compile(
    r"\[(?:Wikipedia|News|Academic|Policy/Public|Policy|Public|Web|Structured\s+data)\s*\|[^\]]*\]",
    re.IGNORECASE,
)
_NON_PLAIN_BRACKET_RE = re.compile(r"\[(?!\d+\])[^\]]*\]")
_SOCIAL_URL_HOSTS = ("linkedin.com", "twitter.com", "x.com", "facebook.com", "fb.com")
_KNOWN_ANALYTICAL_HEADINGS = (
    "Key Findings",
    "Regional & Country Picture",
    "Production, Trade & Markets",
    "Drivers & Context",
    "Data Notes",
    "Executive summary",
    "Regional overview",
)
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

_PRICE_TREND_RE = re.compile(
    r"\b("
    r"going up|going down|up or down|trending|"
    r"prices (?:are|were) (?:rising|falling|increasing|decreasing)"
    r")\b",
    re.IGNORECASE,
)

_BQ_FAILURE_MARKERS = (
    "[bq execution error",
    "[bq validation failed",
    "[bq no_project",
    "[bq no_valid_sql",
    "[bq empty_result",
    "bigquery retrieve timed out",
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

# Warehouse source_name / source_natural_key → public institution (dim_source.organisation_name).
_SOURCE_KEY_PUBLIC_LABELS: dict[str, str] = {
    "yield_raw_data": "FEWS NET",
    "fews_food_security": "FEWS NET",
    "fews_food_security_master": "FEWS NET",
    "fews_market_prices": "FEWS NET",
    "fews_cross_border_trade": "FEWS NET",
    "faostat_production": "FAOSTAT",
    "faostat_prices": "FAOSTAT",
    "faostat_trade": "FAOSTAT",
    "faostat_food_balances": "FAOSTAT",
    "faostat_emissions": "FAOSTAT",
    "faostat_land_inputs": "FAOSTAT",
    "faostat_population_employment": "FAOSTAT",
    "faostat_investment_asti": "FAOSTAT",
    "faostat_sdg_hdi": "FAOSTAT",
    "wfp_vampire_prices": "WFP",
    "ilri_household_food_security": "ILRI",
    "ilri_animal_health": "ILRI",
    "nasa_power": "NASA POWER",
    "copernicus_era5": "Copernicus ERA5",
    "isric_africa_soil": "ISRIC",
    "isda_soil_enriched": "iSDA",
    "africa_hdi": "UNDP / HDI",
    "africa_gdp_ppp": "World Bank / GDP",
    "openaire_projects": "OpenAIRE",
    "biodiversity": "GBIF / biodiversity",
}


def _label_from_source_key(raw: str) -> str | None:
    """Map a warehouse source key or natural key to a public institution name."""
    key = str(raw or "").strip().lower()
    if not key:
        return None
    if key in _SOURCE_KEY_PUBLIC_LABELS:
        return _SOURCE_KEY_PUBLIC_LABELS[key]
    if key.startswith("stg_"):
        key = key[4:]
    if key in _SOURCE_KEY_PUBLIC_LABELS:
        return _SOURCE_KEY_PUBLIC_LABELS[key]
    if "fews" in key or key == "yield_raw_data" or key == "yield extract":
        return "FEWS NET"
    if "faostat" in key:
        return "FAOSTAT"
    if "ilri" in key:
        return "ILRI"
    if "wfp" in key or "vampire" in key:
        return "WFP"
    if "nasa" in key:
        return "NASA POWER"
    if "copernicus" in key or "era5" in key:
        return "Copernicus ERA5"
    if "isric" in key:
        return "ISRIC"
    if "isda" in key:
        return "iSDA"
    if "openaire" in key:
        return "OpenAIRE"
    if "hdi" in key:
        return "UNDP / HDI"
    if "gdp" in key:
        return "World Bank / GDP"
    if key in {"fews net", "fewsnet"}:
        return "FEWS NET"
    return None


def _public_source_label(table_id: str | None, meta: dict[str, Any] | None = None) -> str | None:
    """Return a clean institutional source name, or None to use the structured-data fallback."""
    payload = meta or {}
    org = str(payload.get("organisation_name") or "").strip()
    if org and org.lower() not in {"other", "unknown"}:
        known = _label_from_source_key(org)
        return known or org

    for key_field in ("source_name", "source_natural_key", "price_source"):
        hit = _label_from_source_key(str(payload.get(key_field) or ""))
        if hit:
            return hit

    tid = (table_id or "").lower().strip()
    domain = str(payload.get("source_domain") or "").strip().lower()
    if domain:
        if "faostat" in domain or domain == "fao":
            return "FAOSTAT"
        if "fews" in domain:
            return "FEWS NET"
        if "ilri" in domain:
            return "ILRI"
        if "wfp" in domain:
            return "WFP"
        if "nasa" in domain:
            return "NASA POWER"
        if "copernicus" in domain or "era5" in domain:
            return "Copernicus ERA5"
        if "isric" in domain:
            return "ISRIC"
        if "isda" in domain:
            return "iSDA"

    if "faostat" in tid:
        return "FAOSTAT"
    if "fews" in tid:
        return "FEWS NET"
    if "ilri" in tid:
        return "ILRI"
    if "wfp" in tid or "vampire" in tid:
        return "WFP"
    if "nasa_power" in tid or "nasa" in tid:
        return "NASA POWER"
    if "copernicus" in tid or "era5" in tid:
        return "Copernicus ERA5"
    if "isric" in tid:
        return "ISRIC"
    if "isda" in tid:
        return "iSDA"
    if "hdi" in tid:
        return "UNDP / HDI"
    if "gdp" in tid:
        return "World Bank / GDP"
    return None

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
    generate_input_chars: int | None = None


EvidenceTier = Literal["strong", "partial", "empty"]

_TASK_MODE_MAX_TOKENS: dict[str, int] = {
    "fact_lookup": 512,
    "chat": 512,
    "briefing": 768,
    "data_export_only": 256,
    "analytical": 1536,
    "research": 1024,
    "clarify": 512,
}

_TASK_MODE_CONTEXT_CHARS: dict[str, int] = {
    "fact_lookup": 6000,
    "chat": 6000,
    "briefing": 9000,
    "data_export_only": 5000,
    "analytical": 12000,
    "research": 10000,
    "clarify": 4000,
}

_EXPORT_CAPTION_BLOCK = (
    "\n\nARTIFACT EXPORT MODE: "
    + export_caption_instruction().replace("\n", " ")
)

_SLIM_BASE_PROMPT = (
    "You are Ask ADZA, OpenTrace Africa's agricultural intelligence assistant. "
    "Lead with the direct answer. Be precise and compact. "
    "Use only evidence in Context from numbered [Source N] chunks. "
    "Never invent statistics, rankings, or dates. "
    "Never mention warehouses, datasets, table IDs, SQL, vector DBs, or pipelines. "
    "Attribute structured figures to public sources (FAOSTAT, FEWS NET, ILRI, WFP, etc.) "
    "or the neutral label OpenTrace agricultural data (country=…, year=…, product=…). "
    "Put a blank line before every ## heading. "
    "If evidence is insufficient, say so in 2–4 sentences and stop — no report skeleton. "
    "Never open with Unfortunately, Based on the context, or The context provided."
)

_PARTIAL_EVIDENCE_RULES = (
    "\n\nPARTIAL EVIDENCE: Only state claims clearly supported by Context. "
    "One short closing line on years or coverage limits is enough. "
    "No ## Key Findings or multi-section report headings."
)

_YIELD_ONLY_RULE = (
    "\n\nYIELD DATA ONLY: Context rows are crop yields (t/ha), not production tonnage. "
    "Do not describe a single yield observation as a multi-year production trend. "
    "State yield for the specific year shown."
)

_CROP_TOKEN_RE = re.compile(
    r"\b(maize|rice|wheat|sorghum|millet|cocoa|coffee|tomato|cassava|bean|soybean|cotton|"
    r"livestock|cattle|dairy|groundnut|yam|cowpea|barley|tea|sugar)\b",
    re.IGNORECASE,
)

_INTERNAL_ID_RE = re.compile(
    r"\b(?:mart_dev|staging_dev|BQ_DATASET(?:_\w+)?|bigquery)\b|`[^`]+`|\b(?:fct|agg|stg|dim|bridge)_[\w]+\b",
    re.IGNORECASE,
)

_PRODUCTION_TREND_MISLABEL_RE = re.compile(
    r"\bproduction trend\b.*\b(19|20)\d{2}\b.*\b(19|20)\d{2}\b",
    re.IGNORECASE,
)

_NARRATIVE_KINDS = frozenset(
    {
        "news",
        "academic",
        "policy",
        "public_report",
        "ota_insight",
        "formation",
    }
)


def _generate_max_tokens(task_mode: str | None = None) -> int:
    env_ceiling = int(os.environ.get("RAG_GENERATE_MAX_TOKENS", "2048") or 2048)
    mode = (task_mode or "chat").strip().lower()
    default = _TASK_MODE_MAX_TOKENS.get(mode, 512)
    return min(env_ceiling, default)


def _context_max_chars(
    memory_block: str = "",
    *,
    task_mode: str | None = None,
    soft_cap: bool = False,
) -> int:
    mode = (task_mode or "chat").strip().lower()
    base = _TASK_MODE_CONTEXT_CHARS.get(mode, 12000)
    env_override = os.environ.get("RAG_GENERATE_CONTEXT_MAX_CHARS", "").strip()
    if env_override:
        try:
            base = min(base, int(env_override))
        except ValueError:
            pass
    if soft_cap:
        try:
            halved = int(
                os.environ.get(
                    "RAG_GENERATE_CONTEXT_MAX_CHARS_NO_BQ",
                    str(max(4000, base // 2)),
                )
                or base // 2
            )
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
    if meta.get("semantic_row_rejected"):
        return False
    if str(meta.get("source") or "").strip().lower() == "mart_yaml":
        return False
    status = str(meta.get("status") or "").strip().lower()
    if status in {
        "no_project",
        "no_valid_sql",
        "validation_failed",
        "execution_error",
        "empty_result",
        "bq_timeout",
    }:
        return False
    raw = str(item.get("content") or item.get("text") or "").strip().lower()
    return not any(marker in raw for marker in _BQ_FAILURE_MARKERS)


def is_usable_structured_bq_row(item: dict[str, Any]) -> bool:
    """True when a BQ item carries a real numeric fact for generation or export."""
    if not is_usable_context_item(item):
        return False
    if str(item.get("source") or "") != "bigquery":
        return False
    meta = _item_metadata(item)
    semantics = meta.get("value_semantics")
    if not isinstance(semantics, dict):
        return False
    measure_val = semantics.get("measure_value")
    if measure_val is None:
        return False
    if isinstance(measure_val, (int, float)):
        col = str(semantics.get("measure_column") or "").strip().lower()
        if col in ("bq_timeout_s", "bq_timeout"):
            return False
    return True


def _context_has_structured_numeric(context_items: list[dict[str, Any]] | None) -> bool:
    for item in filter_context_items(list(context_items or [])):
        if is_usable_structured_bq_row(item):
            return True
        meta = _item_metadata(item)
        if meta.get("semantic_row_rejected"):
            continue
        semantics = meta.get("value_semantics")
        if isinstance(semantics, dict) and semantics.get("measure_value") is not None:
            col = str(semantics.get("measure_column") or "").strip().lower()
            if col not in ("bq_timeout_s", "bq_timeout"):
                return True
    return False


def _query_topic_tokens(query: str, decomposition: dict[str, Any] | None) -> set[str]:
    tokens: set[str] = set()
    for match in _CROP_TOKEN_RE.finditer(query or ""):
        tokens.add(match.group(1).lower())
    if isinstance(decomposition, dict):
        for raw in decomposition.get("entities") or []:
            text = str(raw).strip().lower()
            if len(text) >= 3:
                tokens.add(text)
    return tokens


def _item_text_blob(item: dict[str, Any]) -> str:
    meta = _item_metadata(item)
    parts = [str(item.get("content") or "")]
    for key in (
        "country",
        "country_name",
        "product",
        "product_name",
        "title",
        "geo_countries",
        "published_at",
    ):
        val = meta.get(key)
        if val is not None and str(val).strip():
            parts.append(str(val))
    return " ".join(parts).lower()


def _narrative_overlaps_query(
    item: dict[str, Any],
    query: str,
    decomposition: dict[str, Any] | None,
) -> bool:
    blob = _item_text_blob(item)
    countries = _query_target_countries(decomposition)
    if countries and not any(c.lower() in blob for c in countries):
        return False
    topics = _query_topic_tokens(query, decomposition)
    if topics:
        return any(t in blob for t in topics)
    return bool(countries)


def _structured_row_on_topic(
    item: dict[str, Any],
    decomposition: dict[str, Any] | None,
) -> bool:
    if not is_usable_structured_bq_row(item):
        return False
    countries = _query_target_countries(decomposition)
    if not countries:
        return True
    blob = _item_text_blob(item)
    return any(c.lower() in blob for c in countries)


def classify_evidence_tier(
    query: str,
    context_items: list[dict[str, Any]] | None,
    decomposition: dict[str, Any] | None = None,
    *,
    structured_bq_numeric_available: bool = False,
    answer_shape: str = "",
) -> EvidenceTier:
    """Classify retrieval context strength before generation."""
    if str(answer_shape or "").strip() == "gap_ack":
        return "empty"
    usable = filter_context_items(list(context_items or []))
    if not usable:
        return "empty"
    if structured_bq_numeric_available:
        return "strong"

    on_topic_structured = sum(
        1 for item in usable if _structured_row_on_topic(item, decomposition)
    )
    on_topic_narrative = sum(
        1
        for item in usable
        if normalize_context_kind(item) in _NARRATIVE_KINDS
        and _narrative_overlaps_query(item, query, decomposition)
    )
    geo_matched = sum(
        1
        for item in usable
        if _narrative_overlaps_query(item, query, decomposition)
        or (
            not _query_target_countries(decomposition)
            or any(
                c.lower() in _item_text_blob(item)
                for c in _query_target_countries(decomposition)
            )
        )
    )

    if on_topic_structured >= 1 or on_topic_narrative >= 2:
        return "strong"

    only_web = all(
        normalize_context_kind(item) in ("web_wikipedia", "web_search") for item in usable
    )
    if only_web and on_topic_narrative == 0:
        return "empty"

    if on_topic_narrative >= 1 or geo_matched >= 1:
        return "partial"

    return "empty"


def context_is_yield_only(context_items: list[dict[str, Any]] | None) -> bool:
    """True when all structured rows are yield measures, not production tonnage."""
    rows = [
        item
        for item in filter_context_items(list(context_items or []))
        if is_usable_structured_bq_row(item)
    ]
    if not rows:
        return False
    for item in rows:
        meta = _item_metadata(item)
        element = str(meta.get("element") or "").lower()
        metric = str(meta.get("metric") or "").lower()
        semantics = meta.get("value_semantics")
        if isinstance(semantics, dict):
            element = element or str(semantics.get("element") or "").lower()
            metric = metric or str(semantics.get("metric") or "").lower()
        if "production" in element and "yield" not in element:
            return False
        if "production" in metric and "yield" not in metric:
            return False
    return True


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


def _item_published_year(item: dict[str, Any]) -> int | None:
    meta = _item_metadata(item)
    for key in ("published_at", "date", "year"):
        raw = str(meta.get(key) or "").strip()
        if len(raw) >= 4 and raw[:4].isdigit():
            year = int(raw[:4])
            if 1900 <= year <= 2100:
                return year
    return None


def _historical_year_window(
    decomposition: dict[str, Any] | None,
) -> tuple[int, int] | None:
    if not isinstance(decomposition, dict):
        return None
    ts = str(decomposition.get("time_start") or "").strip()[:4]
    te = str(decomposition.get("time_end") or "").strip()[:4]
    start = int(ts) if ts.isdigit() else None
    end = int(te) if te.isdigit() else None
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    if start > end:
        start, end = end, start
    if end >= date.today().year:
        return None
    return start, end


def prefer_in_window_narrative(
    items: list[dict[str, Any]],
    decomposition: dict[str, Any] | None,
    *,
    analytical: bool,
) -> list[dict[str, Any]]:
    """For historical analytical queries, pack in-window narrative ahead of later news."""
    if not analytical or not items:
        return items
    window = _historical_year_window(decomposition)
    if window is None:
        return items
    y0, y1 = window
    bq: list[dict[str, Any]] = []
    in_window: list[dict[str, Any]] = []
    out_window: list[dict[str, Any]] = []
    for item in items:
        if _source_kind(item) == "bigquery":
            bq.append(item)
            continue
        year = _item_published_year(item)
        if year is not None and y0 <= year <= y1:
            in_window.append(item)
        else:
            out_window.append(item)
    return bq + in_window + out_window


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
        table_id = str(meta.get("table_id") or _bq_table_from_meta(meta) or "").strip()
        source_name = _public_source_label(table_id, meta)
        hint = _bq_row_hint(meta)
        if source_name:
            return f"[{source_name}] {hint}" if hint else f"[{source_name}]"
        if hint:
            return f"[Structured data] {_BQ_PUBLIC_LABEL} ({hint})"
        return f"[Structured data] {_BQ_PUBLIC_LABEL}"

    if kind in ("academic_article", "academic"):
        raw_title = str(meta.get("article_title") or meta.get("title") or "")
        raw_authors = str(meta.get("authors") or "")
        if (
            len(raw_title) > 300
            or len(raw_authors) > 300
            or "\n" in raw_title
            or "\n" in raw_authors
        ):
            return None
        cite = format_academic_citation(meta)
        line = f"[Academic] {cite}" if cite else None
        return None if line and _looks_like_body_prose(line) else line

    if kind in ("policy_document", "public_report", "policy", "public_report"):
        raw_title = str(meta.get("article_title") or meta.get("title") or "")
        raw_authors = str(meta.get("authors") or "")
        if (
            len(raw_title) > 300
            or len(raw_authors) > 300
            or "\n" in raw_title
            or "\n" in raw_authors
        ):
            return None
        cite = format_academic_citation(meta)
        line = f"[Policy/Public] {cite}" if cite else None
        return None if line and _looks_like_body_prose(line) else line

    if kind in ("news_article", "news"):
        title = str(meta.get("title") or meta.get("source_file") or "").strip()
        src = str(meta.get("source") or meta.get("publisher") or "").strip()
        date = str(meta.get("published_at") or meta.get("date") or "").strip()[:10]
        if title:
            entry = f"{title} — {src}" if src else title
            if date:
                entry += f" ({date})"
            line = f"[News] {entry}"
            return None if _looks_like_body_prose(line) else line
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


def _expand_id_bracket_content(inner: str) -> str:
    """Expand comma/range id lists into sequential plain [N] markers."""
    text = (inner or "").strip()
    if not text:
        return ""
    range_m = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", text)
    if range_m:
        start, end = int(range_m.group(1)), int(range_m.group(2))
        if start <= end and end <= 30 and (end - start) <= 20:
            return "".join(f"[{i}]" for i in range(start, end + 1))
        return ""
    ids = re.findall(r"\d+", text)
    return "".join(f"[{i}]" for i in ids)


def _strip_non_plain_brackets(text: str) -> str:
    """Remove bracket labels that are not plain numeric footnotes."""
    return _NON_PLAIN_BRACKET_RE.sub("", text or "")


def _collapse_duplicate_citation_markers(text: str) -> str:
    """Collapse immediate duplicate footnote markers such as [3][3] or ([3]) [3]."""
    if not text:
        return text
    out = text
    out = re.sub(r"\(\[(\d+)\]\)\s*\[\1\]", r"[\1]", out)
    out = re.sub(r"\[(\d+)\]\s*\(\[\1\]\)", r"[\1]", out)
    prev = None
    while prev != out:
        prev = out
        out = re.sub(r"\[(\d+)\]\s*\[\1\]", r"[\1]", out)
    return out


def _cleanup_citation_spacing(text: str) -> str:
    """Tidy spacing after marker removal without collapsing newlines."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;])", r"\1", text)
    text = re.sub(r"\s+\.", ".", text)
    return text.strip()


def _looks_like_body_prose(line: str) -> bool:
    """True when a citation line looks like dumped body text rather than a one-liner."""
    if not line:
        return True
    if "\n" in line:
        return True
    if len(line) > 300:
        return True
    stripped = line.strip()
    return stripped.startswith("In ,") or stripped.startswith(", ")


def _normalize_markdown_headings(text: str) -> str:
    """Ensure ATX headings start on their own line and split known headings from body."""
    if not text:
        return text
    out = re.sub(r"(\S)\s+(#{2,3}\s)", r"\1\n\n\2", text)
    for heading in _KNOWN_ANALYTICAL_HEADINGS:
        pattern = rf"(#{{2,3}}\s+{re.escape(heading)})\s+(\S.+)"
        out = re.sub(pattern, r"\1\n\n\2", out, flags=re.IGNORECASE)
    return out


def _is_social_url(url: str) -> bool:
    from urllib.parse import urlparse

    host = urlparse(url).netloc.lower()
    return any(s in host for s in _SOCIAL_URL_HOSTS)


def _pick_best_http_url(candidates: list[str]) -> str | None:
    http = [u.strip() for u in candidates if str(u or "").strip().startswith("http")]
    if not http:
        return None
    non_social = [u for u in http if not _is_social_url(u)]
    return non_social[0] if non_social else http[0]


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
    text = _SOURCE_MULTI_REF_RE.sub(lambda m: _expand_id_bracket_content(m.group(1)), text)
    text = re.sub(r"\[Source\s+(\d+)\]", r"[\1]", text, flags=re.IGNORECASE)
    text = _PLAIN_MULTI_REF_RE.sub(lambda m: _expand_id_bracket_content(m.group(1)), text)
    text = _PIPE_LABEL_REF_RE.sub("", text)
    text = re.sub(r"(?<!\[)\bSource\s+(\d+)\b(?!\])", r"[\1]", text, flags=re.IGNORECASE)
    text = _collapse_duplicate_citation_markers(text)
    text = _strip_non_plain_brackets(text)
    return text


def _strip_invalid_citation_markers(answer: str, source_registry: list[SourceRef]) -> str:
    """Remove [N] / [Source N] footnotes that reference missing registry IDs."""
    if not answer:
        return answer
    text = _normalize_inline_citations(answer)
    if not source_registry:
        return _cleanup_citation_spacing(_strip_non_plain_brackets(text))
    valid = {ref.source_id for ref in source_registry}

    def _keep_bracket(m: re.Match[str]) -> str:
        try:
            n = int(m.group(1))
        except (TypeError, ValueError):
            return ""
        return m.group(0) if n in valid else ""

    text = re.sub(r"\[(?!\s*Source\s)(\d+)\]", _keep_bracket, text)
    text = re.sub(r"\[Source\s+(\d+)\]", _keep_bracket, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\[)\bSource\s+(\d+)\b(?!\])", _keep_bracket, text, flags=re.IGNORECASE)
    text = _strip_non_plain_brackets(text)
    return _cleanup_citation_spacing(text)


def _strip_all_inline_citation_markers(answer: str) -> str:
    """Remove every [N] / [Source N] marker when inline footnotes are disabled."""
    if not answer:
        return answer
    text = _normalize_inline_citations(answer)
    text = re.sub(r"\[(?!\s*Source\s)\d+\]", "", text)
    text = re.sub(r"\[Source\s+\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\[)\bSource\s+\d+\b(?!\])", "", text, flags=re.IGNORECASE)
    text = _strip_non_plain_brackets(text)
    return _cleanup_citation_spacing(text)


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
    structured_bq_timed_out: bool = False,
    structured_bq_never_executed: bool = False,
    structured_bq_empty: bool = False,
    structured_bq_validation_failed: bool = False,
    structured_bq_numeric_available: bool = False,
    structured_bq_comparative_available: bool = False,
    analytical_mode: bool = False,
    task_mode: str = "chat",
    measure_id: str | None = None,
    recency_tier: str | None = None,
    context_source_kinds: list[str] | None = None,
    inline_citations: bool = False,
    generation_plan: dict[str, Any] | None = None,
    export_intent: str | None = None,
    evidence_tier: EvidenceTier = "strong",
    yield_only: bool = False,
    composer_addendum: str = "",
) -> list[dict[str, str]]:
    lang = (answer_lang or "").strip() or detect_answer_language(query)
    mode = (task_mode or ("analytical" if analytical_mode else "chat")).strip().lower()
    if analytical_mode and mode == "chat":
        mode = "analytical"

    system = _SLIM_BASE_PROMPT + language_instruction(lang, inline_citations=inline_citations)

    if evidence_tier == "partial":
        system += _PARTIAL_EVIDENCE_RULES
    elif evidence_tier == "empty":
        system += (
            "\n\nINSUFFICIENT EVIDENCE: Reply in 2–4 sentences that you cannot answer "
            "from the sources provided. Do not use report headings or invent facts."
        )

    if inline_citations:
        system += (
            "\n\nCITATION HYGIENE: Use plain [N] only — never [Source N], [Source 1, 3], "
            "or [Wikipedia | …]. Do not repeat the same [N] twice in a row. "
            "Blank line before every ## heading."
        )
    else:
        system += (
            "\n\nDo NOT insert [N] or [Source N] footnote markers — structured citations "
            "are returned separately for the client."
        )

    effective_category = (category or "").strip()
    answer_shape = ""
    use_bullet_layout = False
    if generation_plan and isinstance(generation_plan, dict):
        ec = str(generation_plan.get("effective_category") or "").strip()
        if ec:
            effective_category = ec
        answer_shape = str(generation_plan.get("answer_shape") or "")
        use_bullet_layout = bool(generation_plan.get("use_bullet_layout"))

    register_block = prose_register_addendum(
        effective_category or None,
        task_mode=mode,
        answer_shape=answer_shape,
        inline_citations=inline_citations,
    )
    if register_block:
        system += "\n\n" + register_block

    cat_tone = instruction_for_category(
        effective_category or None,
        measure_id=measure_id,
        recency_tier=recency_tier,
    ) if (effective_category or measure_id or recency_tier) else ""
    if cat_tone:
        system += "\n\n" + cat_tone

    plan_addendum = plan_generation_addendum(plan_type) if plan_type else ""
    if plan_addendum:
        system += "\n\n" + plan_addendum

    if generation_plan:
        from ml.rag.chatbot.generation_plan import generation_plan_addendum

        gen_plan_addendum = generation_plan_addendum(generation_plan)
        if gen_plan_addendum:
            system += "\n\n" + gen_plan_addendum

    if composer_addendum:
        system += "\n\n" + composer_addendum

    from ml.rag.chatbot.generation_plan import format_template_for_shape

    if mode == "fact_lookup":
        system += (
            "\n\nFACT LOOKUP: First sentence = number/fact + unit + year + [N] when citing. "
            "No report headings."
        )
    elif mode == "briefing":
        system += (
            "\n\nBRIEFING: 3–6 bullet points from the most recent narrative sources; "
            "one closing line on limits."
        )
    elif mode == "research":
        system += (
            "\n\nRESEARCH: Synthesize academic/policy evidence; do not invent citations."
        )
    elif mode == "data_export_only":
        system += _EXPORT_CAPTION_BLOCK.replace("ARTIFACT EXPORT MODE", "DATA EXPORT MODE")
    elif answer_shape and evidence_tier != "empty":
        from ml.rag.chatbot.output_format import (
            format_prompt_for_type,
            output_type_from_plan,
            persona_implications_block,
        )

        out_type = output_type_from_plan(generation_plan)
        gw = str((generation_plan or {}).get("grain_window_line") or "").strip()
        answer_subtopics = tuple((generation_plan or {}).get("answer_subtopics") or ())
        has_spine = bool((generation_plan or {}).get("has_usable_spine"))
        persona = effective_category
        plan_type_s = str((generation_plan or {}).get("plan_type") or "").strip()
        include_impl = bool(
            (persona or plan_type_s)
            and evidence_tier != "empty"
            and has_spine
            and out_type != "insufficient"
        )
        impl_text = ""
        if include_impl and persona:
            impl = persona_implications_block(persona, out_type, has_spine=has_spine)
            if impl:
                impl_text = impl
        type_fmt = format_prompt_for_type(
            out_type,
            persona=persona,
            grain_window_line=gw,
            include_implications=include_impl and bool(impl_text),
            answer_subtopics=answer_subtopics,
            implications_text=impl_text,
        )
        if type_fmt:
            system += "\n\n" + type_fmt
        else:
            shape_fmt = format_template_for_shape(answer_shape, use_bullets=use_bullet_layout)
            if shape_fmt:
                system += "\n\n" + shape_fmt

    if export_intent and mode != "data_export_only":
        system += _EXPORT_CAPTION_BLOCK

    if yield_only:
        system += _YIELD_ONLY_RULE

    if effective_category and cat_tone and not is_english_answer_lang(lang) and lang not in ("unknown", ""):
        system += (
            "\n\nAnswer in the user's language while keeping the category audience rules."
        )

    kinds = [str(k).strip().lower() for k in (context_source_kinds or []) if str(k).strip()]
    unique_kinds = list(dict.fromkeys(kinds))
    if evidence_tier == "strong" and len(unique_kinds) >= 2:
        system += (
            "\n\nCross-source synthesis: weave structured figures with narrative sources "
            "when both are relevant; do not invent numbers."
        )

    if structured_bq_numeric_available:
        system = (
            "CRITICAL: Context includes authoritative OpenTrace figures with units. "
            "Use them for numeric answers; do not override with conflicting narrative.\n\n"
        ) + system
    elif structured_bq_comparative_available:
        system = (
            "Context includes OpenTrace structured data — use it for comparisons; "
            "use narrative sources for drivers only.\n\n"
        ) + system
    elif structured_bq_timed_out and is_numeric_data_query(query, decomposition):
        system = (
            "OpenTrace warehouse queries were submitted but did not return rows in time — "
            "say so clearly. Do not invent totals, yields, prices, or rankings.\n\n"
        ) + system
    elif structured_bq_validation_failed and is_numeric_data_query(query, decomposition):
        system = (
            "OpenTrace warehouse SQL failed validation before execution — do not claim warehouse "
            "figures. State the scoped filter could not be executed.\n\n"
        ) + system
    elif structured_bq_never_executed and is_numeric_data_query(query, decomposition):
        system = (
            "OpenTrace warehouse SQL was planned but not submitted — do not claim warehouse "
            "figures. Use narrative sources only or state the data gap.\n\n"
        ) + system
    elif structured_bq_empty and is_numeric_data_query(query, decomposition):
        system = (
            "OpenTrace warehouse returned zero rows for the scoped filters — say so clearly. "
            "Do not invent totals or substitute unrelated web anecdotes.\n\n"
        ) + system
    elif structured_bq_unavailable and is_numeric_data_query(query, decomposition):
        system = (
            "OpenTrace warehouse returned no rows or timed out for the scoped filters — "
            "say so clearly. Do not invent totals, yields, prices, or rankings, and do not "
            "substitute unrelated web product anecdotes.\n\n"
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


def _strip_internal_identifiers(text: str) -> str:
    if not text:
        return text
    cleaned = _INTERNAL_ID_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() or text.strip()


def _strip_forbidden_headings(
    text: str,
    output_type: str = "",
    allowed_subtopics: tuple[str, ...] = (),
) -> str:
    if not text or not output_type:
        return text
    from ml.rag.chatbot.output_format import forbidden_headings_for_type

    forbidden = {
        h.lower()
        for h in forbidden_headings_for_type(
            output_type,  # type: ignore[arg-type]
            allowed_subtopics=allowed_subtopics,
        )
    }
    allowed_subtopic_set = {s.lower() for s in allowed_subtopics}
    if output_type in ("fact", "insufficient", "export"):
        forbidden |= {"trend", "comparison", "context for drivers"}

    heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return text.strip()

    parts: list[str] = []
    if matches[0].start() > 0:
        parts.append(text[: matches[0].start()].rstrip())

    for i, m in enumerate(matches):
        title = m.group(2).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        if title in allowed_subtopic_set:
            parts.append(f"{m.group(0)}{body.rstrip()}")
            continue
        if title in forbidden:
            body_stripped = body.lstrip("\n")
            if "\n\n" in body_stripped:
                _, _, trailing = body_stripped.partition("\n\n")
                if trailing.strip():
                    parts.append(trailing.strip())
            continue
        parts.append(f"{m.group(0)}{body.rstrip()}")

    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(p for p in parts if p)).strip()


def _strip_empty_analytical_skeleton(text: str, evidence_tier: EvidenceTier) -> str:
    if not text:
        return text
    out = text
    if evidence_tier != "strong":
        for heading in _KNOWN_ANALYTICAL_HEADINGS:
            pattern = rf"(#{{2,3}}\s+{re.escape(heading)})\s*\n\s*(?=#{{2,3}}|\Z)"
            out = re.sub(pattern, "", out, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _cap_export_caption(text: str, *, max_sentences: int = 5) -> str:
    if not text or not text.strip():
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()


def _sanitize_citation_text(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for cite in citations:
        row = dict(cite)
        raw = str(row.get("text") or "").strip()
        if not raw:
            continue
        one_line = re.sub(r"\s+", " ", raw)
        if len(one_line) > 200:
            one_line = one_line[:197].rstrip() + "..."
        if one_line.lower() in seen:
            continue
        seen.add(one_line.lower())
        row["text"] = one_line
        out.append(row)
    return out


def _fix_yield_mislabeled_as_production(text: str, *, yield_only: bool) -> str:
    if not yield_only or not text:
        return text
    if _PRODUCTION_TREND_MISLABEL_RE.search(text):
        return re.sub(
            r"\bproduction trend\b",
            "yield",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    return text


def _cap_acf_for_evidence_tier(acf: ACFResult, evidence_tier: EvidenceTier) -> ACFResult:
    if evidence_tier == "empty":
        return no_evidence_acf()
    if evidence_tier != "partial":
        return acf
    if acf.band in ("very_strong", "strong", "moderate"):
        return ACFResult(
            band="limited",
            band_label="Limited confidence",
            score=min(acf.score, 45),
            explanation=acf.explanation,
            note=acf.explanation,
            components=acf.components,
            applied_ceiling="partial_evidence",
            config_version=acf.config_version,
            claim_level=acf.claim_level,
            question_type=acf.question_type,
        )
    return acf


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


def _clean_answer(
    text: str,
    *,
    evidence_tier: EvidenceTier = "strong",
    export_intent: str | None = None,
    yield_only: bool = False,
    output_type: str = "",
    allowed_subtopics: tuple[str, ...] = (),
) -> str:
    """Remove Llama chat template echoes and other non-answer artifacts from LLM output."""
    if not text:
        return text
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    text = re.sub(r"^(Context:|Question:).*$", "", text, flags=re.MULTILINE | re.IGNORECASE).strip()
    text = _normalize_markdown_headings(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_sql_from_answer(text) or text.strip()
    text = _strip_doc_table_figure_labels(text) or text.strip()
    text = _strip_model_sources_appendix(text)
    text = _strip_internal_identifiers(text)
    text = _strip_forbidden_headings(text, output_type, allowed_subtopics=allowed_subtopics)
    text = _strip_empty_analytical_skeleton(text, evidence_tier)
    text = _fix_yield_mislabeled_as_production(text, yield_only=yield_only)
    if export_intent:
        text = _cap_export_caption(text)
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
    """Extract a clickable URL for a citation, trying multiple metadata fields."""
    k = kind.lower()
    url_keys = ("canonical_url", "url", "link", "source_url")

    if k in ("web_wikipedia", "web_search"):
        return _pick_best_http_url([str(meta.get(key) or "") for key in url_keys])

    if k in ("news_article", "news"):
        return _pick_best_http_url([str(meta.get(key) or "") for key in url_keys])

    if k in ("academic_article", "academic", "policy_document", "policy", "public_report"):
        doi = str(meta.get("doi") or "").strip()
        if doi.startswith("http"):
            return doi
        if doi:
            return f"https://doi.org/{doi.lstrip('doi:').strip()}"
        return _pick_best_http_url([str(meta.get(key) or "") for key in url_keys])

    if k in ("ota_insight", "ota_metric"):
        return _pick_best_http_url([str(meta.get(key) or "") for key in url_keys])

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
        if cited_ids:
            return [r for r in source_registry if r.source_id in cited_ids]
        # Author-year-only prose with inline on: attach packed registry rather than [].
        return list(source_registry)
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


def _compact_gap_message(
    query: str,
    decomposition: dict[str, Any] | None = None,
    answer_lang: str | None = None,
) -> str:
    """Short 2–4 sentence gap when evidence tier is EMPTY."""
    lang = (answer_lang or "").strip() or detect_answer_language(query)
    lead = insufficient_context_answer(lang=lang, query=query)
    geo = _decomposition_geo_hint(decomposition)
    if geo:
        detail = f"Name a specific crop or metric and time period for {geo}."
    else:
        detail = "Name a specific country, crop or metric, and time period."
    return f"{lead} {detail}"


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
    generate_input_chars: int | None = None,
    evidence_tier: EvidenceTier = "strong",
    export_intent: str | None = None,
    yield_only: bool = False,
    output_type: str = "",
    allowed_subtopics: tuple[str, ...] = (),
    bq_exec_flags: dict[str, bool] | None = None,
    usable_bq: bool = False,
    bq_sql_debug: list[dict[str, Any]] | None = None,
) -> GenerationResult:
    """Attach structured citations and score ACF Path B on cited sources only."""
    t0 = time.perf_counter()
    with observed_span("citations", input_data={"registry_size": len(source_registry)}):
        prose = _strip_model_sources_appendix(answer)
        if inline_citations:
            prose = _strip_invalid_citation_markers(prose, source_registry)
        else:
            prose = _strip_all_inline_citation_markers(prose)
        prose = _clean_answer(
            prose,
            evidence_tier=evidence_tier,
            export_intent=export_intent,
            yield_only=yield_only,
            output_type=output_type,
            allowed_subtopics=allowed_subtopics,
        )
        citations = referenced_citations(
            prose, source_registry, inline_citations=inline_citations
        )
        citations = _sanitize_citation_text(citations)
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

        acf = _cap_acf_for_evidence_tier(acf, evidence_tier)
        acf = apply_bq_execute_ceiling(
            acf,
            bq_exec_flags,
            usable_bq=usable_bq,
            bq_sql_debug=list(bq_sql_debug or []),
        )

        update_current_span_metadata(
            {
                "citation_count": len(citations),
                "registry_size": len(source_registry),
                "citations_mode": _citations_mode(),
                "inline_citations": inline_citations,
                "cited_ids": cited_ids,
                "kind_counts": kind_counts,
                "latency_ms": trace_elapsed_ms(t0),
                "generate_ms": trace_elapsed_ms(t0),
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
        return GenerationResult(
            answer=prose,
            citations=citations,
            acf=acf,
            generate_input_chars=generate_input_chars,
        )


def _call_llama(
    messages: list[dict[str, str]],
    *,
    purpose: str = "generate",
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Call configured LLM backend; never raises on HTTP errors.

    model: override the global RAG_LLM_MODEL_ID (used for per-plan routing, ML-041).
    When None, falls back to llm_model_id() which reads RAG_LLM_MODEL_ID from env.
    """
    gen_timeout = float(os.environ.get("RAG_GENERATE_TIMEOUT_S", "0") or 0) or llm_default_timeout_s()
    max_toks = max_tokens if max_tokens is not None else _generate_max_tokens()
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
    structured_bq_timed_out = bool(kwargs.get("structured_bq_timed_out"))
    structured_bq_never_executed = bool(kwargs.get("structured_bq_never_executed"))
    structured_bq_empty = bool(kwargs.get("structured_bq_empty"))
    structured_bq_validation_failed = bool(kwargs.get("structured_bq_validation_failed"))
    pre_queries = list(kwargs.get("pre_queries") or [])
    usable_bq = bool(kwargs.get("usable_bq"))
    bq_sql_debug = list(kwargs.get("bq_sql_debug") or [])
    bq_exec_flags = {
        k: bool(kwargs.get(k))
        for k in (
            "structured_bq_timed_out",
            "structured_bq_never_executed",
            "structured_bq_empty",
            "structured_bq_validation_failed",
            "structured_bq_unavailable",
        )
        if kwargs.get(k)
    }
    structured_bq_numeric_available = bool(kwargs.get("structured_bq_numeric_available"))
    structured_bq_comparative_available = bool(kwargs.get("structured_bq_comparative_available"))
    analytical_mode = bool(kwargs.get("analytical_mode"))
    task_mode = str(kwargs.get("task_mode") or ("analytical" if analytical_mode else "chat")).strip()
    measure_id = str(kwargs.get("measure_id") or "").strip() or None
    recency_tier = str(kwargs.get("recency_tier") or "").strip() or None
    export_intent = kwargs.get("export_intent")
    export_intent_s = str(export_intent).strip() if export_intent else None
    composer_addendum = str(kwargs.get("composer_addendum") or "").strip()
    generation_plan = kwargs.get("generation_plan")
    if generation_plan is not None and not isinstance(generation_plan, dict):
        generation_plan = None
    inline_citations = want_inline_citations(
        query,
        task_mode=task_mode,
        export_intent=export_intent_s,
    )

    if structured_bq_unavailable and is_numeric_data_query(query, decomposition) and not pre_queries:
        usable_preview = filter_context_items(context_items or [])
        has_narrative = any(is_usable_context_item(item) for item in usable_preview)
        if not has_narrative:
            return GenerationResult(
                answer=_no_data_fallback_message(query, decomposition),
                citations=[],
                acf=no_evidence_acf(
                    explanation=(
                        "No OpenTrace sources matched this numeric question."
                    )
                ),
            )

    shape = str((generation_plan or {}).get("answer_shape") or "")
    from ml.rag.chatbot.output_format import output_type_from_plan

    output_type = output_type_from_plan(generation_plan)
    answer_subtopics = tuple((generation_plan or {}).get("answer_subtopics") or ())
    if (
        task_mode == "fact_lookup"
        and shape == "numeric_fact"
        and is_numeric_data_query(query, decomposition)
        and not _PRICE_TREND_RE.search(query or "")
        and not _context_has_structured_numeric(context_items)
    ):
        usable = filter_context_items(context_items or [])
        has_narrative = any(
            normalize_context_kind(it) in _NARRATIVE_KINDS for it in usable
        )
        if not has_narrative:
            return GenerationResult(
                answer=_no_data_fallback_message(query, decomposition),
                citations=[],
                acf=no_evidence_acf(
                    explanation=(
                        "No structured OpenTrace numeric data matched this question."
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
                structured_bq_timed_out=structured_bq_timed_out,
                structured_bq_never_executed=structured_bq_never_executed,
                structured_bq_empty=structured_bq_empty,
                structured_bq_validation_failed=structured_bq_validation_failed,
                structured_bq_numeric_available=structured_bq_numeric_available,
                structured_bq_comparative_available=structured_bq_comparative_available,
                analytical_mode=analytical_mode,
                task_mode=task_mode,
                measure_id=measure_id,
                recency_tier=recency_tier,
                inline_citations=inline_citations,
                generation_plan=generation_plan,
                export_intent=export_intent_s,
                evidence_tier="empty",
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = (
                    "CRITICAL: The Context below is empty ('[No external context]'). "
                    "Reply in 2–4 sentences only that you cannot answer from sources.\n\n"
                ) + messages[0]["content"]
            gen_max = _generate_max_tokens(task_mode)
            input_chars = sum(len(str(m.get("content") or "")) for m in messages)
            llama_answer = _call_llama(
                messages,
                purpose="generate",
                model=model_for_plan(plan_type),
                max_tokens=gen_max,
            )
            if llama_answer:
                cleaned = _normalize_inline_citations(
                    _clean_answer(llama_answer, evidence_tier="empty")
                )
                return _finalize_generation_result(
                    cleaned,
                    [],
                    query=query,
                    decomposition=decomposition,
                    inline_citations=inline_citations,
                    generate_input_chars=input_chars,
                    evidence_tier="empty",
                    bq_exec_flags=bq_exec_flags,
                    usable_bq=usable_bq,
                    bq_sql_debug=bq_sql_debug,
                )
        # Default (RAG_ALLOW_UNGROUNDED off or LLM returned nothing): structured
        # gap message so testers can distinguish "no data" from "low confidence".
        return GenerationResult(
            answer=_compact_gap_message(query, decomposition, answer_lang),
            citations=[],
            acf=no_evidence_acf(),
        )

    usable_context = dedupe_bq_context_items(
        usable_context_after_geo_purity(context_items, decomposition)
    )
    usable_context = dedupe_context_items(usable_context)
    usable_context = prefer_in_window_narrative(
        usable_context,
        decomposition,
        analytical=analytical_mode or task_mode == "analytical",
    )

    plan_tier = str((generation_plan or {}).get("evidence_tier") or "").strip()
    if plan_tier in ("strong", "partial", "empty"):
        evidence_tier: EvidenceTier = plan_tier  # type: ignore[assignment]
    else:
        evidence_tier = classify_evidence_tier(
            query,
            usable_context,
            decomposition,
            structured_bq_numeric_available=structured_bq_numeric_available,
            answer_shape=shape,
        )

    yield_only = context_is_yield_only(usable_context)

    min_usable = _min_usable_context()
    if len(usable_context) <= min_usable or evidence_tier == "empty":
        if task_mode == "data_export_only" and export_intent_s:
            pass
        elif pre_queries and bq_exec_flags:
            pass
        else:
            return GenerationResult(
                answer=_compact_gap_message(query, decomposition, answer_lang),
                citations=[],
                acf=no_evidence_acf(),
            )

    ctx_budget = _context_max_chars(
        memory_block,
        task_mode=task_mode,
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
        structured_bq_timed_out=structured_bq_timed_out,
        structured_bq_never_executed=structured_bq_never_executed,
        structured_bq_empty=structured_bq_empty,
        structured_bq_validation_failed=structured_bq_validation_failed,
        structured_bq_numeric_available=structured_bq_numeric_available,
        structured_bq_comparative_available=structured_bq_comparative_available,
        analytical_mode=analytical_mode,
        task_mode=task_mode,
        measure_id=measure_id,
        recency_tier=recency_tier,
        context_source_kinds=source_kinds,
        inline_citations=inline_citations,
        generation_plan=generation_plan,
        export_intent=export_intent_s,
        evidence_tier=evidence_tier,
        yield_only=yield_only,
        composer_addendum=composer_addendum,
    )
    gen_max = _generate_max_tokens(task_mode)
    input_chars = sum(len(str(m.get("content") or "")) for m in messages)
    update_current_span_metadata(
        {
            "generate_max_tokens": gen_max,
            "generate_input_chars": input_chars,
            "task_mode": task_mode,
            "evidence_tier": evidence_tier,
        }
    )
    llama_answer = _call_llama(
        messages,
        purpose="generate",
        model=model_for_plan(plan_type),
        max_tokens=gen_max,
    )
    if llama_answer:
        cleaned = _normalize_inline_citations(
            _clean_answer(
                llama_answer,
                evidence_tier=evidence_tier,
                export_intent=export_intent_s,
                yield_only=yield_only,
                output_type=output_type,
                allowed_subtopics=answer_subtopics,
            )
        )
        return _finalize_generation_result(
            cleaned,
            source_registry,
            query=query,
            decomposition=decomposition,
            inline_citations=inline_citations,
            generate_input_chars=input_chars,
            evidence_tier=evidence_tier,
            export_intent=export_intent_s,
            yield_only=yield_only,
            output_type=output_type,
            allowed_subtopics=answer_subtopics,
            bq_exec_flags=bq_exec_flags,
            usable_bq=usable_bq,
            bq_sql_debug=bq_sql_debug,
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
        answer=hint + _compact_gap_message(query, decomposition, answer_lang),
        citations=[],
        acf=no_evidence_acf(
            explanation="Generation failed before citations could be scored."
        ),
        generate_input_chars=input_chars,
    )
