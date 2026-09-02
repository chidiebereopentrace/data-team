"""
RAG graph: query → decompose → parallel retrieval (six Qdrant corpora)
→ BQ SQL reasoner (mart_dev YAML) → BigQuery → merge → rerank → generate.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any, TypedDict, cast

from ml.rag.chatbot.acf_scoring import acf_result_to_state, curated_product_acf, no_evidence_acf
from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    detect_canned_insufficient_lang,
    insufficient_context_answer,
    language_unclear_answer,
)
from ml.rag.chatbot.memory_relevance import memory_relevant_for_query
from ml.rag.chatbot.assistant_identity import is_meta_query
from ml.rag.chatbot.ofia import infer_source_tier
from ml.rag.chatbot.export_intent import EXPORT_UPGRADE_MESSAGE, detect_export_intent
from ml.rag.chatbot.geo_regions import expand_regions_in_decomposition
from ml.rag.chatbot.export_runner import run_exports
from ml.rag.chatbot.geo_policy import effective_geo_override
from ml.rag.chatbot.plan_policy import (
    apply_category_domain_hints,
    apply_plan_decomposition_gates,
)
from ml.rag.chatbot.task_mode import clarify_answer, resolve_task_mode
from ml.rag.chatbot.query_enricher import enrich_query_with_memory
from ml.rag.chatbot.agri_measure_ontology import MEASURES, MeasureHit, resolve_measure, resolve_measures, resolve_recency_tier
from ml.rag.chatbot.product_knowledge import is_help_query, is_product_query
from ml.rag.chatbot.query_gate import (
    classify_social_query,
    early_non_rag_route,
    generate_social_answer,
    is_greeting_query,
    is_out_of_scope_query,
)
from ml.rag.chatbot.bq_byte_budget import trim_bq_result_contents
from ml.rag.chatbot.bq_context_enrich import enrich_bq_results
from ml.rag.chatbot.bq_ranking_cache import (
    bq_results_from_cache,
    cache_entry_from_bq_results,
    is_ranking_follow_up,
)
from ml.rag.chatbot.bq_sql_reasoner import reason_bq_sql_plan, _slot_reasoner_active
from ml.rag.chatbot.class_engine_runner import engine_results_to_bq_plan, run_class_engines
from ml.rag.chatbot.sql_compiler import sql_compiler_enabled
from ml.rag.chatbot.class_supervisor import SupervisorPlan, compile_supervisor_plan, corpora_for_supervisor_plan, tables_for_supervisor_plan
from ml.rag.chatbot.context_diversity import diversify_context_pack
from ml.rag.chatbot.corpus_catalog import CORPUS_CATALOG, CorpusSelection, select_corpora
from ml.rag.chatbot.facet_enrich import enrich_decomposition_facets
from ml.rag.chatbot.facet_compiler import compile_turn_contract, compile_breakdown
from ml.rag.chatbot.intent_bundles import bundle_primary_measures, bundle_required_measures, match_intent_bundles
from ml.rag.chatbot.capability_registry import apply_reasoner_to_turn, resolve_capability, unsupported_answer_hint
from ml.rag.chatbot.empty_answer_templates import empty_answer_for_contract
from ml.rag.chatbot.empty_policy import (
    build_filter_miss_block,
    primary_class_from_state,
    resolve_empty_policy,
    should_generate_weak,
)
from ml.rag.chatbot.bq_gap_messages import (
    first_prep_error,
    forbid_generic_insufficient,
    should_hard_return_bq_gap,
    typed_bq_gap_answer,
    warehouse_was_attempted,
)
from ml.rag.chatbot.bq_execute_state import bq_execute_flags
from ml.rag.chatbot.time_retrieval import sync_decomposition_time, time_fallback_enabled, time_kwargs_from_contract
from ml.rag.chatbot.turn_contract import NUMERIC_JOBS, TurnContract
from ml.rag.chatbot.typed_pack import typed_context_pack, should_zero_pack
from ml.rag.chatbot.generation_plan import build_generation_plan
from ml.rag.chatbot.global_reasoner import compile_reasoner_plan, reasoner_plan_to_bq_plan
from ml.rag.chatbot.composer import (
    composer_addendum,
    composer_context_block,
    partition_bq_by_subquestion,
)
from ml.rag.chatbot.reasoner_plan import ReasonerPlan, is_heavy_plan_type
from ml.rag.chatbot.qdrant_planner import plan_vector_corpora_for_reasoner
from ml.rag.chatbot.generator import (
    _generate_max_tokens,
    filter_context_items,
    generate,
    is_comparative_bq_query,
    is_numeric_data_query,
    is_usable_context_item,
    is_usable_structured_bq_row,
    pin_bq_context_first,
    should_elevate_bq_context,
)
from ml.rag.chatbot.ontology_context import sanitize_decomposition_for_bq
from ml.rag.chatbot.retrieval_contract import build_retrieval_contract
from ml.rag.chatbot.query_decomposer import (
    decompose_query,
    normalize_geography_for_filter,
    resolve_retrieval_geographies,
)
from ml.rag.chatbot.query_normalize import normalize_query_text
from ml.rag.chatbot.routing_plan import RoutingPlan, build_routing_plan
from ml.rag.chatbot.reranker import last_rerank_mode, rerank
from ml.rag.retrievers.bq_retriever import BQRetriever
from ml.rag.retrievers.vector_retriever import VectorRetriever
from ml.rag.llm_chat import get_llm_usage, reset_llm_usage
from ml.rag.retrievers.web_retriever import (
    _warehouse_attempt_blocks_web,
    format_web_chunk_for_context,
    needs_web_fallback,
    retrieve_web_fallback_detailed,
    route_after_rerank,
)

from ml.rag.text_processors.preprocess.bibliographic_metadata import format_academic_citation
from ml.rag.observability import (
    build_rag_invoke_config,
    observed_span,
    run_with_tracing_context,
    trace_elapsed_ms,
    update_current_span_metadata,
)

logger = logging.getLogger(__name__)

_DEFAULT_NEWS_TOP_K = 12
_DEFAULT_ACADEMIC_TOP_K = 10
_DEFAULT_OTA_TOP_K = 10
_DEFAULT_BQ_TOP_K = 12
_DEFAULT_RERANK_TOP_K = 18
_DEFAULT_RERANK_POOL_SIZE = 24
_FACT_LOOKUP_RERANK_TOP_K = 12
_FACT_LOOKUP_RERANK_POOL = 14
_BRIEFING_RERANK_TOP_K = 14
_BRIEFING_RERANK_POOL = 18

_BRIEFING_NEWS_TOP_K = 16
_BRIEFING_OTA_TOP_K = 12
_FACT_LOOKUP_NEWS_TOP_K = 10
_FACT_LOOKUP_ACADEMIC_TOP_K = 8
_DECISION_SUPPORT_OTA_TOP_K = 12


def _env_on(name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "on", "yes")


def _kwargs_attempt_key(kwargs: dict[str, Any]) -> str:
    """Hashable key for deduplicating cascade attempts (doc_kinds is a list)."""

    def _norm(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    return json.dumps({k: _norm(v) for k, v in sorted(kwargs.items())}, sort_keys=True, default=str)


def _strict_compare_filters(dec: dict[str, Any]) -> bool:
    """Compare / multi-country queries should not drop geo filters for irrelevant news."""
    if str(dec.get("intent") or "").strip().lower() != "compare":
        return False
    geo = dec.get("geography")
    return isinstance(geo, list) and len(normalize_geography_for_filter(geo)) >= 2


def _chunk_metadata(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("metadata")
    return raw if isinstance(raw, dict) else {}


def _post_filter_geography(items: list[dict[str, Any]], countries: list[str]) -> list[dict[str, Any]]:
    """
    Keep chunks whose geo metadata or content mentions at least one target country.
    Uses set intersection on normalized geo lists (preferred) + safe whole-word fallback on content.
    Avoids "niger" matching "nigeria".
    """
    if not countries:
        return items

    import re

    def _norm_list(s: str) -> set[str]:
        if not s:
            return set()
        parts = re.split(r"[;,/]", s)
        return {p.strip().lower() for p in parts if p.strip()}

    def _contains_whole_word(text: str, word: str) -> bool:
        if not text or not word:
            return False
        pattern = r"\b" + re.escape(word) + r"\b"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    allowed = {c.strip().lower() for c in countries if c.strip()}
    if not allowed:
        return items

    out: list[dict[str, Any]] = []
    for item in items:
        meta = _chunk_metadata(item)
        primary = str(meta.get("geo_country_primary") or meta.get("country") or "")
        blob = str(meta.get("geo_countries") or "")
        content = str(item.get("content") or "")

        meta_countries = _norm_list(primary) | _norm_list(blob)

        if meta_countries & allowed:
            out.append(item)
            continue

        # Content fallback: whole-word only (safer than substring)
        if any(_contains_whole_word(content, c) for c in allowed):
            out.append(item)

    return out


def _merge_dedupe_vector_hits(batches: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for batch in batches:
        for h in batch:
            meta = _chunk_metadata(h)
            content_preview = str(h.get("content") or "")[:120]
            key = str(meta.get("id") or meta.get("document_id") or content_preview)
            score = float(h.get("score") or 0.0)
            prev = best.get(key)
            if prev is None or score > float(prev.get("score") or 0.0):
                best[key] = h
    ranked = sorted(best.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return ranked[:top_k]


def _news_time_filter_at_qdrant() -> bool:
    """When false (default), date range is applied in Python so undated news is not dropped."""
    return os.environ.get("RAG_NEWS_TIME_QDRANT_FILTER", "").strip().lower() in ("1", "true", "on", "yes")


def _apply_geo_to_kwargs(kwargs: dict[str, Any], countries: list[str]) -> None:
    kwargs.pop("geo_country", None)
    kwargs.pop("geo_countries", None)
    if len(countries) >= 2:
        kwargs["geo_countries"] = countries
    elif len(countries) == 1:
        kwargs["geo_country"] = countries[0]


def _widen_time_kwargs(base_kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Expand published_at range by ±1 calendar year; None if no time bounds."""
    ts = str(base_kwargs.get("published_at_from") or "").strip()[:10]
    te = str(base_kwargs.get("published_at_to") or "").strip()[:10]
    if not ts and not te:
        return None
    out = dict(base_kwargs)
    if ts and len(ts) >= 4 and ts[:4].isdigit():
        out["published_at_from"] = f"{max(1900, int(ts[:4]) - 1)}{ts[4:]}"
    if te and len(te) >= 4 and te[:4].isdigit():
        out["published_at_to"] = f"{min(2100, int(te[:4]) + 1)}{te[4:]}"
    if out.get("published_at_from") == base_kwargs.get("published_at_from") and out.get(
        "published_at_to"
    ) == base_kwargs.get("published_at_to"):
        return None
    return out


def _stamp_constraint_relaxed(
    items: list[dict[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    if label == "none" or not items:
        return items
    out: list[dict[str, Any]] = []
    for item in items:
        meta = dict(_chunk_metadata(item))
        meta["constraint_relaxed"] = label
        out.append({**item, "metadata": meta})
    return out


def _cascade_attempt_label(base_kwargs: dict[str, Any], kwargs: dict[str, Any]) -> str:
    base_geo = bool(base_kwargs.get("geo_country") or base_kwargs.get("geo_countries"))
    base_time = bool(base_kwargs.get("published_at_from") or base_kwargs.get("published_at_to"))
    has_geo = bool(kwargs.get("geo_country") or kwargs.get("geo_countries"))
    has_time = bool(kwargs.get("published_at_from") or kwargs.get("published_at_to"))
    if has_geo == base_geo and has_time == base_time:
        # Distinguish time_widen from full by comparing year bounds.
        if (
            str(kwargs.get("published_at_from") or "") != str(base_kwargs.get("published_at_from") or "")
            or str(kwargs.get("published_at_to") or "") != str(base_kwargs.get("published_at_to") or "")
        ):
            return "time_widen"
        return "none"
    if not has_geo and not has_time and base_geo and base_time:
        return "no_geo_time"
    if not has_geo and base_geo and has_time == base_time:
        return "no_geo"
    if not has_time and base_time and has_geo == base_geo:
        return "no_time"
    if not has_geo and not has_time:
        return "no_geo_time"
    return "relaxed"


def _retrieve_vector_cascade(
    vr: VectorRetriever,
    query: str,
    *,
    base_kwargs: dict[str, Any],
    countries: list[str],
    has_time: bool,
    geo_fallback_env: str,
    time_fallback_env: str,
    allow_geo_fallback: bool = True,
    max_levels: int | None = None,
    allow_time_fallback: bool | None = None,
) -> list[dict[str, Any]]:
    """Try full filters, widen time ±1y, then drop time/geo when fallbacks allow."""
    attempts: list[dict[str, Any]] = [dict(base_kwargs)]
    has_geo = bool(countries)
    time_fb = allow_time_fallback if allow_time_fallback is not None else _env_on(time_fallback_env)

    widened = _widen_time_kwargs(base_kwargs) if has_time else None
    if widened is not None:
        attempts.append(widened)

    if has_time and time_fb:
        no_time = dict(base_kwargs)
        no_time.pop("published_at_from", None)
        no_time.pop("published_at_to", None)
        attempts.append(no_time)

    if has_geo and allow_geo_fallback and _env_on(geo_fallback_env):
        no_geo = dict(base_kwargs)
        no_geo.pop("geo_country", None)
        no_geo.pop("geo_countries", None)
        attempts.append(no_geo)

    if (
        has_geo
        and has_time
        and allow_geo_fallback
        and _env_on(geo_fallback_env)
        and time_fb
    ):
        relaxed = dict(base_kwargs)
        relaxed.pop("geo_country", None)
        relaxed.pop("geo_countries", None)
        relaxed.pop("published_at_from", None)
        relaxed.pop("published_at_to", None)
        attempts.append(relaxed)

    seen: set[str] = set()
    levels_tried = 0
    for kwargs in attempts:
        if max_levels is not None and levels_tried >= max_levels:
            break
        levels_tried += 1
        key = _kwargs_attempt_key(kwargs)
        if key in seen:
            continue
        seen.add(key)
        try:
            raw = vr.retrieve(query, **kwargs)
        except Exception:
            logger.warning(
                "Vector retrieve failed for %s (filters=%s)",
                vr.collection_name,
                {k: kwargs[k] for k in kwargs if k not in ("top_k", "vector_search_mode", "doc_kind", "doc_kinds")},
                exc_info=True,
            )
            raw = []
        if raw:
            label = _cascade_attempt_label(base_kwargs, kwargs)
            if label != "none":
                logger.debug(
                    "Vector retrieve cascade for %s: %d hits (relaxed=%s)",
                    vr.collection_name,
                    len(raw),
                    label,
                )
            return _stamp_constraint_relaxed(raw, label=label)
    return []


def _vector_retrieve_for_corpus(
    state: RAGGraphState,
    *,
    collection_env: str,
    default_collection: str,
    build_kwargs: Any,
    geo_fallback_env: str,
    time_fallback_env: str,
) -> list[dict[str, Any]]:
    """Shared retrieval: decomposition geo/time + optional multi-country filter."""
    q = (state.get("query") or "").strip()
    dec_raw = state.get("decomposition") or {}
    dec = sync_decomposition_time(
        dec_raw if isinstance(dec_raw, dict) else {},
        TurnContract.from_dict(state.get("turn_contract") if isinstance(state.get("turn_contract"), dict) else None),
    )
    coll = os.environ.get(collection_env, default_collection).strip() or default_collection
    vr = VectorRetriever(collection_name=coll)

    countries = resolve_retrieval_geographies(
        geo_override=str(state.get("geo_override") or ""),
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )
    contract = TurnContract.from_dict(state.get("turn_contract") if isinstance(state.get("turn_contract"), dict) else None)
    time_kw = time_kwargs_from_contract(contract if contract.measure_id or contract.time_spec.start else None)
    ts = str(
        time_kw.get("published_at_from")
        or state.get("time_start_override")
        or dec.get("time_start")
        or ""
    ).strip()[:10]
    te = str(
        time_kw.get("published_at_to")
        or state.get("time_end_override")
        or dec.get("time_end")
        or ""
    ).strip()[:10]
    hard_filter = bool(time_kw.get("hard_filter"))

    kwargs = build_kwargs(state, dec)
    _apply_geo_to_kwargs(kwargs, countries)
    if ts:
        kwargs["published_at_from"] = ts
    if te:
        kwargs["published_at_to"] = te

    allow_geo_fb = not _strict_compare_filters(dec)
    task_mode = str(state.get("task_mode") or "chat").strip().lower()
    cascade_max = 1 if task_mode in ("fact_lookup", "data_export_only") else None
    time_fb = time_fallback_enabled(hard_filter=hard_filter, env_var=time_fallback_env)
    raw = _retrieve_vector_cascade(
        vr,
        q,
        base_kwargs=kwargs,
        countries=countries,
        has_time=bool(ts or te),
        geo_fallback_env=geo_fallback_env,
        time_fallback_env=time_fallback_env,
        allow_geo_fallback=allow_geo_fb,
        max_levels=cascade_max,
        allow_time_fallback=time_fb,
    )
    if countries:
        return _post_filter_geography(raw, countries)
    return raw


class RAGGraphState(TypedDict, total=False):
    query: str
    decomposition: dict[str, Any]
    bq_table_candidates: list[dict[str, Any]]
    bq_sql_plan: dict[str, Any]
    vector_news_results: list[dict[str, Any]]
    vector_academic_papers_results: list[dict[str, Any]]
    vector_policies_results: list[dict[str, Any]]
    vector_public_reports_results: list[dict[str, Any]]
    vector_formation_results: list[dict[str, Any]]
    vector_ota_results: list[dict[str, Any]]
    vector_academic_results: list[dict[str, Any]]  # deprecated alias; prefer per-corpus keys
    vector_results: list[dict[str, Any]]
    bq_results: list[dict[str, Any]]
    bq_sql_queries: list[str]
    bq_sql_debug: list[dict[str, Any]]
    structured_ranking_cache: dict[str, Any] | None
    bq_cache_hit: bool | None
    merged_context: list[dict[str, Any]]
    reranked_context: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    web_fallback_status: str | None
    web_fallback_reason: str | None
    insufficient_context: bool | None
    answer: str
    citations: list[dict[str, Any]]
    usage: dict[str, int]
    error: str | None
    # Optional UI / API overrides (see run_rag)
    geo_override: str | None
    time_start_override: str | None
    time_end_override: str | None
    news_top_k: int | None
    academic_top_k: int | None
    ota_top_k: int | None
    bq_top_k: int | None
    rerank_top_k: int | None
    # Generator memory: rolling summary + last N verbatim pairs (see ml.rag.chat_memory)
    conversation_summary: str | None
    recent_turns: list[dict[str, Any]] | None
    chat_history: list[dict[str, Any]] | None  # legacy: verbatim-only, no summary
    is_meta_query: bool | None
    is_product_query: bool | None
    is_help_query: bool | None
    is_greeting_query: bool | None
    is_out_of_scope_query: bool | None
    is_language_unknown: bool | None
    plan_type: str | None
    category: str | None
    user_profile: dict[str, Any] | None
    corpus_selection: dict[str, Any] | None
    # ACF Path B (ADZA Confidence Framework) — cited evidence scoring
    acf_band: str | None
    acf_band_label: str | None
    acf_score: int | float | None
    acf_note: str | None
    acf_explanation: str | None
    acf_components: dict[str, Any] | None
    acf_applied_ceiling: str | None
    acf_config_version: str | None
    acf_claim_level: str | None
    acf_question_type: str | None
    # Named answer-language tag (en|sw|fr|pcm|ar|am|ig|…|mixed|unknown)
    answer_lang: str | None
    # Session context — Sprint 1, Week 2 (session isolation)
    session_id: str | None
    # Export / enriched outputs (Agribusinesses + Integrated routes only)
    export_enabled: bool | None
    export_intent: str | None
    artifacts: list[dict[str, Any]]
    # Task specialization for full_rag (clarify / analytical / fact / briefing / export-only / chat)
    task_mode: str | None
    # analytical_mode is True when task_mode == "analytical"
    analytical_mode: bool | None
    # Enriched query when memory merges elliptical follow-ups
    enriched_query: str | None
    measure_id: str | None
    recency_tier: str | None
    turn_contract: dict[str, Any]
    routing_plan: dict[str, Any]
    supervisor_plan: dict[str, Any] | None
    generation_plan: dict[str, Any] | None
    reasoner_plan: dict[str, Any] | None
    rows_by_subquestion: dict[str, Any] | None
    passages: list[dict[str, Any]] | None
    heavy_path: bool | None
    slot_path: bool | None
    # Latency / cost observability (internal; not part of the public API schema)
    route_candidate: str | None
    early_short_circuit: bool | None
    skipped_decompose_llm: bool | None
    skipped_retrieval: bool | None
    decompose_llm_ms: float | None
    vector_ms: float | None
    corpus_count: int | None
    cascade_level: int | None
    bq_nl2sql_ms: float | None
    bq_execute_ms: float | None
    sql_source: str | None
    rerank_ms: float | None
    rerank_pool_size: int | None
    rerank_mode: str | None
    generate_ms: float | None
    generate_max_tokens: int | None
    generate_input_chars: int | None


def _reasoner_from_state(state: RAGGraphState) -> ReasonerPlan | None:
    raw = state.get("reasoner_plan")
    return ReasonerPlan.from_dict(raw if isinstance(raw, dict) else None)


def _slot_path_active(state: RAGGraphState) -> bool:
    if not state.get("slot_path"):
        return False
    rp = _reasoner_from_state(state)
    return rp is not None and bool(rp.bq_subquestions())


def _early_route_decompose_state(raw_q: str, route: str) -> dict[str, Any]:
    """Build decompose output for early non-RAG short-circuit on raw user text."""
    answer_lang = detect_answer_language(raw_q)
    export_intent = detect_export_intent(raw_q)
    base: dict[str, Any] = {
        "decomposition": {},
        "is_meta_query": route == "meta",
        "is_product_query": route in ("product", "help"),
        "is_help_query": route == "help",
        "is_greeting_query": route == "greeting",
        "is_out_of_scope_query": route == "out_of_scope",
        "is_language_unknown": False,
        "answer_lang": answer_lang,
        "export_intent": export_intent,
        "task_mode": "chat",
        "analytical_mode": False,
        "route_candidate": route,
        "early_short_circuit": True,
        "skipped_decompose_llm": True,
        "skipped_retrieval": True,
        "decompose_llm_ms": 0.0,
    }
    return base


def node_decompose(state: RAGGraphState) -> dict[str, Any]:
    raw_q = normalize_query_text((state.get("query") or "").strip())
    route = early_non_rag_route(raw_q)
    if route:
        answer_lang = detect_answer_language(raw_q)
        export_intent = detect_export_intent(raw_q)
        with observed_span(
            "decompose",
            input_data={"query": raw_q[:200], "enriched": False, "early_short_circuit": True},
        ):
            update_current_span_metadata(
                {
                    "route_candidate": route,
                    "answer_lang": answer_lang,
                    "export_intent": export_intent,
                    "early_short_circuit": True,
                    "skipped_decompose_llm": True,
                    "skipped_retrieval": True,
                }
            )
        return _early_route_decompose_state(raw_q, route)
    enrich = enrich_query_with_memory(
        raw_q,
        conversation_summary=state.get("conversation_summary")
        if isinstance(state.get("conversation_summary"), str)
        else None,
        recent_turns=state.get("recent_turns")
        if isinstance(state.get("recent_turns"), list)
        else None,
    )
    q = str(enrich.get("enriched_query") or raw_q).strip()
    with observed_span("decompose", input_data={"query": q[:200], "enriched": bool(enrich.get("enriched"))}):
        dec_raw = decompose_query(q)
        decompose_llm_ms = float(dec_raw.pop("_decompose_llm_ms", 0.0) or 0.0)
        skipped_decompose_llm = bool(dec_raw.pop("_skipped_decompose_llm", False))
        dec = dec_raw
        profile = state.get("user_profile") if isinstance(state.get("user_profile"), dict) else None
        country = str((profile or {}).get("country") or "").strip() or None
        task_mode = resolve_task_mode(q, dec, profile_country=country)
        analytical = task_mode == "analytical"
        # Always expand known Africa zones for retrieval / geo purity (all plan tiers).
        # Plan-tier "no multi-country comparison" is answer-tone only.
        dec = expand_regions_in_decomposition(dec, q)
        if analytical or task_mode == "data_export_only" or dec.get("africa_panel"):
            # Re-run after analytical/panel flags may add africa cues in entities.
            dec = expand_regions_in_decomposition(dec, q)
        dec = apply_plan_decomposition_gates(dec, state.get("plan_type"), country)
        category = str(state.get("category") or (profile or {}).get("category") or "").strip() or None
        dec = apply_category_domain_hints(dec, category)
        # UI / API scopes must land in decomp so clarify, BQ plans, and retrieval agree.
        dec = _apply_ui_scope_overrides(dec, state)
        # Grounded entity/domain enrich → multi-measure retrieval contract tags.
        dec = enrich_decomposition_facets(q, dec)
        matched_bundles = match_intent_bundles(q, dec, breakdown=compile_breakdown(q))
        if matched_bundles:
            dec["matched_bundles"] = [mb.spec.id for mb in matched_bundles]
        contract = build_retrieval_contract(q, decomposition=dec, known_tables=set())
        if contract.corpus_domain_tags:
            dec["corpus_domain_tags"] = list(contract.corpus_domain_tags)
        bundle_pm = list(bundle_primary_measures(matched_bundles, q) or bundle_required_measures(matched_bundles))
        if bundle_pm:
            dec["primary_measures"] = bundle_pm
        elif contract.primary_measures:
            dec["primary_measures"] = list(contract.primary_measures)
        dec = sanitize_decomposition_for_bq(dec, primary_measures=dec.get("primary_measures"))
        if contract.companion_measures:
            dec["companion_measures"] = list(contract.companion_measures)
        measure_hints = resolve_measures(q, dec)
        measure_hit = measure_hints[0] if measure_hints else None
        # Re-resolve after gates/overrides may change geography or time.
        task_mode = resolve_task_mode(q, dec, profile_country=country)
        analytical = task_mode == "analytical"
        recency = resolve_recency_tier(q, measure_hit)
        answer_lang = detect_answer_language(q)
        turn = compile_turn_contract(
            q,
            dec,
            answer_lang=answer_lang,
            measure_hit=measure_hit,
            task_mode_hint=task_mode,
            matched_bundles=matched_bundles,
        )
        turn = resolve_capability(turn)
        if turn.is_fail_closed() and turn.job == "clarify":
            task_mode = "clarify"
        meta = is_meta_query(q)
        help_q = (not meta) and is_help_query(q, dec)
        product = (not meta) and (help_q or is_product_query(q, dec))
        greeting = (not meta) and (not product) and is_greeting_query(q)
        out_of_scope = (
            (not meta) and (not product) and (not greeting) and is_out_of_scope_query(q, dec)
        )
        answer_lang = detect_answer_language(q)
        lang_unknown = (
            (not meta)
            and (not product)
            and (not greeting)
            and (not out_of_scope)
            and answer_lang == "unknown"
        )
        if meta:
            route_candidate = "meta"
        elif help_q:
            route_candidate = "help"
        elif product:
            route_candidate = "product"
        elif greeting:
            route_candidate = "greeting"
        elif out_of_scope:
            route_candidate = "out_of_scope"
        elif lang_unknown:
            route_candidate = "language_unknown"
        elif task_mode == "clarify":
            route_candidate = "clarify"
        else:
            route_candidate = "full_rag"
        export_intent = detect_export_intent(q)
        update_current_span_metadata(
            {
                "route_candidate": route_candidate,
                "answer_lang": answer_lang,
                "export_intent": export_intent,
                "task_mode": task_mode,
                "analytical_mode": analytical,
                "measure_id": turn.measure_id or (measure_hit.measure.id if measure_hit else None),
                "turn_job": turn.job,
                "serve_status": turn.serve_status,
                "recency_tier": recency,
                "query_enriched": bool(enrich.get("enriched")),
                "decompose_llm_ms": decompose_llm_ms,
                "skipped_decompose_llm": skipped_decompose_llm,
            }
        )
        out: dict[str, Any] = {
            "decomposition": dec,
            "is_meta_query": meta,
            "is_product_query": product,
            "is_help_query": help_q,
            "is_greeting_query": greeting,
            "is_out_of_scope_query": out_of_scope,
            "is_language_unknown": lang_unknown,
            "answer_lang": answer_lang,
            "export_intent": export_intent,
            "task_mode": task_mode,
            "analytical_mode": analytical,
            "enriched_query": q if enrich.get("enriched") else None,
            "measure_id": turn.measure_id or (measure_hit.measure.id if measure_hit else None),
            "recency_tier": recency,
            "turn_contract": turn.to_dict(),
            "route_candidate": route_candidate,
            "early_short_circuit": False,
            "skipped_decompose_llm": skipped_decompose_llm,
            "skipped_retrieval": not turn.should_retrieve_vector(),
            "decompose_llm_ms": decompose_llm_ms,
        }
        if enrich.get("enriched"):
            out["query"] = q
        sp = compile_supervisor_plan(
            q,
            decomposition=dec,
            matched_bundles=matched_bundles,
            measure_hit=measure_hit,
            primary_measures=dec.get("primary_measures") if isinstance(dec.get("primary_measures"), list) else None,
        )
        out["supervisor_plan"] = sp.to_dict()
        out["routing_plan"] = build_routing_plan(
            turn=turn,
            measure_hit=measure_hit,
            supervisor_plan=sp,
            matched_bundles=matched_bundles,
            primary_measures=dec.get("primary_measures") if isinstance(dec.get("primary_measures"), list) else None,
        ).to_dict()
        if _slot_reasoner_active():
            rp = compile_reasoner_plan(
                q,
                decomposition=dec,
                turn_contract=turn,
                plan_type=str(state.get("plan_type") or ""),
                task_mode=task_mode,
                matched_bundles=matched_bundles,
            )
        else:
            rp = None
        if rp is not None:
            turn = apply_reasoner_to_turn(turn, rp)
            out["reasoner_plan"] = rp.to_dict()
            out["heavy_path"] = rp.heavy_path
            out["slot_path"] = bool(rp.bq_subquestions())
            dec["reasoner_job"] = rp.job
            dec["reasoner_shape"] = rp.shape
            if rp.geos:
                dec["geography"] = list(rp.geos)
            if rp.time_start:
                dec["time_start"] = rp.time_start
            if rp.time_end:
                dec["time_end"] = rp.time_end
            if rp.primary_measure:
                dec["primary_measures"] = list(
                    bundle_primary_measures(matched_bundles, q)
                    or bundle_required_measures(matched_bundles)
                    or [rp.primary_measure]
                )
            out["decomposition"] = dec
            out["turn_contract"] = turn.to_dict()
            out["measure_id"] = turn.measure_id or rp.primary_measure
        else:
            out["heavy_path"] = False
            out["slot_path"] = False
        return out


def _apply_ui_scope_overrides(
    decomposition: dict[str, Any],
    state: RAGGraphState,
) -> dict[str, Any]:
    """Merge UI/API geo and time overrides into decomposition for control-plane + BQ."""
    out = dict(decomposition or {})
    ts = str(state.get("time_start_override") or "").strip()[:10]
    te = str(state.get("time_end_override") or "").strip()[:10]
    if ts:
        out["time_start"] = ts
    if te:
        out["time_end"] = te
    geo_ov = str(state.get("geo_override") or "").strip()
    if geo_ov:
        geo = list(out.get("geography") or []) if isinstance(out.get("geography"), list) else []
        if geo_ov not in geo:
            geo = [geo_ov, *geo]
        out["geography"] = geo
    return out


def _tag_vector(
    item: dict[str, Any],
    kind: str,
    *,
    corpus_boost: float | None = None,
) -> dict[str, Any]:
    meta = dict(_chunk_metadata(item))
    if corpus_boost is not None:
        meta["corpus_boost"] = float(corpus_boost)
    return {
        **item,
        "source": kind,
        "_context_kind": kind,
        "metadata": meta,
    }


def _news_kwargs(state: RAGGraphState, dec: dict[str, Any]) -> dict[str, Any]:
    top_k = int(state.get("news_top_k") or _DEFAULT_NEWS_TOP_K)
    kwargs: dict[str, Any] = {
        "doc_kind": "news_article",
        "top_k": top_k,
        "vector_search_mode": "dense_named",
    }
    if _env_on("RAG_NEWS_DOMAIN_FILTER", default=True):
        domains = dec.get("domains") or []
        if domains:
            kwargs["domains_substring"] = str(domains[0])
    return kwargs


def _dense_corpus_kwargs(state: RAGGraphState, _dec: dict[str, Any], *, doc_kind: str) -> dict[str, Any]:
    """Kwargs for academic / policy / public_reports / formation dense named search."""
    return {
        "top_k": int(state.get("academic_top_k") or _DEFAULT_ACADEMIC_TOP_K),
        "doc_kind": doc_kind,
        "vector_search_mode": "dense_named",
    }


def _use_legacy_research_collection() -> bool:
    return os.environ.get("RAG_USE_LEGACY_RESEARCH_COLLECTION", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def _retrieve_legacy_research(state: RAGGraphState) -> list[dict[str, Any]]:
    """Optional single pre-split research collection (off by default)."""
    if not _use_legacy_research_collection():
        return []

    def _legacy_kwargs(st: RAGGraphState, dec: dict[str, Any]) -> dict[str, Any]:
        return {
            "top_k": int(st.get("academic_top_k") or _DEFAULT_ACADEMIC_TOP_K),
            "doc_kinds": [
                "academic_article",
                "policy_document",
                "public_report",
                "agricultural_practise",
            ],
            "vector_search_mode": "dense_named",
        }

    try:
        raw = _vector_retrieve_for_corpus(
            state,
            collection_env="QDRANT_COLLECTION_RESEARCH_PAPERS",
            default_collection="research_other_papers",
            build_kwargs=_legacy_kwargs,
            geo_fallback_env="RAG_RESEARCH_GEO_FALLBACK",
            time_fallback_env="RAG_RESEARCH_TIME_FALLBACK",
        )
    except Exception:
        logger.exception("Legacy research retrieval failed")
        raw = []
    return [_tag_vector(x, "research") for x in raw]


def _retrieve_dense_corpus(
    state: RAGGraphState,
    *,
    collection_env: str,
    default_collection: str,
    doc_kind: str,
    context_kind: str,
) -> list[dict[str, Any]]:
    """One Qdrant collection, soft-fail to [] on empty/missing/error."""

    def _build(st: RAGGraphState, dec: dict[str, Any]) -> dict[str, Any]:
        return _dense_corpus_kwargs(st, dec, doc_kind=doc_kind)

    try:
        raw = _vector_retrieve_for_corpus(
            state,
            collection_env=collection_env,
            default_collection=default_collection,
            build_kwargs=_build,
            geo_fallback_env="RAG_RESEARCH_GEO_FALLBACK",
            time_fallback_env="RAG_RESEARCH_TIME_FALLBACK",
        )
    except Exception:
        logger.exception(
            "Retrieval failed for collection %s (%s)",
            default_collection,
            collection_env,
        )
        raw = []
    return [_tag_vector(x, context_kind) for x in raw]


def _retrieve_news(state: RAGGraphState) -> list[dict[str, Any]]:
    dec = state.get("decomposition") or {}
    q = (state.get("query") or "").strip()
    countries = resolve_retrieval_geographies(
        geo_override=str(state.get("geo_override") or ""),
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )
    ts = (state.get("time_start_override") or dec.get("time_start") or "").strip()[:10]
    te = (state.get("time_end_override") or dec.get("time_end") or "").strip()[:10]
    top_k = int(state.get("news_top_k") or _DEFAULT_NEWS_TOP_K)
    news_coll = os.environ.get("QDRANT_COLLECTION_NEWS", "news_data").strip() or "news_data"
    vr = VectorRetriever(collection_name=news_coll)
    strict_compare = _strict_compare_filters(dec)
    allow_geo_fb = not strict_compare

    def _run_for_countries(target_countries: list[str], *, allow_geo_fb: bool) -> list[dict[str, Any]]:
        if len(target_countries) >= 2:
            per_k = max(3, (top_k + len(target_countries) - 1) // len(target_countries))
            batches: list[list[dict[str, Any]]] = []
            for country in target_countries:
                kw = _news_kwargs(state, dec)
                kw["top_k"] = per_k
                kw["geo_country"] = country
                kw["time_filter_at_qdrant"] = _news_time_filter_at_qdrant()
                if not kw["time_filter_at_qdrant"]:
                    kw["overfetch_multiplier"] = 12
                if ts:
                    kw["published_at_from"] = ts
                if te:
                    kw["published_at_to"] = te
                batch = _retrieve_vector_cascade(
                    vr,
                    q,
                    base_kwargs=kw,
                    countries=[country],
                    has_time=bool(ts or te),
                    geo_fallback_env="RAG_NEWS_GEO_FALLBACK",
                    time_fallback_env="RAG_NEWS_TIME_FALLBACK",
                    allow_geo_fallback=allow_geo_fb,
                )
                batches.append(batch)
            merged = _merge_dedupe_vector_hits(batches, top_k)
            return _post_filter_geography(merged, target_countries)

        kw = _news_kwargs(state, dec)
        _apply_geo_to_kwargs(kw, target_countries)
        kw["time_filter_at_qdrant"] = _news_time_filter_at_qdrant()
        if not kw["time_filter_at_qdrant"]:
            kw["overfetch_multiplier"] = 12
        if ts:
            kw["published_at_from"] = ts
        if te:
            kw["published_at_to"] = te
        raw = _retrieve_vector_cascade(
            vr,
            q,
            base_kwargs=kw,
            countries=target_countries,
            has_time=bool(ts or te),
            geo_fallback_env="RAG_NEWS_GEO_FALLBACK",
            time_fallback_env="RAG_NEWS_TIME_FALLBACK",
            allow_geo_fallback=allow_geo_fb,
        )
        return _post_filter_geography(raw, target_countries)

    raw = _run_for_countries(countries, allow_geo_fb=allow_geo_fb)

    # Compare: keep geo strict at Qdrant but allow semantic recall + post-filter by country name in text.
    if not raw and countries and strict_compare and _env_on("RAG_NEWS_COMPARE_SEMANTIC_FALLBACK", default=True):
        kw = _news_kwargs(state, dec)
        kw["time_filter_at_qdrant"] = False
        kw["overfetch_multiplier"] = 20
        if ts:
            kw["published_at_from"] = ts
        if te:
            kw["published_at_to"] = te
        semantic = _retrieve_vector_cascade(
            vr,
            q,
            base_kwargs=kw,
            countries=[],
            has_time=bool(ts or te),
            geo_fallback_env="RAG_NEWS_GEO_FALLBACK",
            time_fallback_env="RAG_NEWS_TIME_FALLBACK",
            allow_geo_fallback=False,
        )
        raw = _post_filter_geography(semantic, countries)[:top_k]
        if raw:
            logger.debug("News compare semantic fallback: %d chunks after geo post-filter", len(raw))

    return [_tag_vector(x, "news") for x in raw]


def _retrieve_academic_papers(state: RAGGraphState) -> list[dict[str, Any]]:
    return _retrieve_dense_corpus(
        state,
        collection_env="QDRANT_COLLECTION_ACADEMIC_PAPERS",
        default_collection="academic_papers",
        doc_kind="academic_article",
        context_kind="academic",
    )


def _retrieve_policies(state: RAGGraphState) -> list[dict[str, Any]]:
    return _retrieve_dense_corpus(
        state,
        collection_env="QDRANT_COLLECTION_POLICIES",
        default_collection="policies",
        doc_kind="policy_document",
        context_kind="policy",
    )


def _retrieve_public_reports(state: RAGGraphState) -> list[dict[str, Any]]:
    return _retrieve_dense_corpus(
        state,
        collection_env="QDRANT_COLLECTION_PUBLIC_REPORTS",
        default_collection="public_reports",
        doc_kind="public_report",
        context_kind="public_report",
    )


def _retrieve_formation(state: RAGGraphState) -> list[dict[str, Any]]:
    return _retrieve_dense_corpus(
        state,
        collection_env="QDRANT_COLLECTION_FORMATION",
        default_collection="formation",
        doc_kind="agricultural_practise",
        context_kind="formation",
    )


def _retrieve_ota(state: RAGGraphState) -> list[dict[str, Any]]:
    """Retrieve OTA insights (triple vectors) from the OTA_insights collection.

    The collection may be empty on initial deployment (analysts populate it later).
    Results are merged into the main context with clear OTA citations.
    """
    coll = os.environ.get("QDRANT_COLLECTION_OTA_INSIGHTS", "OTA_insights").strip() or "OTA_insights"
    vr = VectorRetriever(collection_name=coll)

    q = (state.get("query") or "").strip()
    top_k = int(state.get("ota_top_k") or _DEFAULT_OTA_TOP_K)

    dec = state.get("decomposition") or {}
    countries = resolve_retrieval_geographies(
        geo_override=str(state.get("geo_override") or ""),
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )

    kw: dict[str, Any] = {
        "top_k": top_k,
        "vector_search_mode": "ota_triple",
    }
    _apply_geo_to_kwargs(kw, countries)

    try:
        raw = vr.retrieve(q, doc_kind="ota_insight", **kw)
    except Exception:
        logger.exception("OTA retrieval failed for collection %s", coll)
        raw = []

    if countries:
        raw = _post_filter_geography(raw or [], countries)
    return [_tag_vector(x, "ota_insight") for x in (raw or [])]


def _append_corpus_merge(
    merged: list[dict[str, Any]],
    items: list[dict[str, Any]] | None,
    *,
    source: str,
    context_kind: str,
    label_fn: Any,
) -> None:
    for item in items or []:
        text = item.get("content") or ""
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        label = label_fn(meta if isinstance(meta, dict) else {})
        merged.append(
            {
                "content": f"{label} {text}",
                "source": source,
                "_context_kind": context_kind,
                "metadata": meta,
                "score": item.get("score"),
            }
        )


def _academic_merge_label(meta: dict[str, Any]) -> str:
    cite = format_academic_citation(meta)
    return f"[Academic | {cite}]" if cite else "[Academic]"


def _title_merge_label(prefix: str, meta: dict[str, Any]) -> str:
    title = str(meta.get("section_title") or meta.get("label") or meta.get("source_file") or "").strip()
    return f"[{prefix} | {title}]" if title else f"[{prefix}]"


def node_parallel_retrieve(state: RAGGraphState) -> dict[str, Any]:
    """Run selected corpus retrieves (+ optional legacy research) in parallel."""
    decomposition_raw = state.get("decomposition")
    decomposition: dict[str, Any] = (
        cast(dict[str, Any], decomposition_raw)
        if isinstance(decomposition_raw, dict)
        else {}
    )
    task_mode = str(state.get("task_mode") or "chat")
    domain_tags = decomposition.get("corpus_domain_tags")
    tc_raw = state.get("turn_contract")
    contract = TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None)
    sp_raw = state.get("supervisor_plan")
    sp = SupervisorPlan.from_dict(sp_raw if isinstance(sp_raw, dict) else None)
    contract_job = contract.job if contract.measure_id or contract.job != "fact" else contract.job
    if not contract.should_retrieve_vector() and not (sp and sp.must_search_qdrant):
        return {
            "vector_news_results": [],
            "vector_academic_papers_results": [],
            "vector_policies_results": [],
            "vector_public_reports_results": [],
            "vector_formation_results": [],
            "vector_ota_results": [],
            "vector_results": [],
            "corpus_selection": {"active": [], "boosts": {}, "rationale": "turn_contract_skip_vector"},
            "corpus_count": 0,
            "vector_ms": 0.0,
            "cascade_level": 0,
        }
    selection = select_corpora(
        decomposition,
        plan_type=str(state.get("plan_type") or "") or None,
        query=str(state.get("query") or ""),
        task_mode=task_mode,
        corpus_domain_tags=list(domain_tags) if isinstance(domain_tags, list) else None,
        contract_job=contract_job,
        **(
            {
                "vector_allow": contract.vector_allow,
                "vector_block": contract.vector_block,
                "vector_policy": contract.vector_policy,
            }
            if contract.measure_id
            else {}
        ),
    )
    heavy = bool(state.get("heavy_path"))
    rp = _reasoner_from_state(state)
    planner_meta: dict[str, Any] = {}
    if (heavy or _slot_path_active(state)) and rp:
        planner = plan_vector_corpora_for_reasoner(
            query=str(state.get("query") or ""),
            reasoner=rp,
            turn_contract=contract,
            plan_type=str(state.get("plan_type") or "") or None,
            decomposition=decomposition,
        )
        selection = CorpusSelection(
            active=list(planner.get("active_corpora") or selection.active),
            boosts=selection.boosts,
            rationale=f"{selection.rationale}|reasoner_planner",
        )
        planner_meta = {
            "hard_time_filter": bool(planner.get("hard_time_filter")),
            "news_allowed": bool(planner.get("news_allowed", True)),
            "reasoner_planner": True,
        }
    if sp and sp.classes and sp.must_search_qdrant:
        class_corpora = corpora_for_supervisor_plan(sp)
        if class_corpora:
            filtered = [c for c in class_corpora if c in CORPUS_CATALOG]
            if filtered:
                selection = CorpusSelection(
                    active=filtered,
                    boosts=selection.boosts,
                    rationale=f"{selection.rationale}|class_supervisor",
                )
    active = set(selection.active)
    boosts = selection.boosts

    # Soft top_k nudges for mode specialization.
    news_k = state.get("news_top_k")
    ota_k = state.get("ota_top_k")
    academic_k = state.get("academic_top_k")
    if task_mode == "briefing":
        if news_k is None:
            news_k = _BRIEFING_NEWS_TOP_K
        if ota_k is None:
            ota_k = _BRIEFING_OTA_TOP_K
    elif task_mode in ("fact_lookup", "data_export_only"):
        if news_k is None:
            news_k = _FACT_LOOKUP_NEWS_TOP_K
        if academic_k is None:
            academic_k = _FACT_LOOKUP_ACADEMIC_TOP_K

    intent = str(decomposition.get("intent") or "").strip().lower()
    rationale = selection.rationale or ""
    if ota_k is None and (
        intent == "decision_support"
        or "investment_decision_cues" in rationale
        or "intent_decision_support" in rationale
        or "plan_ota_boost" in rationale
    ):
        ota_k = _DECISION_SUPPORT_OTA_TOP_K

    state_for_retrieve = cast(RAGGraphState, dict(state))
    if news_k is not None:
        state_for_retrieve["news_top_k"] = int(news_k)
    if ota_k is not None:
        state_for_retrieve["ota_top_k"] = int(ota_k)
    if academic_k is not None:
        state_for_retrieve["academic_top_k"] = int(academic_k)

    buckets: dict[str, list[dict[str, Any]]] = {
        "news": [],
        "academic_papers": [],
        "policies": [],
        "public_reports": [],
        "formation": [],
        "ota": [],
        "legacy_research": [],
    }
    corpus_errors: list[str] = []

    retrievers = {
        "news": _retrieve_news,
        "academic_papers": _retrieve_academic_papers,
        "policies": _retrieve_policies,
        "public_reports": _retrieve_public_reports,
        "formation": _retrieve_formation,
        "ota": _retrieve_ota,
    }

    jobs: dict[Any, str] = {}
    workers = max(1, min(6, len(active) + (1 if _use_legacy_research_collection() else 0)))
    try:
        corpus_timeout = float(os.environ.get("RAG_CORPUS_RETRIEVE_TIMEOUT_S", "4") or 4)
    except ValueError:
        corpus_timeout = 8.0
    vector_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, fn in retrievers.items():
            if key in active:
                jobs[ex.submit(run_with_tracing_context(fn, state_for_retrieve))] = key
        if _use_legacy_research_collection():
            jobs[ex.submit(run_with_tracing_context(_retrieve_legacy_research, state_for_retrieve))] = (
                "legacy_research"
            )

        for fut in as_completed(jobs):
            kind = jobs[fut]
            try:
                res = fut.result(timeout=corpus_timeout)
            except FuturesTimeoutError:
                logger.warning("Parallel retrieval timed out for %s after %.1fs", kind, corpus_timeout)
                corpus_errors.append(f"{kind}:timeout")
                res = []
            except Exception as exc:
                logger.exception("Parallel retrieval failed for %s; returning empty list", kind)
                corpus_errors.append(f"{kind}:{type(exc).__name__}")
                res = []
            buckets[kind] = res if isinstance(res, list) else []

    def _with_boost(items: list[dict[str, Any]], corpus_key: str) -> list[dict[str, Any]]:
        boost = float(boosts.get(corpus_key, 0.0))
        out: list[dict[str, Any]] = []
        for item in items:
            meta = dict(_chunk_metadata(item))
            meta["corpus_boost"] = boost
            out.append({**item, "metadata": meta})
        return out

    news_out = _with_boost(buckets["news"], "news")
    academic_papers_out = _with_boost(buckets["academic_papers"], "academic_papers")
    policies_out = _with_boost(buckets["policies"], "policies")
    public_reports_out = _with_boost(buckets["public_reports"], "public_reports")
    formation_out = _with_boost(buckets["formation"], "formation")
    ota_out = _with_boost(buckets["ota"], "ota")
    legacy_out = buckets["legacy_research"]

    meta_update: dict[str, Any] = {
        "corpus_active": ",".join(selection.active),
        "corpus_rationale": selection.rationale,
        "corpus_count": len(selection.active),
        "vector_ms": trace_elapsed_ms(vector_t0),
    }
    if corpus_errors:
        meta_update["corpus_error"] = ",".join(corpus_errors)
    update_current_span_metadata(meta_update)

    combined = (
        list(news_out)
        + list(academic_papers_out)
        + list(policies_out)
        + list(public_reports_out)
        + list(formation_out)
        + list(ota_out)
        + list(legacy_out)
    )
    return {
        "vector_news_results": news_out,
        "vector_academic_papers_results": academic_papers_out,
        "vector_policies_results": policies_out,
        "vector_public_reports_results": public_reports_out,
        "vector_formation_results": formation_out,
        "vector_ota_results": ota_out,
        # Deprecated combined bucket for callers that still read it
        "vector_academic_results": list(academic_papers_out)
        + list(policies_out)
        + list(public_reports_out)
        + list(formation_out)
        + list(legacy_out),
        "vector_results": combined,
        "corpus_selection": {**selection.to_dict(), **planner_meta},
        "vector_ms": meta_update["vector_ms"],
        "corpus_count": len(selection.active),
        "cascade_level": 1 if task_mode in ("fact_lookup", "data_export_only") else None,
    }


def _nl2sql_fallback_enabled() -> bool:
    return os.environ.get("RAG_BQ_NL2SQL_FALLBACK", "").strip().lower() in ("1", "true", "yes", "on")


def _nl2sql_escape_allowed(
    *,
    task_mode: str,
    analytical: bool,
    plan: dict[str, Any],
) -> bool:
    if analytical or task_mode == "analytical":
        return True
    if plan.get("nl2sql_fallback"):
        return True
    return _nl2sql_fallback_enabled() and bool(plan.get("compile_error"))


def _compile_error_bq_plan(
    sp: SupervisorPlan | None,
    *,
    rationale: str,
    nl2sql_fallback: bool = False,
) -> dict[str, Any]:
    tables = tables_for_supervisor_plan(sp) if sp else []
    return {
        "selected_tables": tables,
        "query_intents": [],
        "skip_bq": False,
        "rationale": rationale,
        "bq_sql_queries": [],
        "sql_source": "compile_error",
        "compile_error": True,
        "nl2sql_fallback": nl2sql_fallback,
        "supervisor_plan": sp.to_dict() if sp else {},
        "table_hints": [],
        "hints_truncated": False,
    }


def node_bq_reason(state: RAGGraphState) -> dict[str, Any]:
    """YAML-index SQL reasoner or class engines: select mart_dev tables and query intents."""
    q = (state.get("query") or "").strip()
    analytical = bool(state.get("analytical_mode"))
    task_mode = str(state.get("task_mode") or ("analytical" if analytical else "chat"))
    dec_raw = state.get("decomposition")
    dec: dict[str, Any] = (
        cast(dict[str, Any], dec_raw) if isinstance(dec_raw, dict) else {}
    )
    sp = SupervisorPlan.from_dict(
        state.get("supervisor_plan") if isinstance(state.get("supervisor_plan"), dict) else None
    )
    if sp is None and not _slot_reasoner_active():
        bundles = match_intent_bundles(q, dec, breakdown=compile_breakdown(q))
        measure_hints = resolve_measures(q, dec)
        mh = measure_hints[0] if measure_hints else None
        sp = compile_supervisor_plan(
            q,
            decomposition=dec,
            matched_bundles=bundles,
            measure_hit=mh,
            primary_measures=dec.get("primary_measures") if isinstance(dec.get("primary_measures"), list) else None,
        )

    use_compiler_path = sql_compiler_enabled() and task_mode != "analytical"
    plan: dict[str, Any]
    if use_compiler_path:
        rp_raw = state.get("reasoner_plan")
        reasoner = ReasonerPlan.from_dict(rp_raw if isinstance(rp_raw, dict) else None)
        if reasoner and reasoner.bq_subquestions() and _slot_reasoner_active():
            plan = reasoner_plan_to_bq_plan(reasoner)
        elif sp and sp.classes:
            engine_results = run_class_engines(q, supervisor_plan=sp, facets=dec)
            plan = engine_results_to_bq_plan(engine_results, rationale=sp.rationale)
            plan["supervisor_plan"] = sp.to_dict()
            if not plan.get("bq_sql_queries") and not plan.get("bind_contracts"):
                allow_nl2 = _nl2sql_escape_allowed(
                    task_mode=task_mode,
                    analytical=analytical,
                    plan={"compile_error": True},
                )
                plan = {
                    **plan,
                    "sql_source": "compile_error",
                    "compile_error": True,
                    "nl2sql_fallback": allow_nl2,
                }
        else:
            allow_nl2 = _nl2sql_escape_allowed(
                task_mode=task_mode,
                analytical=analytical,
                plan={"compile_error": True},
            )
            plan = _compile_error_bq_plan(
                sp,
                rationale="no_class_for_compiler_path",
                nl2sql_fallback=allow_nl2,
            )
    elif sp and sp.classes and sql_compiler_enabled():
        engine_results = run_class_engines(q, supervisor_plan=sp, facets=dec)
        plan = engine_results_to_bq_plan(engine_results, rationale=sp.rationale)
        plan["supervisor_plan"] = sp.to_dict()
    else:
        rp_raw = state.get("reasoner_plan")
        reasoner = ReasonerPlan.from_dict(rp_raw if isinstance(rp_raw, dict) else None)
        if reasoner and reasoner.bq_subquestions() and _slot_reasoner_active():
            plan = reasoner_plan_to_bq_plan(reasoner)
        else:
            plan = reason_bq_sql_plan(
                q,
                decomposition=dec,
                plan_type=str(state.get("plan_type") or "") or None,
                category=str(state.get("category") or "") or None,
                analytical_mode=analytical,
                task_mode=task_mode,
            )
    tc_raw = state.get("turn_contract")
    contract = TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None)
    if (
        contract.skip_bq
        and not plan.get("heavy_path")
        and not plan.get("slot_path")
        and not plan.get("bq_sql_queries")
    ):
        plan = {**plan, "skip_bq": True}
    template_key = str((contract.sql_plan or {}).get("template") or "").strip()
    if template_key:
        plan = {**plan, "template_key": template_key}
    hints = list(plan.get("table_hints") or [])
    cands = [
        {
            "content": hint,
            "table_name": tid,
            "metadata": {"table_name": tid, "source": "mart_yaml"},
        }
        for tid, hint in zip(list(plan.get("selected_tables") or []), hints)
    ]
    # If hint count differs, still expose selected table ids as candidates.
    if not cands:
        for tid in list(plan.get("selected_tables") or []):
            cands.append(
                {
                    "content": "",
                    "table_name": tid,
                    "metadata": {"table_name": tid, "source": "mart_yaml"},
                }
            )
    return {
        "bq_sql_plan": plan,
        "bq_table_candidates_inspector": cands,
        "value_hits": plan.get("value_hits"),
    }


def aggregate_bq_sql_debug(results: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Collect distinct SQL strings and per-attempt debug rows from BQ retrieve items."""
    sql_seen: set[str] = set()
    bq_sql_queries: list[str] = []
    bq_sql_debug: list[dict[str, Any]] = []
    debug_seen: set[tuple[Any, ...]] = set()
    for row in results:
        raw_meta = row.get("metadata")
        if isinstance(raw_meta, dict):
            meta = dict(raw_meta)
        else:
            meta = {}
        sql = str(meta.get("sql") or "").strip()
        if sql and sql not in sql_seen:
            sql_seen.add(sql)
            bq_sql_queries.append(sql)
        status = str(meta.get("status") or "").strip()
        if not status:
            if meta.get("validation_failed"):
                status = "validation_failed"
            elif meta.get("execution_error"):
                status = "execution_error"
            elif sql:
                status = "ok"
            else:
                status = "unknown"
        debug_key = (
            sql,
            status,
            meta.get("sql_index"),
            meta.get("prep_error"),
            meta.get("execution_error"),
        )
        if debug_key in debug_seen:
            continue
        debug_seen.add(debug_key)
        prep = meta.get("prep_error")
        exec_err = meta.get("execution_error")
        row_debug: dict[str, Any] = {
            "sql": sql,
            "status": status,
            "prep_error": str(prep)[:500] if prep else None,
            "execution_error": str(exec_err)[:500] if exec_err else None,
        }
        if meta.get("sql_source"):
            row_debug["sql_source"] = meta.get("sql_source")
        if meta.get("template"):
            row_debug["template"] = meta.get("template")
        if meta.get("pattern"):
            row_debug["pattern"] = meta.get("pattern")
        if meta.get("nl2sql_model"):
            row_debug["nl2sql_model"] = meta.get("nl2sql_model")
        if meta.get("nl2sql_raw"):
            row_debug["nl2sql_raw"] = str(meta.get("nl2sql_raw"))[:500]
        for key in ("job_id", "bq_ms", "bytes_processed", "row_count", "class_code", "table_id", "value_hits"):
            if meta.get(key) is not None:
                row_debug[key] = meta.get(key)
        bq_sql_debug.append(row_debug)
    return bq_sql_queries, bq_sql_debug


def _pre_debug_hits_by_sql(pre_debug: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in pre_debug:
        if not isinstance(row, dict):
            continue
        sql = str(row.get("sql") or "").strip()
        hits = row.get("value_hits")
        if sql and isinstance(hits, dict):
            out[sql] = hits
    return out


def reconcile_engine_bq_debug(
    *,
    execute_debug: list[dict[str, Any]],
    pre_queries: list[str],
    pre_debug: list[dict[str, Any]],
    node_timed_out: bool,
    bq_timeout: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build final inspector debug from execute outcomes; never leave stale planned rows."""
    hits_by_sql = _pre_debug_hits_by_sql(pre_debug)
    by_sql: dict[str, dict[str, Any]] = {}
    for row in execute_debug:
        if not isinstance(row, dict):
            continue
        sql = str(row.get("sql") or "").strip()
        if sql:
            by_sql[sql] = dict(row)
    out_debug: list[dict[str, Any]] = []
    out_queries: list[str] = []
    for raw in pre_queries:
        sql = str(raw).strip()
        if not sql:
            continue
        if sql not in out_queries:
            out_queries.append(sql)
        if sql in by_sql:
            row = dict(by_sql[sql])
        elif node_timed_out:
            row = {
                "sql": sql,
                "status": "timeout",
                "prep_error": f"node_bq_retrieve wall deadline ({bq_timeout:.0f}s)",
                "sql_source": "engine",
            }
        else:
            row = {
                "sql": sql,
                "status": "unknown",
                "sql_source": "engine",
            }
        if sql in hits_by_sql and not row.get("value_hits"):
            row["value_hits"] = hits_by_sql[sql]
        out_debug.append(row)
    seen = set(out_queries)
    for row in execute_debug:
        if not isinstance(row, dict):
            continue
        sql = str(row.get("sql") or "").strip()
        if sql and sql not in seen:
            out_debug.append(dict(row))
            out_queries.append(sql)
            seen.add(sql)
    return out_queries, out_debug


def _typed_bq_hard_return(
    state: RAGGraphState,
    exec_flags: dict[str, bool],
    *,
    bq_debug: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Hard-return typed warehouse gap before LLM when SQL was planned."""
    priority = (
        "structured_bq_timed_out",
        "structured_bq_validation_failed",
        "structured_bq_compile_error",
        "structured_bq_empty",
        "structured_bq_never_executed",
    )
    flag = next((k for k in priority if exec_flags.get(k)), None)
    if not flag:
        return None
    dec = state.get("decomposition")
    dec_dict = dec if isinstance(dec, dict) else None
    tc_raw = state.get("turn_contract")
    tc_dict = tc_raw if isinstance(tc_raw, dict) else None
    answer = typed_bq_gap_answer(
        flag=flag,
        decomposition=dec_dict,
        turn_contract=tc_dict,
        prep_error=first_prep_error(bq_debug),
    )
    return {
        "answer": answer,
        "citations": [],
        "answer_lang": str(state.get("answer_lang") or detect_answer_language(str(state.get("query") or ""))),
        **acf_result_to_state(
            no_evidence_acf(
                explanation=(
                    "Warehouse query did not yield scorable structured evidence "
                    f"({flag.replace('structured_bq_', '')})."
                )
            )
        ),
    }


def bq_failure_debug_row(
    retriever: BQRetriever,
    *,
    status: str,
    prep_error: str,
) -> dict[str, Any]:
    """Synthetic inspector row when BQ retrieve times out or raises before returning items."""
    row: dict[str, Any] = {
        "sql": "",
        "status": status,
        "prep_error": prep_error[:500],
    }
    raw_r = getattr(retriever, "_last_nl2sql_raws", None)
    if isinstance(raw_r, list) and raw_r:
        row["nl2sql_raw"] = "; ".join(str(x) for x in raw_r)[:500]
    src = getattr(retriever, "last_sql_source", None)
    if src:
        row["sql_source"] = str(src)
    return row


def node_bq_retrieve(state: RAGGraphState) -> dict[str, Any]:
    q = (state.get("query") or "").strip()
    raw_dec = state.get("decomposition")
    dec: dict[str, Any] = raw_dec if isinstance(raw_dec, dict) else {}
    raw_plan = state.get("bq_sql_plan")
    plan: dict[str, Any] = raw_plan if isinstance(raw_plan, dict) else {}
    if plan.get("skip_bq"):
        return {
            "bq_results": [],
            "bq_sql_queries": [],
            "bq_sql_debug": [],
            "sql_source": "skipped",
            "bq_execute_ms": 0.0,
            "bq_nl2sql_ms": 0.0,
        }

    cached = state.get("structured_ranking_cache")
    if isinstance(cached, dict) and is_ranking_follow_up(q, dec, cached):
        results = bq_results_from_cache(cached)
        bq_sql_queries, bq_sql_debug = aggregate_bq_sql_debug(results)
        update_current_span_metadata({"bq_cache_hit": True})
        return {
            "bq_results": results,
            "bq_sql_queries": bq_sql_queries,
            "bq_sql_debug": bq_sql_debug,
            "bq_cache_hit": True,
            "structured_ranking_cache": cached,
            "sql_source": "cache",
            "bq_execute_ms": 0.0,
            "bq_nl2sql_ms": 0.0,
        }

    hints = [str(h).strip() for h in (plan.get("table_hints") or []) if str(h).strip()]
    if not hints:
        cands = state.get("bq_table_candidates") or []
        if not cands:
            cands = state.get("bq_table_candidates_inspector") or []
        hints = [str(c.get("content") or "") for c in cands if c.get("content")]

    # Enrich NL2SQL leftover with reasoner intents when present; pattern SQL
    # compiles from query_intents + the original question (not concatenated staples).
    raw_intents = plan.get("query_intents")
    intents: list[Any] = raw_intents if isinstance(raw_intents, list) else []

    top_k = int(state.get("bq_top_k") or _DEFAULT_BQ_TOP_K)
    countries = resolve_retrieval_geographies(
        geo_override=str(state.get("geo_override") or ""),
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )
    ts = (state.get("time_start_override") or dec.get("time_start") or "").strip()[:10]
    te = (state.get("time_end_override") or dec.get("time_end") or "").strip()[:10]
    entities = dec.get("entities") if isinstance(dec.get("entities"), list) else None
    domains = dec.get("domains") if isinstance(dec.get("domains"), list) else None
    retriever = BQRetriever()
    bq_geo: dict[str, Any] = {}
    if len(countries) >= 2:
        bq_geo["geo_countries"] = countries
    elif len(countries) == 1:
        bq_geo["geo_country"] = countries[0]

    prev_max_sql: str | None = None
    env_bumped = False
    if bool(state.get("analytical_mode")) or plan.get("analytical_mode"):
        from ml.rag.chatbot.analytical_bq_plan import analytical_sql_query_floor

        floor = int(plan.get("max_sql_queries") or analytical_sql_query_floor())
        prev_max_sql = os.environ.get("RAG_BQ_MAX_SQL_QUERIES")
        os.environ["RAG_BQ_MAX_SQL_QUERIES"] = str(floor)
        env_bumped = True
    task_mode = str(state.get("task_mode") or "chat").strip().lower()
    pre_debug = list(plan.get("bq_sql_debug") or [])
    pre_queries = list(plan.get("bq_sql_queries") or [])
    engine_execute_only = plan.get("sql_source") == "engine" and bool(pre_queries)
    if engine_execute_only:
        intents = []
    default_bq_timeout = 25.0
    try:
        bq_timeout = float(os.environ.get("RAG_BQ_RETRIEVE_TIMEOUT_S", str(default_bq_timeout)) or default_bq_timeout)
    except ValueError:
        bq_timeout = default_bq_timeout
    tc_raw = state.get("turn_contract")
    contract = TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None)
    contract_kwargs: dict[str, Any] = {}
    if contract.measure_id or contract.serve_status != "served":
        contract_kwargs = {
            "serve_status": contract.serve_status,
            "contract_sql_only": contract.contract_sql_only,
            "template_key": str((contract.sql_plan or {}).get("template") or ""),
            "heavy_path": bool(plan.get("heavy_path")),
        }
    bq_failure_row: dict[str, Any] | None = None
    node_timed_out = False
    retrieve_deadline = time.perf_counter() + bq_timeout
    retrieve_kwargs: dict[str, Any] = {
        "crop_required": bool(plan.get("crop_required", True)),
        "geography_required": bool(plan.get("geography_required", True)),
        "decomposition": dec,
        "primary_measures": (
            dec.get("primary_measures")
            if isinstance(dec.get("primary_measures"), list)
            else None
        ),
    }
    if engine_execute_only:
        retrieve_kwargs["engine_execute_only"] = True
    if plan.get("nl2sql_fallback"):
        retrieve_kwargs["nl2sql_fallback"] = True
    bind_raw = plan.get("bind_contracts")
    if isinstance(bind_raw, dict) and bind_raw:
        retrieve_kwargs["bind_contracts"] = bind_raw
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                retriever.retrieve,
                q,
                top_k=top_k,
                table_hints=hints,
                selected_tables=list(plan.get("selected_tables") or []),
                query_intents=intents,
                time_start=ts or None,
                time_end=te or None,
                entities=entities,
                domains=domains,
                task_mode=task_mode,
                sql=pre_queries if pre_queries else None,
                deadline=retrieve_deadline,
                **bq_geo,
                **contract_kwargs,
                **retrieve_kwargs,
            )
            results = fut.result(timeout=bq_timeout)
    except FuturesTimeoutError:
        update_current_span_metadata({"bq_timeout": True, "status": "timeout"})
        node_timed_out = True
        results = []
        if not engine_execute_only:
            bq_failure_row = bq_failure_debug_row(
                retriever,
                status="bq_timeout",
                prep_error=f"BQ retrieve exceeded {bq_timeout:.0f}s timeout",
            )
    except Exception as exc:
        logger.exception("BQ retrieve failed")
        results = []
        bq_failure_row = bq_failure_debug_row(
            retriever,
            status="bq_error",
            prep_error=str(exc)[:500],
        )
    finally:
        if env_bumped:
            if prev_max_sql is None:
                os.environ.pop("RAG_BQ_MAX_SQL_QUERIES", None)
            else:
                os.environ["RAG_BQ_MAX_SQL_QUERIES"] = prev_max_sql
    results = enrich_bq_results(
        results,
        query=q,
        plan=plan,
        decomposition=dec,
    )
    results, ctx_truncated = trim_bq_result_contents(results)
    if ctx_truncated:
        update_current_span_metadata({"bq_context_truncated": True})
    bq_sql_queries, bq_sql_debug = aggregate_bq_sql_debug(results)
    if engine_execute_only and pre_queries:
        bq_sql_queries, bq_sql_debug = reconcile_engine_bq_debug(
            execute_debug=bq_sql_debug,
            pre_queries=pre_queries,
            pre_debug=pre_debug,
            node_timed_out=node_timed_out,
            bq_timeout=bq_timeout,
        )
    elif pre_debug:
        hits_by_sql = _pre_debug_hits_by_sql(pre_debug)
        for row in bq_sql_debug:
            sql = str(row.get("sql") or "").strip()
            if sql in hits_by_sql and not row.get("value_hits"):
                row["value_hits"] = hits_by_sql[sql]
    if bq_failure_row and not bq_sql_debug:
        bq_sql_debug = [bq_failure_row]
    elif bq_failure_row and not engine_execute_only:
        bq_sql_debug = list(bq_sql_debug) + [bq_failure_row]
    cache_entry = cache_entry_from_bq_results(results, query=q, decomposition=dec)
    sql_source = getattr(retriever, "last_sql_source", None)
    if not sql_source:
        for row in bq_sql_debug:
            src = row.get("sql_source")
            if src:
                sql_source = str(src)
                break
    out: dict[str, Any] = {
        "bq_results": results,
        "bq_sql_queries": bq_sql_queries,
        "bq_sql_debug": bq_sql_debug,
        "bq_cache_hit": False,
        "sql_source": sql_source,
        "bq_execute_ms": getattr(retriever, "last_bq_execute_ms", None),
        "bq_nl2sql_ms": getattr(retriever, "last_bq_nl2sql_ms", None),
    }
    if cache_entry:
        out["structured_ranking_cache"] = cache_entry
    return out


def node_merge(state: RAGGraphState) -> dict[str, Any]:
    with observed_span("merge"):
        bq_merged: list[dict[str, Any]] = []
        other_merged: list[dict[str, Any]] = []

        def _append_bq(r: dict[str, Any]) -> None:
            if not is_usable_context_item(r):
                return
            text = str(r.get("content") or "").strip()
            bq_merged.append(
                {
                    **r,
                    "content": f"[Structured data] {text}" if text else "[Structured data]",
                    "source": r.get("source", "bigquery"),
                    "_context_kind": "bigquery",
                }
            )

        for r in state.get("bq_results") or []:
            _append_bq(r)

        for item in state.get("vector_news_results") or []:
            text = item.get("content") or ""
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            other_merged.append(
                {
                    "content": f"[News] {text}",
                    "source": "news",
                    "_context_kind": "news",
                    "metadata": meta,
                    "score": item.get("score"),
                }
            )
        _append_corpus_merge(
            other_merged,
            state.get("vector_academic_papers_results"),
            source="academic",
            context_kind="academic",
            label_fn=_academic_merge_label,
        )
        _append_corpus_merge(
            other_merged,
            state.get("vector_policies_results"),
            source="policy",
            context_kind="policy",
            label_fn=lambda m: _title_merge_label("Policy", m),
        )
        _append_corpus_merge(
            other_merged,
            state.get("vector_public_reports_results"),
            source="public_report",
            context_kind="public_report",
            label_fn=lambda m: _title_merge_label("Public report", m),
        )
        _append_corpus_merge(
            other_merged,
            state.get("vector_formation_results"),
            source="formation",
            context_kind="formation",
            label_fn=lambda m: _title_merge_label("Formation", m),
        )

        for item in state.get("vector_ota_results") or []:
            text = item.get("content") or ""
            meta = _chunk_metadata(item)
            if meta.get("recommendation_text") or "recommendation" in str(meta.get("doc_kind") or "").lower():
                label = "[OTA Recommendation]"
            elif meta.get("metric_text") or "metric" in str(meta.get("doc_kind") or "").lower():
                label = "[OTA Metric]"
            else:
                label = "[OTA Insight]"
            other_merged.append(
                {
                    "content": f"{label} {text}",
                    "source": "ota_insight",
                    "_context_kind": "ota_insight",
                    "metadata": meta,
                    "score": item.get("score"),
                }
            )

        merged = bq_merged + other_merged

        update_current_span_metadata(
            {
                "bq_count": len(state.get("bq_results") or []),
                "news_count": len(state.get("vector_news_results") or []),
                "academic_papers_count": len(state.get("vector_academic_papers_results") or []),
                "policies_count": len(state.get("vector_policies_results") or []),
                "public_reports_count": len(state.get("vector_public_reports_results") or []),
                "formation_count": len(state.get("vector_formation_results") or []),
                "academic_count": len(state.get("vector_academic_results") or []),
                "ota_count": len(state.get("vector_ota_results") or []),
                "merged_count": len(merged),
            }
        )
        # OFIA Pillar 1: annotate every merged chunk with its OFIA evidence tier.
        # This makes tier visible to ACF, logging, and future retrieval filtering
        # without requiring any ingestion-side changes.
        for chunk in merged:
            if "_ofia_tier" not in chunk:
                chunk["_ofia_tier"] = infer_source_tier(chunk)

        if _slot_path_active(state):
            rp = _reasoner_from_state(state)
            assert rp is not None
            rows_by = partition_bq_by_subquestion(list(state.get("bq_results") or []), rp)
            serializable = {k: v for k, v in rows_by.items()}
            return {
                "merged_context": other_merged,
                "passages": other_merged,
                "rows_by_subquestion": serializable,
            }

        return {"merged_context": merged}


def node_rerank(state: RAGGraphState) -> dict[str, Any]:
    query = state.get("query") or ""
    merged = state.get("merged_context") or []
    rerank_t0 = time.perf_counter()
    task_mode = str(state.get("task_mode") or "chat").strip().lower()
    if task_mode == "fact_lookup":
        top_k = int(state.get("rerank_top_k") or _FACT_LOOKUP_RERANK_TOP_K)
        default_pool = _FACT_LOOKUP_RERANK_POOL
    elif task_mode == "briefing":
        top_k = int(state.get("rerank_top_k") or _BRIEFING_RERANK_TOP_K)
        default_pool = _BRIEFING_RERANK_POOL
    else:
        top_k = int(state.get("rerank_top_k") or _DEFAULT_RERANK_TOP_K)
        default_pool = _DEFAULT_RERANK_POOL_SIZE

    corpus_sel = state.get("corpus_selection")
    active_corpora = 6
    if isinstance(corpus_sel, dict):
        active_raw = corpus_sel.get("active")
        if isinstance(active_raw, list):
            active_corpora = len(active_raw)
    if len(merged) <= 8 and active_corpora <= 1:
        packed = diversify_context_pack(
            list(merged),
            top_k=top_k if top_k > 0 else None,
            task_mode=task_mode,
        )
        if top_k > 0 and len(packed) > top_k:
            packed = packed[:top_k]
        return {
            "reranked_context": packed,
            "rerank_mode": "skipped_trivial",
            "rerank_ms": trace_elapsed_ms(rerank_t0),
            "rerank_pool_size": 0,
        }

    try:
        pool = int(
            os.environ.get("RAG_RERANK_POOL_SIZE", str(default_pool))
            or default_pool
        )
    except ValueError:
        pool = default_pool
    score_k = max(top_k, min(pool, max(len(merged), top_k)))
    dec = state.get("decomposition") if isinstance(state.get("decomposition"), dict) else None
    numeric_query = is_numeric_data_query(str(query), dec)
    comparative_query = is_comparative_bq_query(str(query), dec)
    scored = rerank(
        query,
        merged,
        top_k=score_k,
        numeric_query=numeric_query,
        comparative_query=comparative_query,
    )
    packed = diversify_context_pack(
        scored,
        top_k=None,
        task_mode=task_mode,
    )
    tc_raw = state.get("turn_contract")
    contract = TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None)
    slot_active = _slot_path_active(state)
    reasoner = _reasoner_from_state(state)
    rows_raw = state.get("rows_by_subquestion")
    rows_by_slot: dict[str, list[dict[str, Any]]] = {}
    if isinstance(rows_raw, dict):
        for k, v in rows_raw.items():
            if isinstance(v, list):
                rows_by_slot[str(k)] = [r for r in v if isinstance(r, dict)]
    bq_results = state.get("bq_results") or []
    if slot_active and reasoner and not rows_by_slot and bq_results:
        rows_by_slot = partition_bq_by_subquestion(list(bq_results), reasoner)
    bq_failed = not any(is_usable_structured_bq_row(r) for r in bq_results)
    if slot_active and reasoner and rows_by_slot:
        bq_failed = not any(
            is_usable_structured_bq_row(r)
            for bag in rows_by_slot.values()
            for r in bag
        )
    if contract.measure_id or contract.job != "fact":
        packed = typed_context_pack(
            packed,
            contract,
            top_k=top_k if top_k > 0 else None,
            bq_failed=bq_failed,
        )
    elif should_zero_pack(contract, bq_failed=bq_failed, context_items=packed):
        packed = []
    # Honor explicit rerank_top_k as an upper bound after diversity packing.
    if top_k > 0 and len(packed) > top_k:
        packed = packed[:top_k]
    return {
        "reranked_context": packed,
        "rerank_mode": last_rerank_mode(),
        "rerank_ms": trace_elapsed_ms(rerank_t0),
        "rerank_pool_size": pool,
    }


def _has_usable_internal_context(reranked: list[dict[str, Any]]) -> bool:
    """Did internal retrieval produce anything trustworthy on its own?

    Mirrors the existing "min usable chunks" gate used by ``needs_web_fallback``.
    """
    usable = filter_context_items(reranked or [])
    min_chunks = max(1, int(os.environ.get("RAG_WEB_FALLBACK_MIN_CHUNKS", "3") or 3))
    # Internal context is "usable on its own" only when we cleared the same bar
    # that would have skipped the web fallback in the first place.
    return len(usable) >= min_chunks


def _warehouse_attempt_made(state: RAGGraphState) -> bool:
    """True when BQ/class engines produced SQL debug or planned queries."""
    if state.get("bq_sql_debug") or state.get("bq_sql_queries"):
        return True
    plan = state.get("bq_sql_plan")
    if isinstance(plan, dict):
        if plan.get("bq_sql_debug") or plan.get("bq_sql_queries"):
            return True
        if plan.get("engine_results"):
            return True
    sp = state.get("supervisor_plan")
    if isinstance(sp, dict) and sp.get("classes"):
        return True
    return False


def node_web_fallback(state: RAGGraphState) -> dict[str, Any]:
    """Append Wikipedia / Tavily chunks when internal retrieval is weak.

    Guardrails:
    - If supplemental web search is rate-limited or errors out AND we don't
      already have enough usable internal context, set
      ``insufficient_context=True`` so the graph routes to the "I don't have
      enough information" branch instead of letting the generator fabricate
      around stale / tangential internal chunks.
    """
    with observed_span("web_fallback"):
        reranked = list(state.get("reranked_context") or [])
        bq_results = state.get("bq_results") or []
        has_bq = any(
            is_usable_structured_bq_row(r) for r in bq_results if isinstance(r, dict)
        )
        if not needs_web_fallback(
            reranked,
            task_mode=str(state.get("task_mode") or ""),
            has_usable_bq=has_bq,
            bq_warehouse_attempted=_warehouse_attempt_blocks_web(cast(dict[str, Any], state)),
            heavy_path=bool(state.get("heavy_path")),
        ):
            update_current_span_metadata({"web_fallback_status": "skipped"})
            return {}

        _pt = str(state.get("plan_type") or "").strip()
        if _pt.lower() == "free":
            update_current_span_metadata({"web_fallback_status": "disabled"})
            return {"web_fallback_status": "disabled", "web_fallback_reason": "Free plan - web fallback not available"}

        q = (state.get("query") or "").strip()
        dec = state.get("decomposition") if isinstance(state.get("decomposition"), dict) else None
        ts = (state.get("time_start_override") or (dec or {}).get("time_start") or "").strip()[:10]
        te = (state.get("time_end_override") or (dec or {}).get("time_end") or "").strip()[:10]
        try:
            result = retrieve_web_fallback_detailed(
                q,
                dec,
                geo_override=str(state.get("geo_override") or ""),
                time_start=ts or None,
                time_end=te or None,
                plan_type=str(state.get("plan_type") or ""),
                task_mode=str(state.get("task_mode") or ""),
                heavy_path=bool(state.get("heavy_path")),
            )
        except Exception:
            logger.exception("Web fallback retrieval raised; treating as error")
            result = None  # type: ignore[assignment]

        out: dict[str, Any] = {}

        if result is None:
            out["web_fallback_status"] = "error"
            out["web_fallback_reason"] = "exception in retrieve_web_fallback_detailed"
            if not _has_usable_internal_context(reranked):
                if _warehouse_attempt_made(state):
                    out["insufficient_context"] = True
            update_current_span_metadata(
                {
                    "web_fallback_status": out["web_fallback_status"],
                    "insufficient_context": bool(out.get("insufficient_context")),
                }
            )
            return out

        out["web_fallback_status"] = result.status
        out["web_fallback_reason"] = result.reason

        if result.items:
            web_chunks = [format_web_chunk_for_context(item) for item in result.items]
            out["web_results"] = web_chunks
            out["reranked_context"] = reranked + web_chunks
            update_current_span_metadata(
                {
                    "web_fallback_status": result.status,
                    "web_result_count": len(web_chunks),
                }
            )
            return out

        # No web items recovered the query. Only proceed to the standard generator
        # if internal context was already strong enough on its own — otherwise mark
        # the turn as insufficient so we don't hallucinate around tangential chunks.
        if result.status in ("rate_limited", "error", "disabled", "empty") and not _has_usable_internal_context(reranked):
            logger.info(
                "Web fallback status=%s with weak internal context — routing to insufficient_context. reason=%s",
                result.status,
                result.reason,
            )
            out["insufficient_context"] = True

        update_current_span_metadata(
            {
                "web_fallback_status": result.status,
                "insufficient_context": bool(out.get("insufficient_context")),
            }
        )
        return out


_INSUFFICIENT_CONTEXT_ANSWER = insufficient_context_answer("en")


def node_insufficient_context(state: RAGGraphState) -> dict[str, Any]:
    """Deterministic, non-hallucinating response when grounding is unavailable."""
    status = state.get("web_fallback_status") or "unknown"
    reason = state.get("web_fallback_reason") or ""
    query = str(state.get("query") or "")
    answer_lang = str(state.get("answer_lang") or detect_answer_language(query))
    canned_lang = detect_canned_insufficient_lang(query)
    bq_debug = list(state.get("bq_sql_debug") or []) if isinstance(state.get("bq_sql_debug"), list) else []
    raw_plan = state.get("bq_sql_plan")
    plan_dict = raw_plan if isinstance(raw_plan, dict) else {}
    pre_queries = list(plan_dict.get("bq_sql_queries") or [])
    exec_flags = bq_execute_flags(bq_debug, pre_queries=pre_queries, usable_bq=False)
    dec = state.get("decomposition")
    dec_dict = dec if isinstance(dec, dict) else None
    tc_raw = state.get("turn_contract")
    tc_dict = tc_raw if isinstance(tc_raw, dict) else None
    tc = TurnContract.from_dict(tc_dict) if isinstance(tc_dict, dict) else None
    context = list(state.get("reranked_context") or state.get("merged_context") or [])
    if should_generate_weak(
        cast(dict[str, Any], state),
        context_items=context,
        exec_flags=exec_flags,
        turn_contract=tc,
    ):
        miss_block = build_filter_miss_block(
            query=query,
            decomposition=dec_dict,
            turn_contract=tc,
            exec_flags=exec_flags,
            class_code=primary_class_from_state(cast(dict[str, Any], state)),
        )
        gen_result = generate(
            query,
            [{"content": miss_block, "_context_kind": "filter_miss"}],
            generate_weak=True,
            evidence_tier="weak",
            task_mode=str(state.get("task_mode") or ""),
            turn_contract=tc_dict,
        )
        acf = gen_result.acf or no_evidence_acf()
        with observed_span(
            "insufficient_context",
            input_data={"web_fallback_status": str(status)[:80], "generate_weak": True},
        ):
            update_current_span_metadata(
                {
                    "web_fallback_status": str(status),
                    "reason": str(reason)[:200],
                    "answer_lang": answer_lang,
                    "generate_weak": True,
                    "evidence_tier": "weak",
                }
            )
        return {
            "answer": gen_result.answer,
            "citations": [],
            "insufficient_context": True,
            "answer_lang": answer_lang,
            "evidence_tier": "weak",
            **acf_result_to_state(acf),
        }
    hard = _typed_bq_hard_return(state, exec_flags, bq_debug=bq_debug)
    if hard:
        answer = hard["answer"]
    elif forbid_generic_insufficient(cast(dict[str, Any], state)):
        if tc is not None and isinstance(tc_dict, dict):
            answer = empty_answer_for_contract(tc, query=query) or typed_bq_gap_answer(
                flag="structured_bq_never_executed",
                decomposition=dec_dict,
                turn_contract=tc_dict,
            )
        else:
            answer = typed_bq_gap_answer(
                flag="structured_bq_never_executed",
                decomposition=dec_dict,
                turn_contract=tc_dict,
            )
    else:
        answer = insufficient_context_answer(query=query)
    with observed_span(
        "insufficient_context",
        input_data={"web_fallback_status": str(status)[:80]},
    ):
        update_current_span_metadata(
            {
                "web_fallback_status": str(status),
                "reason": str(reason)[:200],
                "answer_lang": answer_lang,
                "insufficient_canned_lang": canned_lang,
                "warehouse_attempted": warehouse_was_attempted(cast(dict[str, Any], state)),
            }
        )
        logger.info(
            "insufficient_context: status=%s reason=%s",
            status,
            reason,
        )
        return {
            "answer": answer,
            "citations": [],
            "insufficient_context": True,
            "answer_lang": answer_lang,
            **acf_result_to_state(no_evidence_acf()),
        }


def _route_after_web_fallback(state: RAGGraphState) -> str:
    """Route post-web_fallback: either explicit 'insufficient' branch or normal generate."""
    if state.get("insufficient_context"):
        return "insufficient_context"
    return "generate"


def node_generate_language_help(state: RAGGraphState) -> dict[str, Any]:
    """Short-circuit when query language cannot be named. No retrieval, no memory."""
    query = state.get("query") or ""
    answer_lang = str(state.get("answer_lang") or "unknown")
    with observed_span("generate_language_help", input_data={"query": str(query)[:200]}):
        update_current_span_metadata({"answer_lang": answer_lang, "route": "language_unknown"})
        answer = language_unclear_answer()
    return {
        "answer": answer,
        "citations": [],
        "answer_lang": answer_lang,
        "is_language_unknown": True,
        **acf_result_to_state(curated_product_acf()),
    }


def node_generate(state: RAGGraphState) -> dict[str, Any]:
    query = state.get("query") or ""
    dec = state.get("decomposition")
    dec_dict = dec if isinstance(dec, dict) else None
    context = filter_context_items(state.get("reranked_context") or [])
    bq_results = state.get("bq_results") or []
    usable_bq = [r for r in bq_results if is_usable_structured_bq_row(r)]
    analytical = bool(state.get("analytical_mode"))
    task_mode = str(state.get("task_mode") or ("analytical" if analytical else "chat"))
    if (
        analytical
        or task_mode in ("fact_lookup", "data_export_only")
        or should_elevate_bq_context(str(query), dec_dict, usable_bq=bool(usable_bq))
    ):
        context = pin_bq_context_first(context)
    gkw: dict[str, Any] = {"decomposition": dec_dict, "task_mode": task_mode}
    if analytical:
        gkw["analytical_mode"] = True
    cs = state.get("conversation_summary")
    rt = state.get("recent_turns")
    has_mem = (isinstance(cs, str) and cs.strip()) or (
        isinstance(rt, list) and len(rt) > 0
    )
    if has_mem:
        if memory_relevant_for_query(
            query,
            cs if isinstance(cs, str) else "",
            list(rt) if isinstance(rt, list) else None,
            dec_dict,
        ):
            gkw["conversation_summary"] = cs if isinstance(cs, str) else ""
            gkw["recent_turns"] = list(rt) if isinstance(rt, list) else []
    elif state.get("chat_history"):
        from ml.rag.chat_history import normalize_messages as _norm_msgs

        hist = _norm_msgs(state.get("chat_history"))
        hist_text = "\n".join(m.get("content") or "" for m in hist)
        if memory_relevant_for_query(query, hist_text, hist, dec_dict):
            gkw["chat_history"] = state.get("chat_history")
    if state.get("plan_type"):
        gkw["plan_type"] = state.get("plan_type")
    if state.get("category"):
        gkw["category"] = state.get("category")
    if state.get("measure_id"):
        gkw["measure_id"] = state.get("measure_id")
    if state.get("recency_tier"):
        gkw["recency_tier"] = state.get("recency_tier")
    if state.get("answer_lang"):
        gkw["answer_lang"] = state.get("answer_lang")
    if state.get("export_intent"):
        gkw["export_intent"] = state.get("export_intent")
    raw_bq_debug = state.get("bq_sql_debug")
    bq_debug = list(raw_bq_debug) if isinstance(raw_bq_debug, list) else []
    raw_plan = state.get("bq_sql_plan")
    plan_dict = raw_plan if isinstance(raw_plan, dict) else {}
    pre_queries = list(plan_dict.get("bq_sql_queries") or [])
    exec_flags = bq_execute_flags(
        bq_debug,
        pre_queries=pre_queries,
        usable_bq=bool(usable_bq),
        compile_error=bool(plan_dict.get("compile_error")),
    )
    gkw.update(exec_flags)
    gkw["pre_queries"] = pre_queries
    gkw["usable_bq"] = bool(usable_bq)
    gkw["bq_sql_debug"] = bq_debug
    tc_raw = state.get("turn_contract")
    tc = TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None)
    is_numeric_job = bool(tc and tc.job in NUMERIC_JOBS) or is_numeric_data_query(
        str(query), dec_dict
    )
    hard = _typed_bq_hard_return(state, exec_flags, bq_debug=bq_debug)
    if should_hard_return_bq_gap(
        exec_flags=exec_flags,
        pre_queries=pre_queries,
        usable_bq=bool(usable_bq),
        context_items=context,
        is_numeric_job=is_numeric_job,
        state=cast(dict[str, Any], state),
    ):
        if hard is not None:
            return hard
    if not usable_bq and not any(
        exec_flags.get(k)
        for k in (
            "structured_bq_timed_out",
            "structured_bq_never_executed",
            "structured_bq_empty",
            "structured_bq_validation_failed",
            "structured_bq_compile_error",
            "structured_bq_unavailable",
        )
    ):
        gkw["structured_bq_unavailable"] = True
    if usable_bq and is_numeric_data_query(str(query), dec_dict):
        gkw["structured_bq_numeric_available"] = True
    elif usable_bq and is_comparative_bq_query(str(query), dec_dict):
        gkw["structured_bq_comparative_available"] = True
    reranked = state.get("reranked_context") or []
    measure_hit = None
    mid = state.get("measure_id")
    if mid:
        spec = MEASURES.get(str(mid))
        if spec:
            measure_hit = MeasureHit(spec, score=100, matched_alias=str(mid))
    contract_dict: dict[str, Any] | None = None
    if dec_dict:
        contract_dict = {
            "primary_measures": dec_dict.get("primary_measures") or [],
            "companion_measures": dec_dict.get("companion_measures") or [],
        }
    slot_active = _slot_path_active(state)
    reasoner = _reasoner_from_state(state)
    rows_raw = state.get("rows_by_subquestion")
    rows_by: dict[str, list[dict[str, Any]]] = {}
    if isinstance(rows_raw, dict):
        for k, v in rows_raw.items():
            if isinstance(v, list):
                rows_by[str(k)] = [r for r in v if isinstance(r, dict)]
    if slot_active and reasoner and not rows_by and usable_bq:
        rows_by = partition_bq_by_subquestion(list(bq_results), reasoner)
    reasoner_dict = (
        state.get("reasoner_plan") if isinstance(state.get("reasoner_plan"), dict) else None
    )
    gen_plan = build_generation_plan(
        str(query),
        task_mode=task_mode,
        decomposition=dec_dict,
        measure_hit=measure_hit,
        retrieval_contract=contract_dict,
        reranked_context=list(reranked),
        plan_type=str(state.get("plan_type") or "") or None,
        category=str(state.get("category") or "") or None,
        measure_id=str(mid) if mid else None,
        turn_contract=state.get("turn_contract") if isinstance(state.get("turn_contract"), dict) else None,
        reasoner_plan=reasoner_dict,
        rows_by_subquestion=rows_by if slot_active else None,
    )
    gkw["generation_plan"] = gen_plan.to_dict()
    has_required_bq_rows = False
    if slot_active and reasoner:
        passages = filter_context_items(
            state.get("passages") or list(reranked) or context
        )
        comp_block = composer_context_block(
            rows_by_slot=rows_by,
            passages=passages,
            reasoner=reasoner,
        )
        context = [
            {
                "content": comp_block,
                "_context_kind": "composer",
                "metadata": {"context_kind": "composer"},
            },
            *passages,
        ]
        gkw["composer_addendum"] = composer_addendum(reasoner, rows_by)
        has_required_bq_rows = any(
            rows_by.get(sq.id) for sq in reasoner.bq_subquestions() if sq.required
        )
    tc_raw = state.get("turn_contract")
    if isinstance(tc_raw, dict):
        gkw["turn_contract"] = tc_raw
    if isinstance(tc_raw, dict):
        tc = TurnContract.from_dict(tc_raw)
        academic_count = sum(
            1 for it in (state.get("reranked_context") or [])
            if str((it.get("metadata") or {}).get("context_kind") or it.get("_context_kind") or "").lower()
            in ("academic", "public_report")
        )
        empty_tpl = empty_answer_for_contract(
            tc,
            query=str(query),
            academic_count=academic_count,
            category=gen_plan.effective_category,
        )
        weak_ok = should_generate_weak(
            cast(dict[str, Any], state),
            context_items=context,
            exec_flags=exec_flags,
            turn_contract=tc,
        )
        if weak_ok:
            gen_t0_weak = time.perf_counter()
            miss_block = build_filter_miss_block(
                query=str(query),
                decomposition=dec_dict,
                turn_contract=tc,
                exec_flags=exec_flags,
                class_code=primary_class_from_state(cast(dict[str, Any], state)),
            )
            gkw_weak = dict(gkw)
            gkw_weak["generate_weak"] = True
            gkw_weak["evidence_tier"] = "weak"
            gen_result = generate(
                str(query),
                [{"content": miss_block, "_context_kind": "filter_miss"}],
                **gkw_weak,
            )
            acf = gen_result.acf or no_evidence_acf()
            return {
                "answer": gen_result.answer,
                "citations": [],
                "answer_lang": str(state.get("answer_lang") or detect_answer_language(query)),
                "generation_plan": gen_plan.to_dict(),
                "evidence_tier": "weak",
                "generate_ms": trace_elapsed_ms(gen_t0_weak),
                **acf_result_to_state(acf),
                **{k: True for k in exec_flags if exec_flags.get(k)},
            }
        if gen_plan.output_type == "insufficient" and empty_tpl and not (slot_active and has_required_bq_rows):
            if resolve_empty_policy(cast(dict[str, Any], state)) != "generate_weak":
                return {
                    "answer": empty_tpl,
                    "citations": [],
                    "answer_lang": str(state.get("answer_lang") or detect_answer_language(query)),
                    "generation_plan": gen_plan.to_dict(),
                    **acf_result_to_state(no_evidence_acf()),
                }
        if empty_tpl and (
            not context
            or should_zero_pack(tc, bq_failed=not usable_bq, context_items=context)
        ) and not (slot_active and has_required_bq_rows):
            if resolve_empty_policy(cast(dict[str, Any], state)) != "generate_weak":
                return {
                    "answer": empty_tpl,
                    "citations": [],
                    "answer_lang": str(state.get("answer_lang") or detect_answer_language(query)),
                    "generation_plan": gen_plan.to_dict(),
                    **acf_result_to_state(no_evidence_acf()),
                }
        if (
            tc.is_fail_closed()
            and tc.serve_status != "clarify"
            and tc.vector_policy != "fallback_only"
            and not (slot_active and has_required_bq_rows)
        ):
            from ml.rag.chatbot.answer_language import insufficient_context_answer

            hint = unsupported_answer_hint(tc) or empty_tpl
            return {
                "answer": hint or insufficient_context_answer(query=str(query)),
                "citations": [],
                "answer_lang": str(state.get("answer_lang") or detect_answer_language(query)),
                "generation_plan": gen_plan.to_dict(),
                **acf_result_to_state(no_evidence_acf()),
            }
    if gen_plan.effective_category:
        gkw["category"] = gen_plan.effective_category
    gen_t0 = time.perf_counter()
    gen_result = generate(query, context, **gkw)
    gen_max = _generate_max_tokens(task_mode)

    # ACF Path B is computed post-cite inside generate/_finalize_generation_result.
    acf = gen_result.acf or no_evidence_acf()
    answer_lang = str(state.get("answer_lang") or detect_answer_language(query))

    return {
        "answer": gen_result.answer,
        "citations": gen_result.citations,
        "answer_lang": answer_lang,
        "generation_plan": gen_plan.to_dict(),
        "generate_ms": trace_elapsed_ms(gen_t0),
        "generate_max_tokens": gen_max,
        "generate_input_chars": getattr(gen_result, "generate_input_chars", None),
        **acf_result_to_state(acf),
        **{k: True for k in exec_flags if exec_flags.get(k)},
    }


def node_export(state: RAGGraphState) -> dict[str, Any]:
    """Build downloadable artifacts when export is enabled on the route."""
    export_intent = state.get("export_intent")
    export_enabled = bool(state.get("export_enabled"))
    out: dict[str, Any] = {"artifacts": []}

    if not export_intent:
        return out

    answer = str(state.get("answer") or "")

    if not export_enabled:
        if EXPORT_UPGRADE_MESSAGE not in answer:
            out["answer"] = f"{answer}\n\n{EXPORT_UPGRADE_MESSAGE}".strip()
        return out

    try:
        artifacts = run_exports(
            export_kind=export_intent,  # type: ignore[arg-type]
            query=str(state.get("query") or ""),
            answer=answer,
            bq_results=state.get("bq_results"),
            citations=state.get("citations"),
            state=dict(state),
            export_enabled=export_enabled,
            plan_type=state.get("plan_type"),
        )
        out["artifacts"] = artifacts
        if artifacts:
            links = ", ".join(a.get("filename", "") for a in artifacts if a.get("filename"))
            suffix = f"\n\nDownloadable files are attached to this response: {links}."
            if suffix.strip() not in answer:
                out["answer"] = f"{answer}{suffix}".strip()
    except Exception as exc:
        logger.warning("export node failed: %s", exc)
        out["answer"] = (
            f"{answer}\n\nI could not generate the requested export ({exc}). "
            "Try asking again once structured data is available in the answer."
        ).strip()

    return out


def node_generate_clarify(state: RAGGraphState) -> dict[str, Any]:
    """Ask for missing slots (measure-aware) before retrieval."""
    query = state.get("query") or ""
    answer_lang = str(state.get("answer_lang") or detect_answer_language(query))
    dec = state.get("decomposition") if isinstance(state.get("decomposition"), dict) else None
    with observed_span("generate_clarify", input_data={"query": str(query)[:200]}):
        update_current_span_metadata(
            {
                "answer_lang": answer_lang,
                "route": "clarify",
                "task_mode": "clarify",
                "measure_id": state.get("measure_id"),
            }
        )
        answer = clarify_answer(str(query), decomposition=dec)
    return {
        "answer": answer,
        "citations": [],
        "answer_lang": answer_lang,
        "task_mode": "clarify",
        **acf_result_to_state(
            no_evidence_acf(
                explanation="Clarification requested — missing slot(s) before federated retrieval."
            )
        ),
    }


def node_generate_meta(state: RAGGraphState) -> dict[str, Any]:
    """Short-circuit node for identity meta questions. No retrieval."""
    from ml.rag.chatbot.assistant_identity import generate_meta_answer

    query = state.get("query") or ""
    answer_lang = str(state.get("answer_lang") or detect_answer_language(query))
    gkw: dict[str, Any] = {}
    cs = state.get("conversation_summary")
    rt = state.get("recent_turns")
    has_mem = (isinstance(cs, str) and cs.strip()) or (isinstance(rt, list) and len(rt) > 0)
    if has_mem:
        gkw["conversation_summary"] = cs if isinstance(cs, str) else ""
        gkw["recent_turns"] = list(rt) if isinstance(rt, list) else []
    elif state.get("chat_history"):
        gkw["chat_history"] = state.get("chat_history")
    if state.get("plan_type"):
        gkw["plan_type"] = state.get("plan_type")
    if state.get("category"):
        gkw["category"] = state.get("category")
    with observed_span("generate_meta", input_data={"query": str(query)[:200]}):
        update_current_span_metadata({"answer_lang": answer_lang, "route": "meta"})
        answer = generate_meta_answer(query, **gkw)

    return {
        "answer": answer,
        "citations": [],
        "answer_lang": answer_lang,
        **acf_result_to_state(curated_product_acf()),
    }


def node_generate_social(state: RAGGraphState) -> dict[str, Any]:
    """Short-circuit greetings / out-of-scope. No retrieval and no chat-memory injection."""
    query = state.get("query") or ""
    answer_lang = str(state.get("answer_lang") or detect_answer_language(query))
    kind = classify_social_query(query, state.get("decomposition"))
    if kind is None:
        kind = "greeting" if state.get("is_greeting_query") else "out_of_scope"
    route = "greeting" if kind == "greeting" else "out_of_scope"
    with observed_span("generate_social", input_data={"query": str(query)[:200]}):
        update_current_span_metadata({"answer_lang": answer_lang, "route": route})
        answer = generate_social_answer(kind, query, answer_lang=answer_lang)

    return {
        "answer": answer,
        "citations": [],
        "answer_lang": answer_lang,
        "is_greeting_query": kind == "greeting",
        "is_out_of_scope_query": kind == "out_of_scope",
        **acf_result_to_state(curated_product_acf()),
    }


def node_generate_product(state: RAGGraphState) -> dict[str, Any]:
    """Short-circuit node for OpenTrace product questions. Uses product KB, no retrieval."""
    from ml.rag.chatbot.product_knowledge import generate_product_answer

    query = state.get("query") or ""
    answer_lang = str(state.get("answer_lang") or detect_answer_language(query))
    gkw: dict[str, Any] = {}
    cs = state.get("conversation_summary")
    rt = state.get("recent_turns")
    has_mem = (isinstance(cs, str) and cs.strip()) or (isinstance(rt, list) and len(rt) > 0)
    if has_mem:
        gkw["conversation_summary"] = cs if isinstance(cs, str) else ""
        gkw["recent_turns"] = list(rt) if isinstance(rt, list) else []
    elif state.get("chat_history"):
        gkw["chat_history"] = state.get("chat_history")
    if state.get("plan_type"):
        gkw["plan_type"] = state.get("plan_type")
    if state.get("category"):
        gkw["category"] = state.get("category")
    with observed_span("generate_product", input_data={"query": str(query)[:200]}):
        route = "help" if state.get("is_help_query") else "product"
        update_current_span_metadata({"answer_lang": answer_lang, "route": route})
        gen_t0 = time.perf_counter()
        answer = generate_product_answer(query, **gkw)
        generate_ms = trace_elapsed_ms(gen_t0)

    return {
        "answer": answer,
        "citations": [],
        "answer_lang": answer_lang,
        "generate_ms": generate_ms,
        "generate_max_tokens": 0 if state.get("is_help_query") else _generate_max_tokens("chat"),
        "generate_input_chars": len(answer),
        **acf_result_to_state(curated_product_acf()),
    }


def build_graph():
    """Build and compile the LangGraph RAG graph. Requires langgraph."""
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("Install langgraph: pip install langgraph") from None

    graph = StateGraph(RAGGraphState)

    graph.add_node("decompose", node_decompose)
    graph.add_node("parallel_retrieve", node_parallel_retrieve)
    graph.add_node("bq_reason", node_bq_reason)
    graph.add_node("bq_retrieve", node_bq_retrieve)
    graph.add_node("merge", node_merge)
    graph.add_node("rerank", node_rerank)
    graph.add_node("web_fallback", node_web_fallback)
    graph.add_node("insufficient_context", node_insufficient_context)
    graph.add_node("generate", node_generate)
    graph.add_node("export", node_export)
    graph.add_node("generate_meta", node_generate_meta)
    graph.add_node("generate_product", node_generate_product)
    graph.add_node("generate_social", node_generate_social)
    graph.add_node("generate_language_help", node_generate_language_help)
    graph.add_node("generate_clarify", node_generate_clarify)

    graph.add_edge(START, "decompose")

    def _route_after_decompose(state: RAGGraphState) -> str:
        if state.get("is_meta_query"):
            return "generate_meta"
        if state.get("is_product_query"):
            return "generate_product"
        if state.get("is_greeting_query") or state.get("is_out_of_scope_query"):
            return "generate_social"
        if state.get("is_language_unknown"):
            return "generate_language_help"
        if state.get("task_mode") == "clarify":
            return "generate_clarify"
        tc_raw = state.get("turn_contract")
        contract = TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None)
        if not contract.should_retrieve_vector():
            return "bq_reason"
        return "parallel_retrieve"

    graph.add_conditional_edges("decompose", _route_after_decompose)
    graph.add_edge("parallel_retrieve", "bq_reason")
    graph.add_edge("bq_reason", "bq_retrieve")
    graph.add_edge("bq_retrieve", "merge")
    graph.add_edge("merge", "rerank")
    graph.add_conditional_edges("rerank", route_after_rerank)
    graph.add_conditional_edges("web_fallback", _route_after_web_fallback)
    graph.add_edge("generate", "export")
    graph.add_edge("export", END)
    graph.add_edge("insufficient_context", END)
    graph.add_edge("generate_meta", END)
    graph.add_edge("generate_product", END)
    graph.add_edge("generate_social", END)
    graph.add_edge("generate_language_help", END)
    graph.add_edge("generate_clarify", END)

    return graph.compile()


# Compiled once at import time (cheap and safe for the production container).
_COMPILED_GRAPH = None

def _get_compiled_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def run_rag(query: str, **kwargs: Any) -> dict[str, Any]:
    """Run the RAG pipeline and return the state (including answer)."""
    reset_llm_usage()
    graph = _get_compiled_graph()
    initial: RAGGraphState = {"query": query}
    user_profile = kwargs.get("user_profile") if isinstance(kwargs.get("user_profile"), dict) else None
    plan_type = kwargs.get("plan_type")
    profile_geo = effective_geo_override(plan_type, user_profile)
    if profile_geo:
        initial["geo_override"] = profile_geo  # type: ignore[assignment]
    if plan_type:
        initial["plan_type"] = plan_type  # type: ignore[assignment]
    if kwargs.get("category"):
        initial["category"] = kwargs["category"]  # type: ignore[assignment]
    if user_profile is not None:
        initial["user_profile"] = user_profile  # type: ignore[assignment]
    for key in (
        "time_start_override",
        "time_end_override",
        "news_top_k",
        "academic_top_k",
        "bq_top_k",
        "rerank_top_k",
        "chat_history",
    ):
        if key in kwargs and kwargs[key] is not None:
            initial[key] = kwargs[key]  # type: ignore[assignment]
    if "session_id" in kwargs and kwargs["session_id"]:
        initial["session_id"] = kwargs["session_id"]  # type: ignore[assignment]
        sid = str(kwargs["session_id"]).strip()
        if sid:
            from ml.rag.session_store import get_session_blob

            blob = get_session_blob(sid)
            if isinstance(blob, dict) and blob.get("last_structured_ranking"):
                initial["structured_ranking_cache"] = blob["last_structured_ranking"]  # type: ignore[assignment]
    if "conversation_summary" in kwargs:
        initial["conversation_summary"] = kwargs["conversation_summary"]  # type: ignore[assignment]
    if "recent_turns" in kwargs:
        initial["recent_turns"] = kwargs["recent_turns"]  # type: ignore[assignment]
    if kwargs.get("export_enabled"):
        initial["export_enabled"] = True  # type: ignore[assignment]
    session_id = kwargs.get("session_id")
    trace_tags = kwargs.get("trace_tags")
    extra_tags = list(trace_tags) if isinstance(trace_tags, list) else None
    cfg = build_rag_invoke_config(
        base_config=kwargs.get("config"),
        session_id=str(session_id).strip() if session_id else None,
        plan_type=str(plan_type).strip() if plan_type else None,
        category=str(kwargs.get("category") or "").strip() or None,
        tags=extra_tags,
    )
    result = graph.invoke(initial, config=cfg)
    out = dict(result)
    out.setdefault("citations", [])
    out.setdefault("artifacts", [])
    out["usage"] = get_llm_usage().to_dict()
    return out
