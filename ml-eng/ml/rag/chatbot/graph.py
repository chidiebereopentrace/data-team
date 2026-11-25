"""
RAG graph: query → decompose → (Qdrant ‖ BQ reason+retrieve) → merge → rerank → generate.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import wait
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Literal, cast

from ml.rag.chatbot import rarr
from ml.rag.chatbot.acf_scoring import (
    ACFState,
    acf_result_to_state,
    curated_product_acf,
    no_evidence_acf,
)
from ml.rag.chatbot.active_request import (
    add_decomposed_slots,
    fill_decomposition,
    resolve_turn,
    resolved_request_text,
)
from ml.rag.chatbot.agri_measure_ontology import (
    MEASURES,
    MeasureHit,
    resolve_measures,
    resolve_recency_tier,
)
from ml.rag.chatbot.answer_language import (
    detect_answer_language,
    detect_language_with_confidence,
    language_unclear_answer,
)
from ml.rag.chatbot.answer_outcome import decide_outcome, gap_answer
from ml.rag.chatbot.assistant_identity import is_meta_query
from ml.rag.chatbot.bq_context_enrich import enrich_bq_results
from ml.rag.chatbot.bq_execute_state import (
    bq_execute_flags,
    collect_pre_queries,
    ensure_planned_attempt_debug,
    execution_results_from_items,
    plan_indicates_warehouse_attempt,
    warehouse_was_attempted,
)
from ml.rag.chatbot.bq_plan import normalize_bq_plan
from ml.rag.chatbot.bq_ranking_cache import (
    bq_results_from_cache,
    cache_entry_from_bq_results,
    is_ranking_follow_up,
)
from ml.rag.chatbot.bq_sql_diagnostics import (
    sql_attempts_for_diagnostics,
    sql_statements_for_diagnostics,
)
from ml.rag.chatbot.bq_sql_reasoner import _slot_reasoner_active, reason_bq_sql_plan
from ml.rag.chatbot.capability_registry import (
    apply_reasoner_to_turn,
    resolve_capability,
)
from ml.rag.chatbot.class_engine_runner import (
    engine_results_to_bq_plan,
    run_class_engines,
)
from ml.rag.chatbot.class_supervisor import (
    SupervisorPlan,
    compile_supervisor_plan,
    corpora_for_supervisor_plan,
    tables_for_supervisor_plan,
)
from ml.rag.chatbot.composer import (
    composer_addendum,
    composer_context_block,
    partition_bq_by_subquestion,
)
from ml.rag.chatbot.context_diversity import diversify_context_pack
from ml.rag.chatbot.corpus_catalog import (
    CORPUS_CATALOG,
    CorpusSelection,
    comparison_document_corpora,
    formation_doc_kinds,
    select_corpora,
)
from ml.rag.chatbot.decompose_context import DecomposeContext
from ml.rag.chatbot.document_evidence import (
    document_requirement_coverage,
    requirement_document_candidates,
)
from ml.rag.chatbot.evidence_kinds import is_document_item, is_warehouse_item
from ml.rag.chatbot.evidence_requirements import (
    fact_gap_description,
    fact_requirement_coverage,
    is_fact_comparison,
    matches_bound_fact,
    plan_fact_requirements,
    prioritize_bound_facts,
)
from ml.rag.chatbot.export_intent import (
    EXPORT_UPGRADE_MESSAGE,
    ExportKind,
    resolve_export_intent,
)
from ml.rag.chatbot.export_runner import run_exports
from ml.rag.chatbot.facet_compiler import compile_breakdown, compile_turn_contract
from ml.rag.chatbot.facet_enrich import enrich_decomposition_facets
from ml.rag.chatbot.generation_plan import build_generation_plan
from ml.rag.chatbot.generator import (
    _generate_max_tokens,
    drop_geo_conflicting,
    drop_price_off_topic,
    filter_context_items,
    generate,
    is_comparative_bq_query,
    is_mergeable_bq_evidence,
    is_numeric_data_query,
    is_usable_context_item,
    is_usable_structured_bq_row,
    pin_bq_context_first,
    should_elevate_bq_context,
)
from ml.rag.chatbot.geo_policy import (
    apply_profile_geography_default,
    clear_profile_ui_retrieval_scope,
    effective_geo_override,
    sanitize_profile_country,
)
from ml.rag.chatbot.geo_regions import expand_regions_in_decomposition
from ml.rag.chatbot.global_reasoner import (
    compile_reasoner_plan,
    reasoner_plan_to_bq_plan,
)
from ml.rag.chatbot.intent_bundles import (
    bundle_primary_measures,
    bundle_required_measures,
    match_intent_bundles,
)
from ml.rag.chatbot.language_enrich import enrich_query_to_english
from ml.rag.chatbot.memory_relevance import memory_relevant_for_query
from ml.rag.chatbot.metacog import (
    MetacogPlan,
    MetacogVerify,
    apply_metacog_to_turn,
    apply_revise_decision,
    build_metacog_plan,
    complete_self_check,
    corpora_for_evidence_needs,
    evidence_sample,
    metacog_diagnosis_addendum,
    metacog_enabled,
    metacog_verify_enabled,
    propose_metacog_revise,
    verify_metacog_evidence,
)
from ml.rag.chatbot.ofia import infer_source_tier
from ml.rag.chatbot.ontology_context import sanitize_decomposition_for_bq
from ml.rag.chatbot.plan_policy import (
    apply_plan_decomposition_gates,
)
from ml.rag.chatbot.price_sql import requested_price_list_products
from ml.rag.chatbot.product_knowledge import is_help_query, is_product_query
from ml.rag.chatbot.qdrant_planner import plan_vector_corpora_for_reasoner
from ml.rag.chatbot.query_decomposer import (
    _extract_countries,
    decompose_query,
    facet_grounded_in_query,
    normalize_geography_for_filter,
    resolve_retrieval_geographies,
)
from ml.rag.chatbot.query_gate import (
    classify_social_query,
    early_non_rag_route,
    generate_social_answer,
    is_greeting_query,
    is_out_of_scope_query,
)
from ml.rag.chatbot.query_ir import compile_early_query_ir, compile_query_ir
from ml.rag.chatbot.query_normalize import normalize_query_text
from ml.rag.chatbot.query_views import build_detail_rewrite, build_query_views
from ml.rag.chatbot.reasoner_plan import ReasonerPlan
from ml.rag.chatbot.reranker import last_rerank_mode, rerank
from ml.rag.chatbot.retrieval_contract import build_corpus_routing_contract
from ml.rag.chatbot.routing_plan import (
    RoutingPlan,
    Source,
    SourceSelection,
    build_routing_plan,
    select_sources,
)
from ml.rag.chatbot.sql_compiler import sql_compiler_enabled
from ml.rag.chatbot.stage_budget import (
    RetrievalWorkers,
    remaining_seconds,
    request_budget,
    stage_budget_seconds,
)
from ml.rag.chatbot.task_mode import clarify_answer, resolve_task_mode
from ml.rag.chatbot.time_resolver import (
    parse_iso_date,
    resolve_relative_time,
    utc_today,
)
from ml.rag.chatbot.time_retrieval import (
    sync_decomposition_time,
    time_fallback_enabled,
    time_kwargs_from_contract,
)
from ml.rag.chatbot.turn_contract import TurnContract
from ml.rag.chatbot.typed_pack import PackResult, typed_context_pack
from ml.rag.llm_chat import (
    get_llm_usage,
    length_capped_call_count,
    last_finish_reason,
    reset_llm_truncation,
    reset_llm_usage,
)
from ml.rag.observability import (
    build_rag_invoke_config,
    observed_span,
    run_with_tracing_context,
    trace_elapsed_ms,
    update_current_span_metadata,
)
from ml.rag.retrievers.bq_retriever import BQRetriever
from ml.rag.retrievers.vector_retriever import VectorRetriever
from ml.rag.retrievers.web_retriever import (
    format_web_chunk_for_context,
    needs_web_fallback,
    retrieve_web_fallback_detailed,
    route_after_rerank,
    warehouse_blocks_web,
)
from ml.rag.text_processors.preprocess.bibliographic_metadata import (
    format_academic_citation,
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
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    """Try full filters, widen time ±1y, then drop time/geo when fallbacks allow."""
    attempts: list[dict[str, Any]] = [dict(base_kwargs)]
    if base_kwargs.get("ontology_ids"):
        # Untagged legacy chunks never match the ontology filter: next level keeps geo/time, drops it.
        no_onto = dict(base_kwargs)
        no_onto.pop("ontology_ids", None)
        attempts.append(no_onto)
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
        if deadline is not None and time.perf_counter() >= deadline:
            raise FuturesTimeoutError("document retrieval deadline")
        raw = vr.retrieve(query, **kwargs)
        health = vr.last_index_health
        if health is not None and health.queryable is False:
            raise RuntimeError("Qdrant index unavailable or incompatible")
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


def _merge_vector_context_dicts(
    lists: list[list[dict[str, Any]]],
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Merge multi-query vector dict hits by point_id/content, keeping higher score."""
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for items in lists:
        for item in items or []:
            if not isinstance(item, dict):
                continue
            raw_meta = item.get("metadata")
            meta: dict[str, Any] = {}
            if isinstance(raw_meta, dict):
                meta = raw_meta
            pid = str(
                meta.get("point_id")
                or meta.get("doc_id")
                or item.get("id")
                or hash(str(item.get("content") or "")[:500])
            )
            score_raw = item.get("score")
            if score_raw is None:
                score_raw = meta.get("score")
            sc = float(score_raw or 0.0)
            if pid not in best or sc > best[pid][0]:
                best[pid] = (sc, item)
    merged = sorted(best.values(), key=lambda x: x[0], reverse=True)
    return [t[1] for t in merged[:limit]]


def _vector_texts_from_state(state: RAGGraphState) -> list[str]:
    from ml.rag.chatbot.query_views import QueryViews, build_query_views

    qv = state.get("query_views")
    if isinstance(qv, dict) and qv.get("user_query"):
        views = QueryViews(
            user_query=str(qv.get("user_query") or ""),
            context_rewrite=str(qv.get("context_rewrite") or ""),
            detail_rewrite=str(qv.get("detail_rewrite") or state.get("detail_rewrite") or ""),
            context_applied=bool(qv.get("context_applied")),
            language_enrich=str(
                qv.get("language_enrich") or state.get("language_enrich") or ""
            ),
        )
        texts = views.vector_texts()
        if texts:
            return texts
    uq = str(state.get("user_query") or state.get("query") or "").strip()
    ctx = str(state.get("context_rewrite") or "").strip()
    detail = str(state.get("detail_rewrite") or "").strip()
    lang_en = str(state.get("language_enrich") or "").strip()
    if uq:
        if not (ctx or detail or lang_en):
            return build_query_views(uq).vector_texts()
        return QueryViews(
            user_query=uq,
            context_rewrite=ctx,
            detail_rewrite=detail,
            context_applied=bool(state.get("context_applied")),
            language_enrich=lang_en,
        ).vector_texts()
    return [str(state.get("query") or "").strip()] if str(state.get("query") or "").strip() else []


def _document_search_texts(state: RAGGraphState) -> list[str]:
    texts = _vector_texts_from_state(state)
    dec = state.get("decomposition") or {}
    raw = state.get("turn_contract")
    topics = (
        [doc.question for doc in TurnContract.from_dict(raw).documents]
        if isinstance(raw, dict) else dec.get("evidence_topics")
    )
    if not isinstance(topics, list) or not topics:
        return texts
    # Keep the whole question and necessary language/context views. Replace the
    # generic detail rewrite with focused searches for the requested evidence.
    detail = state.get("detail_rewrite")
    focused = [topic for topic in topics if isinstance(topic, str) and topic.strip()]
    return list(dict.fromkeys([*(text for text in texts if text != detail), *focused]))


def _document_time_is_observation(state: RAGGraphState, contract: TurnContract) -> bool:
    """Observation dates belong in the search text, even for later releases."""
    sources = state.get("source_selection")
    indicator_context = bool(sources and sources["qdrant"]["reason"] == "indicator_context")
    if not indicator_context and not (is_fact_comparison(contract) and contract.documents):
        return False
    if contract.time_spec.time_role not in ("observation", "historical"):
        return False
    query = _bq_planning_query(state)
    return not re.search(r"\b(?:published|released|issued|dated)\s+(?:in|on|during|between|after|before)\b", query, re.I)


@dataclass
class _VectorProgress:
    deadline: float
    hits: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False


def _vector_retrieve_for_corpus(
    state: RAGGraphState,
    *,
    collection_env: str,
    default_collection: str,
    build_kwargs: Any,
    geo_fallback_env: str,
    time_fallback_env: str,
) -> list[dict[str, Any]]:
    """Shared retrieval: up to three query views + optional session KV."""
    from ml.rag.chatbot.retrieval_cache import (
        geo_year_from_kwargs,
        get_cached_hits,
        retrieval_cache_key,
        set_cached_hits,
    )

    texts = _document_search_texts(state)
    if not texts:
        return []
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
        or dec.get("time_start")
        or ""
    ).strip()[:10]
    te = str(
        time_kw.get("published_at_to")
        or dec.get("time_end")
        or ""
    ).strip()[:10]
    hard_filter = bool(time_kw.get("hard_filter"))

    kwargs = build_kwargs(state, dec)
    _apply_geo_to_kwargs(kwargs, countries)
    if (default_collection == "public_reports" or is_fact_comparison(contract)) and _document_time_is_observation(
        state, contract,
    ):
        # An assessment may be published after the events it explains.
        kwargs.pop("published_at_from", None)
        kwargs.pop("published_at_to", None)
    else:
        if ts:
            kwargs["published_at_from"] = ts
        if te:
            kwargs["published_at_to"] = te

    from ml.rag.chatbot.ontology_routing import (
        ontology_plan_for_state,
        qdrant_expansion_view,
        qdrant_payload_ids,
    )

    onto_plan = ontology_plan_for_state(state)
    if onto_plan:
        if os.environ.get("ADZA_QDRANT_ONTOLOGY_FILTER", "1").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        ):
            oids = qdrant_payload_ids(onto_plan)
            if oids:
                kwargs["ontology_ids"] = oids
        view = qdrant_expansion_view(onto_plan)
        if view and view not in texts:
            texts = [*texts, view]

    allow_geo_fb = not _strict_compare_filters(dec)
    task_mode = str(state.get("task_mode") or "chat").strip().lower()
    cascade_max = 1 if task_mode in ("fact_lookup", "data_export_only") else None
    time_fb = time_fallback_enabled(hard_filter=hard_filter, env_var=time_fallback_env)
    session_id = str(state.get("session_id") or "").strip() or None
    geo_key, year_key = geo_year_from_kwargs(kwargs)
    corpus_key = default_collection
    any_cache_hit = False
    per_text: list[list[dict[str, Any]]] = []
    progress = state.get("vector_progress")
    deadline = progress.deadline if progress else None
    top_k = int(kwargs.get("top_k") or 10)
    limit = max(top_k * 2, 20)
    for text in texts:
        if deadline is not None and time.perf_counter() >= deadline:
            raise FuturesTimeoutError("document retrieval deadline")
        cache_key = retrieval_cache_key(
            session_id=session_id,
            text=text,
            corpora=corpus_key,
            geo=geo_key,
            year=year_key,
            ontology_filter=bool(kwargs.get("ontology_ids")),
        )
        cached = get_cached_hits(cache_key)
        if cached is not None:
            any_cache_hit = True
            raw = cached
        else:
            raw = _retrieve_vector_cascade(
                vr,
                text,
                base_kwargs=kwargs,
                countries=countries,
                has_time=bool(ts or te),
                geo_fallback_env=geo_fallback_env,
                time_fallback_env=time_fallback_env,
                allow_geo_fallback=allow_geo_fb,
                max_levels=cascade_max,
                allow_time_fallback=time_fb,
                deadline=deadline,
            )
            if countries:
                raw = _post_filter_geography(raw, countries)
            set_cached_hits(cache_key, raw)
        per_text.append(raw)
        if progress:
            progress.cache_hit = any_cache_hit
            if raw:
                progress.hits = _merge_vector_context_dicts(per_text, limit=limit)
    # Stash cache flag on state dict when mutable
    if any_cache_hit and isinstance(state, dict):
        state["vector_cache_hit"] = True  # type: ignore[index]
    return _merge_vector_context_dicts(per_text, limit=limit)


class RAGGraphState(ACFState, total=False):
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
    evidence_drops: dict[str, int]
    retrieval_partial: bool
    web_results: list[dict[str, Any]]
    web_fallback_status: str | None
    web_fallback_reason: str | None
    insufficient_context: bool | None
    answer_outcome: str
    profile_scope_retry: int | None
    answer: str
    citations: list[dict[str, Any]]
    usage: dict[str, int]
    error: str | None
    # Optional UI / API overrides (see run_rag)
    geo_override: str | None
    time_start_override: str | None
    time_end_override: str | None
    reference_date: str | None
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
    vector_progress: _VectorProgress
    # Named answer-language tag (en|sw|fr|pcm|ar|am|ig|…|mixed|unknown)
    answer_lang: str | None
    # English enrich shadow for bind/retrieve (original stays in query/user_query)
    language_enrich: str | None
    language_enrich_applied: bool | None
    language_enrich_source: str | None
    language_enrich_subject_guard: bool | None
    # Session context — Sprint 1, Week 2 (session isolation)
    session_id: str | None
    # Export / enriched outputs (Agribusinesses + Integrated routes only)
    export_enabled: bool | None
    export_intent: ExportKind | None
    artifacts: list[dict[str, Any]]
    # Internal SQL planner/debug fields for class-engine and NL2SQL paths
    bq_table_candidates_inspector: list[dict[str, Any]] | None
    value_hits: dict[str, int] | None
    # Observability: stage-level counts for candidate/structured/retrieved (debugging)
    retrieval_stage_counts: dict[str, int] | None
    # Bilingual translation: English query variant for BQ retrieval (multi-lingual vectors use original lang)
    # Task specialization for full_rag (clarify / analytical / fact / briefing / export-only / chat)
    task_mode: str | None
    # analytical_mode is True when task_mode == "analytical"
    analytical_mode: bool | None
    # Enriched / query views (user_query never replaced as sole embed)
    enriched_query: str | None
    user_query: str | None
    context_rewrite: str | None
    detail_rewrite: str | None
    context_applied: bool | None
    query_views: dict[str, Any] | None
    active_request: dict[str, Any] | None
    resolved_request: str | None
    vector_texts_used: list[str] | None
    user_query_dropped: bool | None
    vector_cache_hit: bool | None
    coverage_retry: int | None
    last_bq_facts: list[dict[str, Any]] | None
    execution_results: list[dict[str, Any]]
    measure_id: str | None
    recency_tier: str | None
    turn_contract: dict[str, Any]
    comparison_gaps: list[dict[str, Any]]
    fact_requirement_coverage: list[dict[str, Any]]
    document_requirement_coverage: list[dict[str, Any]]
    routing_plan: dict[str, Any]
    source_selection: SourceSelection
    query_ir: dict[str, Any]
    supervisor_plan: dict[str, Any] | None
    generation_plan: dict[str, Any] | None
    reasoner_plan: dict[str, Any] | None
    metacog_plan: dict[str, Any] | None
    metacog_verify: dict[str, Any] | None
    metacog_revise: int | None
    metacog_revise_action: str | None
    metacog_next: str | None
    metacog_self_check: dict[str, Any] | None
    metacog_self_check_verdict: str | None
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
    metacog_llm_ms: float | None
    vector_ms: float | None
    corpus_count: int | None
    cascade_level: int | None
    bq_nl2sql_ms: float | None
    bq_execute_ms: float | None
    sql_source: str | None
    bq_skipped_reason: str | None
    ontology_plan: dict[str, Any] | None
    source_basis: str | None
    rerank_ms: float | None
    rerank_pool_size: int | None
    rerank_mode: str | None
    generate_ms: float | None
    generate_max_tokens: int | None
    generate_input_chars: int | None
    generate_length_capped_calls: int | None
    generate_finish_reason: str | None


def _reasoner_from_state(state: RAGGraphState) -> ReasonerPlan | None:
    raw = state.get("reasoner_plan")
    return ReasonerPlan.from_dict(raw if isinstance(raw, dict) else None)


def _slot_path_active(state: RAGGraphState) -> bool:
    if not state.get("slot_path"):
        return False
    rp = _reasoner_from_state(state)
    return rp is not None and bool(rp.bq_subquestions())


def _early_route_decompose_state(
    raw_q: str,
    route: str,
    explicit_export: str | None = None,
) -> RAGGraphState:
    """Build decompose output for early non-RAG short-circuit on raw user text."""
    answer_lang = detect_answer_language(raw_q)
    export_intent = resolve_export_intent(raw_q, explicit_export)
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
        "query_ir": compile_early_query_ir(raw_q, route).to_dict(),
    }
    return RAGGraphState(**base)


def node_decompose(state: RAGGraphState) -> RAGGraphState:
    raw_q = normalize_query_text((state.get("query") or "").strip())
    
    route = early_non_rag_route(raw_q)
    if route:
        early = _early_route_decompose_state(raw_q, route, state.get("export_intent"))
        with observed_span(
            "decompose",
            input_data={"query": raw_q[:200], "enriched": False, "early_short_circuit": True},
        ):
            update_current_span_metadata(
                {
                    "route_candidate": route,
                    "answer_lang": early.get("answer_lang"),
                    "export_intent": early.get("export_intent"),
                    "early_short_circuit": True,
                    "skipped_decompose_llm": True,
                    "skipped_retrieval": True,
                }
            )
        early["user_query"] = raw_q
        early["context_rewrite"] = ""
        early["detail_rewrite"] = ""
        early["context_applied"] = False
        early["query_views"] = {
            "user_query": raw_q,
            "context_rewrite": "",
            "detail_rewrite": "",
            "context_applied": False,
            "language_enrich": "",
        }
        early["language_enrich"] = ""
        early["language_enrich_applied"] = False
        return early
    active_turn, active_state = resolve_turn(
        raw_q,
        persisted=state.get("active_request") if isinstance(state.get("active_request"), dict) else None,
        recent_turns=state.get("recent_turns") if isinstance(state.get("recent_turns"), list) else None,
    )

    # Detect language and capture confidence value for enrichment decisions
    answer_lang_user, confidence = detect_language_with_confidence(raw_q)

    price_list_products = requested_price_list_products(raw_q)
    language_enrich = ""
    language_enrich_applied = False
    language_enrich_source = "none"
    language_subject_guard = False
    language_subjects: tuple[str, ...] = ()
    if not price_list_products:
        lang_enr = enrich_query_to_english(
            raw_q,
            answer_lang=answer_lang_user,
            confidence=confidence,
        )
        if lang_enr.answer_lang:
            answer_lang_user = lang_enr.answer_lang
        language_enrich = (lang_enr.english_query or "").strip()
        language_enrich_applied = bool(lang_enr.needs_enrich and language_enrich)
        language_enrich_source = lang_enr.enrich_source
        language_subject_guard = lang_enr.subject_guard_applied
        language_subjects = lang_enr.subjects
    views0 = build_query_views(
        raw_q,
        active=active_turn,
        user_profile=state.get("user_profile") if isinstance(state.get("user_profile"), dict) else None,
        plan_type=str(state.get("plan_type") or "") or None,
        category=str(state.get("category") or "") or None,
        language_enrich=language_enrich or None,
    )
    user_q = views0.user_query
    enrich = {
        "user_query": user_q,
        "original_query": user_q,
        "language_enrich": language_enrich,
        "context_rewrite": views0.context_rewrite,
        "context_applied": views0.context_applied,
    }
    # Planning prefers English enrich; carried slots still applied via ActiveRequest.
    resolved_request = resolved_request_text(user_q, active_turn)
    q = language_enrich or resolved_request or user_q
    ctx = DecomposeContext.from_graph_state(state, enrich)
    with observed_span(
        "decompose",
        input_data={
            "query": q[:200],
            "carried": bool(resolved_request),
            "language_enrich_applied": language_enrich_applied,
        },
    ):
        dec_raw = decompose_query(q, context=ctx)
        decompose_llm_ms = float(dec_raw.pop("_decompose_llm_ms", 0.0) or 0.0)
        skipped_decompose_llm = bool(dec_raw.pop("_skipped_decompose_llm", False))
        dec_raw.pop("_decompose_llm_used", None)
        dec_raw.pop("_legacy_would_skip_llm", None)
        dec = dec_raw
        if language_subject_guard:
            # Facets from a rejected subject replacement are not persisted as
            # user intent. Search still retains the complete original wording.
            dec = {**dec, "entities": list(language_subjects), "domains": list(language_subjects),
                   "primary_measure_hints": list(language_subjects), "primary_measures": list(language_subjects),
                   "evidence_topics": [], "fact_requirements": []}
        profile = state.get("user_profile") if isinstance(state.get("user_profile"), dict) else None
        raw_country = str((profile or {}).get("country") or "").strip() or None
        country = sanitize_profile_country(raw_country)
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
        dec = apply_profile_geography_default(dec, country)
        # UI / API scopes must land in decomp so clarify, BQ plans, and retrieval agree.
        dec = _apply_ui_scope_overrides(dec, state)
        if price_list_products:
            dec["entities"] = price_list_products
        add_decomposed_slots(active_turn, dec)
        dec = fill_decomposition(dec, active_turn)
        if price_list_products:
            dec["entities"] = price_list_products
        # Grounded entity/domain enrich → multi-measure retrieval contract tags.
        dec = enrich_decomposition_facets(q, dec)
        matched_bundles = match_intent_bundles(q, dec, breakdown=compile_breakdown(q))
        if matched_bundles:
            dec["matched_bundles"] = [mb.spec.id for mb in matched_bundles]
        contract = build_corpus_routing_contract(q, decomposition=dec)
        if contract.corpus_domain_tags:
            dec["corpus_domain_tags"] = list(contract.corpus_domain_tags)
        bundle_pm = list(bundle_primary_measures(matched_bundles, q) or bundle_required_measures(matched_bundles))
        primary_measures = list(dict.fromkeys([*bundle_pm, *contract.primary_measures]))
        if primary_measures:
            dec["primary_measures"] = primary_measures
        dec = sanitize_decomposition_for_bq(
            dec, primary_measures=dec.get("primary_measures"), query=q
        )
        if contract.companion_measures:
            dec["companion_measures"] = list(contract.companion_measures)
        measure_hints = resolve_measures(q, dec)
        measure_hit = measure_hints[0] if measure_hints else None
        # Re-resolve after gates/overrides may change geography or time.
        task_mode = resolve_task_mode(q, dec, profile_country=country)
        analytical = task_mode == "analytical"
        recency = resolve_recency_tier(q, measure_hit)
        # answer_lang always from original user text — never from English enrich.
        answer_lang = answer_lang_user or detect_answer_language(user_q)
        turn = compile_turn_contract(
            q,
            dec,
            answer_lang=answer_lang,
            measure_hit=measure_hit,
            task_mode_hint=task_mode,
            matched_bundles=matched_bundles,
            original_question=state.get("query") or user_q,
        )
        sources = select_sources(turn, query=q, decomposition=dec, task_mode=task_mode)
        has_bq = sources["bigquery"]["selected"]
        has_documents = sources["qdrant"]["selected"]
        indicator_context = sources["qdrant"]["reason"] == "indicator_context"
        if has_bq:
            if turn.job == "clarify":
                turn.job = "fact"
                turn.serve_status = "served"
                turn.serve_reason = ""
                turn.plan_type = "numeric"
            turn = resolve_capability(turn)
        elif has_documents:
            # The warehouse's missing slots and capability registry do not
            # prevent serving a document question.
            if sources["qdrant"]["reason"] == "document_content_requested":
                turn.job = "brief"
                task_mode = "research"
            elif turn.job == "clarify" or not turn.documents:
                turn.job = "brief"
            turn.serve_status = "served"
            turn.serve_reason = ""
            turn.plan_type = "narrative"
            turn.skip_bq = True
            turn.skip_vector_retrieval = False
        if not (has_bq or has_documents):
            task_mode = "clarify"
        elif task_mode == "clarify":
            if has_bq:
                task_mode = "analytical" if has_documents and not indicator_context else "fact_lookup"
            else:
                task_mode = "research"
        analytical = task_mode == "analytical"
        # Social / meta / product gates stay on the original — never re-run on enrich.
        meta = is_meta_query(user_q)
        help_q = (not meta) and is_help_query(user_q, dec)
        product = (not meta) and (help_q or is_product_query(user_q, dec))
        greeting = (not meta) and (not product) and is_greeting_query(user_q)
        out_of_scope = (
            (not meta)
            and (not product)
            and (not greeting)
            and is_out_of_scope_query(user_q, dec)
        )
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
        export_intent = resolve_export_intent(q, state.get("export_intent"))
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
                "carried_slots": sorted(active_turn.carried()),
                "context_applied": bool(views0.context_applied),
                "language_enrich_applied": language_enrich_applied,
                "language_enrich": language_enrich[:240] if language_enrich else "",
                "decompose_llm_ms": decompose_llm_ms,
                "skipped_decompose_llm": skipped_decompose_llm,
            }
        )
        detail = build_detail_rewrite(q, dec)
        views = build_query_views(
            user_q,
            active=active_turn,
            user_profile=profile if isinstance(profile, dict) else None,
            plan_type=str(state.get("plan_type") or "") or None,
            category=category,
            decomposition=dec,
            language_enrich=language_enrich or None,
        )
        # Prefer freshly built detail from facets
        detail_final = detail or views.detail_rewrite
        out: dict[str, Any] = {
            "decomposition": dec,
            "is_meta_query": meta,
            "is_product_query": product,
            "is_help_query": help_q,
            "is_greeting_query": greeting,
            "is_out_of_scope_query": out_of_scope,
            "is_language_unknown": lang_unknown,
            "answer_lang": answer_lang,
            "language_enrich": language_enrich or None,
            "language_enrich_applied": language_enrich_applied,
            "language_enrich_source": language_enrich_source,
            "language_enrich_subject_guard": language_subject_guard,
            "export_intent": export_intent,
            "task_mode": task_mode,
            "analytical_mode": analytical,
            "query": user_q,
            "user_query": user_q,
            "context_rewrite": views0.context_rewrite,
            "detail_rewrite": detail_final,
            "context_applied": views0.context_applied,
            "query_views": {
                **views.to_dict(),
                "detail_rewrite": detail_final,
                "detail_rewrite_len": len(detail_final or ""),
                "language_enrich": language_enrich,
                "language_enrich_len": len(language_enrich or ""),
            },
            "enriched_query": views0.context_rewrite if views0.context_applied else None,
            "resolved_request": resolved_request,
            "active_request": active_state.to_dict(),
            "measure_id": turn.measure_id or (measure_hit.measure.id if measure_hit else None),
            "recency_tier": recency,
            "turn_contract": turn.to_dict(),
            "route_candidate": route_candidate,
            "early_short_circuit": False,
            "skipped_decompose_llm": skipped_decompose_llm,
            "skipped_retrieval": not (has_bq or has_documents),
            "source_selection": sources,
            "decompose_llm_ms": decompose_llm_ms,
            "metacog_llm_ms": 0.0,
            "metacog_plan": None,
            "metacog_verify": None,
        }
        # Advisory metacog before supervisor/reasoner so diagnose pins flow into job.
        explicit_comparison = is_fact_comparison(turn)
        if (
            metacog_enabled() and route_candidate == "full_rag" and has_bq and has_documents
            and not indicator_context and not explicit_comparison
        ):
            metacog_t0 = time.perf_counter()
            mp = build_metacog_plan(
                q,
                turn=turn,
                decomposition=dec,
                language_enrich=language_enrich,
            )
            turn = apply_metacog_to_turn(turn, mp)
            out["metacog_plan"] = mp.to_dict()
            out["metacog_llm_ms"] = round((time.perf_counter() - metacog_t0) * 1000.0, 2)
            out["turn_contract"] = turn.to_dict()
            dec["metacog_mode"] = mp.advisory_mode
            dec["metacog_evidence_needs"] = list(mp.evidence_needs)
            out["decomposition"] = dec
            update_current_span_metadata(
                {
                    "metacog_mode": mp.advisory_mode,
                    "metacog_confidence": mp.confidence,
                    "metacog_source": mp.source,
                    "metacog_llm_ms": out["metacog_llm_ms"],
                    "turn_job": turn.job,
                }
            )
        sp = compile_supervisor_plan(
            q,
            decomposition=dec,
            matched_bundles=matched_bundles,
            measure_hit=measure_hit,
            primary_measures=dec.get("primary_measures") if isinstance(dec.get("primary_measures"), list) else None,
        )
        out["supervisor_plan"] = sp.to_dict()
        if _slot_reasoner_active() and has_bq and not explicit_comparison:
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
                # Defensive re-grounding: rp.geos can originate from a prior
                # turn's contract (context carried forward) as well as the
                # current decomposition. Reject any country not evidenced in
                # the raw current-turn query text so a stale/unrelated
                # geography can never silently overwrite a correctly grounded
                # one (see Sprint 2 "Geography: France" routing bug).
                grounded_geos = [
                    g
                    for g in rp.geos
                    if facet_grounded_in_query(g, user_q, language_enrich, q)
                ]
                if grounded_geos:
                    dec["geography"] = grounded_geos
            if rp.time_start:
                dec["time_start"] = rp.time_start
            if rp.time_end:
                dec["time_end"] = rp.time_end
            if rp.primary_measure:
                dec["primary_measures"] = list(dict.fromkeys([
                    *(dec.get("primary_measures") or []), rp.primary_measure,
                ]))
            out["decomposition"] = dec
            out["turn_contract"] = turn.to_dict()
            out["measure_id"] = turn.measure_id or rp.primary_measure
        else:
            out["heavy_path"] = False
            out["slot_path"] = False
        final_dec = out.get("decomposition")
        if not isinstance(final_dec, dict):
            final_dec = dec
        final_turn = TurnContract.from_dict(
            out.get("turn_contract") if isinstance(out.get("turn_contract"), dict) else turn.to_dict()
        )
        routing_obj = build_routing_plan(
            turn=final_turn,
            measure_hit=measure_hit,
            supervisor_plan=sp,
            matched_bundles=matched_bundles,
            primary_measures=(
                final_dec.get("primary_measures")
                if isinstance(final_dec.get("primary_measures"), list)
                else None
            ),
        )
        # Metacog evidence_needs can widen corpus_allow without replacing class routing.
        mp_raw = out.get("metacog_plan")
        if isinstance(mp_raw, dict):
            mp_obj = MetacogPlan.from_dict(mp_raw)
            extra = corpora_for_evidence_needs(mp_obj.evidence_needs if mp_obj else ())
            if extra:
                merged_corpus = tuple(
                    dict.fromkeys([*list(routing_obj.corpus_allow), *extra])
                )
                routing_obj = RoutingPlan(
                    primary_measure_id=routing_obj.primary_measure_id,
                    indicator_classes=routing_obj.indicator_classes,
                    matched_bundles=routing_obj.matched_bundles,
                    job=str(final_turn.job or routing_obj.job),
                    corpus_allow=merged_corpus,
                    supervisor_rationale=routing_obj.supervisor_rationale,
                )
        out["routing_plan"] = routing_obj.to_dict()
        reasoner_obj = ReasonerPlan.from_dict(
            out.get("reasoner_plan") if isinstance(out.get("reasoner_plan"), dict) else None
        )
        out["query_ir"] = compile_query_ir(
            query=q,
            decomposition=final_dec,
            turn=final_turn,
            routing=routing_obj,
            supervisor=sp,
            reasoner=reasoner_obj,
            task_mode=str(out.get("task_mode") or task_mode),
            recency_tier=str(out.get("recency_tier") or recency),
            route_candidate=str(out.get("route_candidate") or route_candidate),
            analytical_mode=bool(out.get("analytical_mode")),
        ).to_dict()
        return RAGGraphState(**out)


def _apply_ui_scope_overrides(
    decomposition: dict[str, Any],
    state: RAGGraphState,
) -> dict[str, Any]:
    """Copy screen dates onto the decomposition only when the question names no time.

    A resolved window is left as it is. Screen dates are copied as exact ISO dates.
    """
    out = dict(decomposition or {})
    query = str(state.get("query") or "")
    ref_raw = str(state.get("reference_date") or "").strip()[:10]
    ref = parse_iso_date(ref_raw) or utc_today()
    named = bool(resolve_relative_time(query, ref))
    ts = str(state.get("time_start_override") or "").strip()
    te = str(state.get("time_end_override") or "").strip()
    applied_time = False
    if not named and not str(out.get("time_start") or "").strip() and not str(out.get("time_end") or "").strip():
        end = parse_iso_date(te)
        start = parse_iso_date(ts) or end
        if start is not None:
            out["time_start"] = start.isoformat()
            out["time_end"] = (end or start).isoformat()
            applied_time = True
    if applied_time:
        out["time_source"] = "ui"
    elif str(out.get("time_start") or "").strip() or str(out.get("time_end") or "").strip():
        out.setdefault("time_source", "query")

    geo_ov = str(state.get("geo_override") or "").strip()
    geo = list(out.get("geography") or []) if isinstance(out.get("geography"), list) else []
    if geo_ov and not normalize_geography_for_filter(geo):
        out["geography"] = [geo_ov]
        out["geography_source"] = "ui"
    elif normalize_geography_for_filter(geo):
        out.setdefault("geography_source", "query")
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
        if domains and len(dec.get("evidence_topics") or []) < 2:
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
        if state.get("vector_progress"):
            raise
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
        if state.get("vector_progress"):
            raise
        logger.exception(
            "Retrieval failed for collection %s (%s)",
            default_collection,
            collection_env,
        )
        raw = []
    return [_tag_vector(x, context_kind) for x in raw]


def _retrieve_news(state: RAGGraphState) -> list[dict[str, Any]]:
    dec = state.get("decomposition") or {}
    texts = _document_search_texts(state)
    if not texts:
        fallback = (state.get("query") or "").strip()
        texts = [fallback] if fallback else []
    countries = resolve_retrieval_geographies(
        geo_override=str(state.get("geo_override") or ""),
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )
    ts = str(dec.get("time_start") or "").strip()[:10]
    te = str(dec.get("time_end") or "").strip()[:10]
    raw_turn = state.get("turn_contract")
    contract = TurnContract.from_dict(raw_turn if isinstance(raw_turn, dict) else None)
    if _document_time_is_observation(state, contract):
        ts, te = "", ""
    top_k = int(state.get("news_top_k") or _DEFAULT_NEWS_TOP_K)
    news_coll = os.environ.get("QDRANT_COLLECTION_NEWS", "news_data").strip() or "news_data"
    vr = VectorRetriever(collection_name=news_coll)
    strict_compare = _strict_compare_filters(dec)
    allow_geo_fb = not strict_compare
    progress = state.get("vector_progress")
    completed: list[list[dict[str, Any]]] = []

    def _cascade_for_text(
        text: str,
        target_countries: list[str],
        *,
        allow_geo_fb: bool,
        base_kw: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        kw = dict(base_kw) if base_kw is not None else _news_kwargs(state, dec)
        if len(target_countries) == 1:
            kw["geo_country"] = target_countries[0]
        elif target_countries:
            _apply_geo_to_kwargs(kw, target_countries)
        raw = _retrieve_vector_cascade(
            vr,
            text,
            base_kwargs=kw,
            countries=target_countries if len(target_countries) <= 1 else target_countries,
            has_time=bool(ts or te),
            geo_fallback_env="RAG_NEWS_GEO_FALLBACK",
            time_fallback_env="RAG_NEWS_TIME_FALLBACK",
            allow_geo_fallback=allow_geo_fb,
            deadline=progress.deadline if progress else None,
        )
        if progress and raw:
            completed.append(_post_filter_geography(raw, target_countries or countries))
            progress.hits = _merge_vector_context_dicts(completed, limit=top_k)
        return raw

    def _run_for_countries(target_countries: list[str], *, allow_geo_fb: bool) -> list[dict[str, Any]]:
        if len(target_countries) >= 2 and strict_compare:
            per_k = max(3, (top_k + len(target_countries) - 1) // len(target_countries))
            batches: list[list[dict[str, Any]]] = []
            for country in target_countries:
                per_text_batches: list[list[dict[str, Any]]] = []
                for text in texts:
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
                    per_text_batches.append(
                        _cascade_for_text(text, [country], allow_geo_fb=allow_geo_fb, base_kw=kw)
                    )
                batches.append(_merge_vector_context_dicts(per_text_batches, limit=per_k * 2))
            merged = _merge_dedupe_vector_hits(batches, top_k)
            return _post_filter_geography(merged, target_countries)

        per_text: list[list[dict[str, Any]]] = []
        for text in texts:
            kw = _news_kwargs(state, dec)
            _apply_geo_to_kwargs(kw, target_countries)
            kw["time_filter_at_qdrant"] = _news_time_filter_at_qdrant()
            if not kw["time_filter_at_qdrant"]:
                kw["overfetch_multiplier"] = 12
            if ts:
                kw["published_at_from"] = ts
            if te:
                kw["published_at_to"] = te
            per_text.append(
                _cascade_for_text(text, target_countries, allow_geo_fb=allow_geo_fb, base_kw=kw)
            )
        raw = _merge_vector_context_dicts(per_text, limit=max(top_k * 2, 20))
        return _post_filter_geography(raw, target_countries)

    raw = _run_for_countries(countries, allow_geo_fb=allow_geo_fb)

    # Compare: keep geo strict at Qdrant but allow semantic recall + post-filter by country name in text.
    if not raw and countries and strict_compare and _env_on("RAG_NEWS_COMPARE_SEMANTIC_FALLBACK", default=True):
        per_text: list[list[dict[str, Any]]] = []
        for text in texts:
            kw = _news_kwargs(state, dec)
            kw["time_filter_at_qdrant"] = False
            kw["overfetch_multiplier"] = 20
            if ts:
                kw["published_at_from"] = ts
            if te:
                kw["published_at_to"] = te
            per_text.append(
                _cascade_for_text(text, [], allow_geo_fb=False, base_kw=kw)
            )
        semantic = _merge_vector_context_dicts(per_text, limit=max(top_k * 2, 20))
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
    def _build(st: RAGGraphState, dec: dict[str, Any]) -> dict[str, Any]:
        kinds = formation_doc_kinds(str(st.get("query") or ""))
        kwargs = _dense_corpus_kwargs(st, dec, doc_kind="agricultural_practise")
        if len(kinds) > 1:
            kwargs["doc_kinds"] = kinds
            kwargs.pop("doc_kind", None)
        return kwargs

    try:
        raw = _vector_retrieve_for_corpus(
            state,
            collection_env="QDRANT_COLLECTION_FORMATION",
            default_collection="formation",
            build_kwargs=_build,
            geo_fallback_env="RAG_RESEARCH_GEO_FALLBACK",
            time_fallback_env="RAG_RESEARCH_TIME_FALLBACK",
        )
    except Exception:
        if state.get("vector_progress"):
            raise
        logger.exception("Retrieval failed for collection formation (QDRANT_COLLECTION_FORMATION)")
        raw = []
    return [_tag_vector(x, "formation") for x in raw]


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
        if vr.last_index_health is not None and vr.last_index_health.queryable is False:
            raise RuntimeError("Qdrant index unavailable or incompatible")
    except Exception:
        if state.get("vector_progress"):
            raise
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


_VECTOR_WORKERS = RetrievalWorkers("rag-vector", workers=12)
_WAREHOUSE_WORKERS = RetrievalWorkers("rag-warehouse", workers=4)
_WAREHOUSE_LEGS = RetrievalWorkers("rag-warehouse-leg", workers=4)
_RERANK_WORKERS = RetrievalWorkers("rag-rerank", workers=4)
_GENERATION_RESERVE_S = 10.0


def node_parallel_retrieve(state: RAGGraphState) -> RAGGraphState:
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
    sources = state.get("source_selection")
    if (
        (sources is not None and not sources["qdrant"]["selected"])
        or (
            sources is None and not contract.should_retrieve_vector() and not contract.documents
            and not (sp and sp.must_search_qdrant)
        )
    ):
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
        vector_allow=contract.vector_allow if contract.measure_id else None,
        vector_block=contract.vector_block if contract.measure_id else None,
        vector_policy=contract.vector_policy if contract.measure_id else None,
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
            list(planner.get("active_corpora") or selection.active),
            boosts=dict(selection.boosts),
            rationale=f"{selection.rationale}|reasoner_planner",
        )
        planner_meta = {
            "hard_time_filter": bool(planner.get("hard_time_filter")),
            "news_allowed": bool(planner.get("news_allowed", True)),
            "reasoner_planner": True,
        }
    document_lookup = sources is not None and sources["qdrant"]["reason"] == "document_content_requested"
    if not document_lookup and sp and sp.classes and sp.must_search_qdrant:
        class_corpora = corpora_for_supervisor_plan(sp)
        if class_corpora:
            filtered = [c for c in class_corpora if c in CORPUS_CATALOG]
            if filtered:
                if contract.documents or len(decomposition.get("primary_measures") or []) > 1:
                    filtered = list(dict.fromkeys([*filtered, *selection.active]))
                selection = CorpusSelection(
                    filtered,
                    dict(selection.boosts),
                    f"{selection.rationale}|class_supervisor",
                )
    mp_raw = state.get("metacog_plan")
    if isinstance(mp_raw, dict):
        mp_obj = MetacogPlan.from_dict(mp_raw)
        extra = [
            c
            for c in corpora_for_evidence_needs(mp_obj.evidence_needs if mp_obj else ())
            if c in CORPUS_CATALOG
        ]
        if extra:
            selection = CorpusSelection(
                active=list(dict.fromkeys([*selection.active, *extra])),
                boosts=selection.boosts,
                rationale=f"{selection.rationale}|metacog_evidence",
            )
    scoped_comparison = bool(contract.documents) and is_fact_comparison(contract)
    if scoped_comparison:
        selection = CorpusSelection(
            comparison_document_corpora([doc.question for doc in contract.documents]),
            boosts=dict(selection.boosts),
            rationale=f"{selection.rationale}|explicit_document_requirements",
        )
    indicator_context = sources is not None and sources["qdrant"]["reason"] == "indicator_context"
    if indicator_context:
        selection = CorpusSelection(
            ["ota", "public_reports", "news"], boosts=selection.boosts, rationale="indicator_context",
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

    state_for_retrieve = state.copy()
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
    progress_by_corpus: dict[str, _VectorProgress] = {}
    statuses: dict[str, str] = {}
    corpus_timeout = min(
        stage_budget_seconds("RAG_CORPUS_RETRIEVE_TIMEOUT_S", default_s=4.0),
        stage_budget_seconds("RAG_STAGE_RETRIEVE_BUDGET_S", default_s=12.0),
    )
    vector_t0 = time.perf_counter()
    deadline = vector_t0 + corpus_timeout
    retrievers = {key: fn for key, fn in retrievers.items() if key in active}
    if not (indicator_context or scoped_comparison) and _use_legacy_research_collection():
        retrievers["legacy_research"] = _retrieve_legacy_research
    for key, fn in retrievers.items():
        if corpus_timeout == 0:
            statuses[key] = "timeout"
            corpus_errors.append(f"{key}:timeout")
            continue
        progress = _VectorProgress(deadline)
        progress_by_corpus[key] = progress
        corpus_state = state_for_retrieve.copy()
        corpus_state["vector_progress"] = progress
        with request_budget(max(0.0, deadline - time.perf_counter())):
            future = _VECTOR_WORKERS.try_submit(run_with_tracing_context(fn, corpus_state))
        if future is None:
            corpus_errors.append(f"{key}:busy")
            statuses[key] = "busy"
        else:
            jobs[future] = key

    remaining = max(0.0, corpus_timeout - (time.perf_counter() - vector_t0))
    done, pending = wait(jobs, timeout=remaining)
    for future in pending:
        future.cancel()
        kind = jobs[future]
        statuses[kind] = "timeout"
        corpus_errors.append(f"{kind}:timeout")
    for future in done:
        kind = jobs[future]
        try:
            result = future.result()
        except FuturesTimeoutError:
            statuses[kind] = "timeout"
            corpus_errors.append(f"{kind}:timeout")
            continue
        except Exception as exc:
            logger.exception("Parallel retrieval failed for %s", kind)
            statuses[kind] = "error"
            corpus_errors.append(f"{kind}:{type(exc).__name__}")
            continue
        buckets[kind] = result if isinstance(result, list) else []
        statuses[kind] = "hits" if buckets[kind] else "empty"

    from ml.rag.chatbot.corpus_catalog import context_kind_for_corpus_key

    for key, status in statuses.items():
        if status in ("timeout", "error"):
            context_kind = "research" if key == "legacy_research" else context_kind_for_corpus_key(key)
            buckets[key] = [_tag_vector(hit, context_kind) for hit in progress_by_corpus[key].hits]
    corpus_results = {
        key: {"status": statuses[key], "hits": len(buckets[key])}
        for key in retrievers
    }

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
        "corpus_results": corpus_results,
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
    from ml.rag.chatbot.query_views import user_query_dropped as _uq_dropped

    texts_used = _document_search_texts(state_for_retrieve)
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
        "corpus_selection": {**selection.to_dict(), **planner_meta, "results": corpus_results},
        "vector_ms": meta_update["vector_ms"],
        "corpus_count": len(selection.active),
        "cascade_level": 1 if task_mode in ("fact_lookup", "data_export_only") else None,
        "vector_texts_used": texts_used,
        "user_query_dropped": _uq_dropped(state_for_retrieve.get("query_views"), texts_used),
        "vector_cache_hit": any(progress.cache_hit for progress in progress_by_corpus.values()),
        "retrieval_partial": bool(corpus_errors),
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
        "plan_source": "compile_error",
    }


def _build_bq_sql_plan(
    *,
    q: str,
    dec: dict[str, Any],
    sp: SupervisorPlan | None,
    state: RAGGraphState,
    analytical: bool,
    task_mode: str,
) -> dict[str, Any]:
    """Single BQ plan builder.

    Explicit comparisons use the bound requirements before other planning paths.
    Priority for other request shapes:
      1. slot reasoner (when active + bq_subquestions)
      2. class engines → bind_contracts (planned path; no engine SELECT strings)
      3. analytical / export → reason_bq_sql_plan escape
      4. compile_error block (default when classes empty and compiler preference on)
      5. legacy reason_bq_sql_plan only when compiler preference off
    """
    raw = state.get("turn_contract")
    turn = TurnContract.from_dict(raw if isinstance(raw, dict) else None)
    sources = state.get("source_selection")
    if sources is not None and turn.facts:
        selected = {
            item["requirement"] for item in sources["bigquery"]["requirements"] if item["selected"]
        }
        turn = replace(turn, facts=[fact for fact in turn.facts if fact.key in selected])
    requirement_plan = plan_fact_requirements(turn, query=q, facets=dec)
    if requirement_plan is not None:
        return normalize_bq_plan(requirement_plan, plan_source="turn_requirements")

    rp_raw = state.get("reasoner_plan")
    reasoner = ReasonerPlan.from_dict(rp_raw if isinstance(rp_raw, dict) else None)
    if reasoner and reasoner.bq_subquestions() and _slot_reasoner_active():
        return normalize_bq_plan(reasoner_plan_to_bq_plan(reasoner), plan_source="slot_reasoner")

    if sp and sp.classes:
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
        return normalize_bq_plan(plan, plan_source="class_engine")

    if analytical or task_mode == "analytical":
        plan = reason_bq_sql_plan(
            q,
            decomposition=dec,
            plan_type=str(state.get("plan_type") or "") or None,
            category=str(state.get("category") or "") or None,
            analytical_mode=analytical,
            task_mode=task_mode,
        )
        return normalize_bq_plan(plan, plan_source="analytical")

    if sql_compiler_enabled():
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
        return normalize_bq_plan(plan, plan_source="compile_error")

    plan = reason_bq_sql_plan(
        q,
        decomposition=dec,
        plan_type=str(state.get("plan_type") or "") or None,
        category=str(state.get("category") or "") or None,
        analytical_mode=analytical,
        task_mode=task_mode,
    )
    return normalize_bq_plan(plan, plan_source="retrieval_contract")


def _bq_planning_query(state: RAGGraphState) -> str:
    """Keep explicit price lists intact, otherwise prefer the enriched query."""
    original = str(state.get("user_query") or state.get("query") or "").strip()
    if requested_price_list_products(original):
        return original
    lang_en = str(state.get("language_enrich") or "").strip()
    if lang_en:
        return lang_en
    resolved = str(state.get("resolved_request") or "").strip()
    return resolved or original


def _fp_equal(a: Any, b: Any) -> bool:
    return str(a or "").strip() == str(b or "").strip()


def node_bq_reason(state: RAGGraphState) -> RAGGraphState:
    """YAML-index SQL reasoner or class engines: select mart_dev tables and query intents."""
    q = _bq_planning_query(state)
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

    plan = _build_bq_sql_plan(
        q=q,
        dec=dec,
        sp=sp,
        state=state,
        analytical=analytical,
        task_mode=task_mode,
    ).copy()
    tc_raw = state.get("turn_contract")
    contract = TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None)
    if (
        contract.skip_bq
        and plan.get("plan_source") != "turn_requirements"
        and not plan.get("heavy_path")
        and not plan.get("slot_path")
        and not plan.get("bq_sql_queries")
    ):
        plan["skip_bq"] = True
    template_key = str((contract.sql_plan or {}).get("template") or "").strip()
    if template_key:
        plan["template_key"] = template_key
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
    out: RAGGraphState = {
        "bq_sql_plan": plan,
        "bq_table_candidates_inspector": cands,
        "value_hits": plan.get("value_hits"),
    }
    if plan.get("plan_source") == "turn_requirements" and plan.get("bind_contracts"):
        # A capability miss for one indicator cannot erase proven sibling binds.
        # Individual gaps remain in requirement coverage.
        contract = replace(
            contract, serve_status="served", serve_reason="", skip_bq=False,
            contract_sql_only=True, sql_plan={},
        )
        out["turn_contract"] = contract.to_dict()
    return out


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
        for key in ("job_id", "bq_ms", "bytes_processed", "row_count", "class_code", "table_id", "value_hits", "requirement_id"):
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


def node_bq_retrieve(state: RAGGraphState) -> RAGGraphState:
    from ml.rag.chatbot.bq_fact_cache import (
        bq_results_from_fact_entry,
        fact_cache_key,
        fact_entry_from_bq_results,
        fingerprint_from_decomposition,
        get_cached_facts,
        set_cached_facts,
        should_reuse_facts,
    )
    from ml.rag.chatbot.ontology_routing import (
        bq_should_run,
        bq_value_filters,
        ontology_plan_for_state,
    )

    q = _bq_planning_query(state)
    raw_dec = state.get("decomposition")
    dec: dict[str, Any] = raw_dec if isinstance(raw_dec, dict) else {}
    raw_plan = state.get("bq_sql_plan")
    plan: dict[str, Any] = raw_plan if isinstance(raw_plan, dict) else {}
    requirement_path = plan.get("plan_source") == "turn_requirements"
    tc_raw = state.get("turn_contract")
    contract = TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None)
    if plan.get("skip_bq"):
        return {
            "bq_results": [],
            "bq_sql_queries": [],
            "bq_sql_debug": [],
            "sql_source": "skipped",
            "bq_execute_ms": 0.0,
            "bq_nl2sql_ms": 0.0,
        }

    onto_plan = ontology_plan_for_state(state)
    if (
        os.environ.get("ADZA_ONTOLOGY_SKIP_BQ", "0").strip().lower()
        in ("1", "true", "on", "yes")
        and not requirement_path and not bq_should_run(onto_plan)
    ):
        # Intentional skip for qdrant_only subjects — do not mark warehouse empty.
        return {
            "bq_results": [],
            "bq_sql_queries": [],
            "bq_sql_debug": [],
            "sql_source": "skipped",
            "bq_skipped_reason": "ontology_no_value_map",
            "bq_execute_ms": 0.0,
            "bq_nl2sql_ms": 0.0,
        }

    cached = state.get("structured_ranking_cache")
    if not requirement_path and isinstance(cached, dict) and is_ranking_follow_up(q, dec, cached):
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

    tables = [str(t).split(".")[-1] for t in (plan.get("selected_tables") or []) if str(t).strip()]
    measure = str(state.get("measure_id") or "")
    if not measure and isinstance(dec.get("primary_measures"), list) and dec["primary_measures"]:
        measure = str(dec["primary_measures"][0])
    fp = fingerprint_from_decomposition(
        dec,
        tables=tables,
        measure=measure,
        shape=str(dec.get("job") or ""),
    )
    facets = {
        "measure": measure,
        "geos": list(dec.get("geography") or []),
        "time_start": dec.get("time_start"),
        "time_end": dec.get("time_end"),
        "tables": tables,
    }
    session_id = str(state.get("session_id") or "").strip() or None
    fact_key = fact_cache_key(session_id, fp)
    fact_cached = get_cached_facts(fact_key)
    if fact_cached is None and isinstance(state.get("last_bq_facts"), list):
        for entry in state.get("last_bq_facts") or []:
            if isinstance(entry, dict) and _fp_equal(entry.get("fingerprint"), fp):
                fact_cached = entry
                break
    if not requirement_path and should_reuse_facts(
        context_applied=bool(state.get("context_applied")),
        current_facets=facets,
        cached_entry=fact_cached,
    ):
        results = bq_results_from_fact_entry(fact_cached)
        if results:
            bq_sql_queries, bq_sql_debug = aggregate_bq_sql_debug(results)
            update_current_span_metadata({"bq_cache_hit": True, "bq_fact_cache": True})
            return {
                "bq_results": results,
                "bq_sql_queries": bq_sql_queries,
                "bq_sql_debug": bq_sql_debug,
                "bq_cache_hit": True,
                "sql_source": "fact_cache",
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
    ts = str(dec.get("time_start") or "").strip()[:10]
    te = str(dec.get("time_end") or "").strip()[:10]
    if requirement_path:
        countries = contract.geo
        ts = min(window.start for window in contract.time_spec.windows).isoformat()
        te = max(window.end for window in contract.time_spec.windows).isoformat()
        dec = {**dec, "geography": countries, "time_start": ts, "time_end": te}
    entities = dec.get("entities") if isinstance(dec.get("entities"), list) else None
    domains = dec.get("domains") if isinstance(dec.get("domains"), list) else None
    retriever = BQRetriever()
    bq_geo: dict[str, Any] = {}
    if len(countries) >= 2:
        bq_geo["geo_countries"] = countries
    elif len(countries) == 1:
        bq_geo["geo_country"] = countries[0]

    max_sql_queries: int | None = None
    if requirement_path:
        max_sql_queries = int(plan["max_sql_queries"])
    elif bool(state.get("analytical_mode")) or plan.get("analytical_mode"):
        from ml.rag.chatbot.analytical_bq_plan import analytical_sql_query_floor

        max_sql_queries = int(plan.get("max_sql_queries") or analytical_sql_query_floor())
    task_mode = str(state.get("task_mode") or "chat").strip().lower()
    pre_debug = list(plan.get("bq_sql_debug") or [])
    pre_queries = list(plan.get("bq_sql_queries") or [])
    engine_execute_only = plan.get("sql_source") == "engine" and bool(pre_queries)
    if engine_execute_only:
        intents = []
    bq_timeout = stage_budget_seconds("RAG_BQ_RETRIEVE_TIMEOUT_S", default_s=25.0)
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
    job_deadline = retrieve_deadline - min(0.20, bq_timeout * 0.15)
    retrieve_kwargs: dict[str, Any] = {
        "crop_required": bool(plan.get("crop_required", True)),
        "geography_required": bool(plan.get("geography_required", True)),
        "decomposition": dec,
        "contract_job": contract.job,
        "breakdown": list(contract.breakdown),
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
    if max_sql_queries is not None:
        retrieve_kwargs["max_sql_queries"] = max_sql_queries
    plan_source = str(plan.get("plan_source") or "").strip()
    if plan_source:
        retrieve_kwargs["plan_source"] = plan_source
    if requirement_path:
        contract_kwargs.update(contract_sql_only=True, serve_status="served", template_key="")
    retrieve_mode = str(plan.get("retrieve_mode") or "").strip()
    if retrieve_mode:
        retrieve_kwargs["retrieve_mode"] = retrieve_mode
    bind_raw = plan.get("bind_contracts")
    if isinstance(bind_raw, dict) and bind_raw:
        retrieve_kwargs["bind_contracts"] = bind_raw
    if contract_kwargs.get("template_key") and isinstance(bind_raw, dict) and bind_raw:
        contract_kwargs.pop("template_key", None)
    if os.environ.get("ADZA_ONTOLOGY_BQ_VALUES", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    ):
        ovf = bq_value_filters(onto_plan)
        if ovf:
            retrieve_kwargs["ontology_value_filters"] = ovf
    future = None
    try:
        future = _WAREHOUSE_WORKERS.try_submit(
            run_with_tracing_context(
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
                deadline=job_deadline,
                **bq_geo,
                **contract_kwargs,
                **retrieve_kwargs,
            )
        )
        if future is None:
            raise FuturesTimeoutError("Warehouse retrieval workers are busy")
        results = future.result(timeout=max(0.0, retrieve_deadline - time.perf_counter()))
    except FuturesTimeoutError as exc:
        if future is not None:
            future.cancel()
        update_current_span_metadata({"bq_timeout": True, "status": "timeout"})
        node_timed_out = True
        results = []
        if not engine_execute_only:
            bq_failure_row = bq_failure_debug_row(
                retriever,
                status="bq_timeout",
                prep_error=str(exc) or f"BQ retrieve exceeded {bq_timeout:g}s timeout",
            )
    except Exception as exc:
        logger.exception("BQ retrieve failed")
        results = []
        bq_failure_row = bq_failure_debug_row(
            retriever,
            status="bq_error",
            prep_error=str(exc)[:500],
        )
    bq_sql_queries, bq_sql_debug = aggregate_bq_sql_debug(results)
    results = enrich_bq_results(
        results,
        query=q,
        plan=plan,
        decomposition=dec,
    )
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
    usable_bq_count = sum(1 for row in results if is_usable_structured_bq_row(row))
    bq_sql_debug = ensure_planned_attempt_debug(
        plan,
        bq_sql_debug,
        usable_bq=usable_bq_count > 0,
    )
    query_planned = plan_indicates_warehouse_attempt(plan)
    planned_sql = plan.get("bq_sql_queries")
    planned_sql_count = len(planned_sql) if isinstance(planned_sql, list) else 0
    execution_status_counts: dict[str, int] = {}
    for row in bq_sql_debug:
        status = str(row.get("status") or "unknown")
        execution_status_counts[status] = execution_status_counts.get(status, 0) + 1
    job_ids = {str(row.get("job_id")) for row in bq_sql_debug if row.get("job_id")}
    execution_flags = bq_execute_flags(
        bq_sql_debug,
        pre_queries=collect_pre_queries(
            plan,
            state_queries=bq_sql_queries,
            bq_sql_debug=bq_sql_debug,
        ),
        usable_bq=usable_bq_count > 0,
        compile_error=bool(plan.get("compile_error")),
        warehouse_attempted=query_planned or bool(bq_sql_queries or bq_sql_debug),
    )
    bq_diagnostics = {
        "query_planned": query_planned,
        "planned_sql_count": planned_sql_count,
        "retrieved_sql_count": len(bq_sql_queries),
        "bq_debug_count": len(bq_sql_debug),
        "job_id_count": len(job_ids),
        "execution_status_counts": execution_status_counts,
        "result_count": len(results),
        "usable_result_count": usable_bq_count,
        "execution_flags": execution_flags,
    }
    if query_planned and not job_ids and usable_bq_count == 0:
        logger.warning("BigQuery retrieval diagnostics: %s", bq_diagnostics)
    else:
        logger.info("BigQuery retrieval diagnostics: %s", bq_diagnostics)
    update_current_span_metadata(
        {
            "bq_query_planned": query_planned,
            "bq_planned_sql_count": planned_sql_count,
            "bq_retrieved_sql_count": len(bq_sql_queries),
            "bq_debug_count": len(bq_sql_debug),
            "bq_job_id_count": len(job_ids),
            "bq_execution_status_counts": execution_status_counts,
            "bq_planned_sql": sql_statements_for_diagnostics(planned_sql),
            "bq_sql_attempts": sql_attempts_for_diagnostics(bq_sql_debug),
            **execution_flags,
        }
    )
    cache_entry = cache_entry_from_bq_results(results, query=q, decomposition=dec)
    sql_source = getattr(retriever, "last_sql_source", None)
    if not sql_source:
        for row in bq_sql_debug:
            src = row.get("sql_source")
            if src:
                sql_source = str(src)
                break
    fact_entry = fact_entry_from_bq_results(
        results, fingerprint=fp, facets=facets, query=q
    )
    if fact_entry:
        set_cached_facts(fact_key, fact_entry)
    out: dict[str, Any] = {
        "bq_results": results,
        "execution_results": [r.to_dict() for r in execution_results_from_items(results)],
        "bq_sql_queries": bq_sql_queries,
        "bq_sql_debug": bq_sql_debug,
        "bq_cache_hit": False,
        "sql_source": sql_source,
        "bq_execute_ms": getattr(retriever, "last_bq_execute_ms", None),
        "bq_nl2sql_ms": getattr(retriever, "last_bq_nl2sql_ms", None),
    }
    if cache_entry:
        out["structured_ranking_cache"] = cache_entry
    if fact_entry:
        prior = list(state.get("last_bq_facts") or []) if isinstance(state.get("last_bq_facts"), list) else []
        prior = [e for e in prior if isinstance(e, dict) and not _fp_equal(e.get("fingerprint"), fp)]
        prior.insert(0, fact_entry)
        out["last_bq_facts"] = prior[:8]
    return RAGGraphState(**out)


def node_merge(state: RAGGraphState) -> RAGGraphState:
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
            return RAGGraphState(
                merged_context=other_merged,
                passages=other_merged,
                rows_by_subquestion=serializable,
            )

        return RAGGraphState(merged_context=merged)


def _pack_fields(pack: PackResult | list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if isinstance(pack, PackResult):
        return pack.items, dict(pack.dropped)
    return list(pack or []), {}


def _with_place_drops(dropped: dict[str, int], place_dropped: int) -> dict[str, int]:
    out = dict(dropped)
    if place_dropped:
        out["place_conflict"] = place_dropped
    return out


def node_rerank(state: RAGGraphState) -> RAGGraphState:
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

    dec = state.get("decomposition") if isinstance(state.get("decomposition"), dict) else None
    before_place = len(merged)
    merged = drop_geo_conflicting(list(merged), dec)
    merged = drop_price_off_topic(merged, str(query), dec)
    place_dropped = before_place - len(merged)
    tc_raw = state.get("turn_contract")
    contract = TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None)

    corpus_sel = state.get("corpus_selection")
    active_corpora = 6
    if isinstance(corpus_sel, dict):
        active_raw = corpus_sel.get("active")
        if isinstance(active_raw, list):
            active_corpora = len(active_raw)
    structured = [item for item in merged if is_warehouse_item(item)]
    bq_plan = state.get("bq_sql_plan")
    if isinstance(bq_plan, dict) and bq_plan.get("plan_source") == "turn_requirements":
        structured = prioritize_bound_facts(structured, contract, bq_plan)
    documents = [item for item in merged if not is_warehouse_item(item)]
    scoped_documents = _has_explicit_document_requirements(contract) or any(f.enumerate_members for f in contract.facts)
    document_scope_dropped = 0
    if scoped_documents:
        admitted = requirement_document_candidates(contract, documents)
        document_scope_dropped = len(documents) - len(admitted)
        documents = admitted
    if not scoped_documents and len(documents) <= 8 and active_corpora <= 1:
        packed = diversify_context_pack(list(documents), top_k=None, task_mode=task_mode)
        pack = typed_context_pack(
            [*structured, *packed], contract, top_k=top_k if top_k > 0 else None
        )
        items, dropped = _pack_fields(pack)
        return {
            "reranked_context": items,
            "evidence_drops": _with_place_drops(dropped, place_dropped),
            "rerank_mode": "skipped_trivial",
            "rerank_ms": trace_elapsed_ms(rerank_t0),
            "rerank_pool_size": 0,
            "retrieval_stage_counts": {
                "candidate": len(merged),
                "structured": len(structured),
                "reranked": 0,
                "packed": len(items),
            },
        }

    try:
        pool = int(
            os.environ.get("RAG_RERANK_POOL_SIZE", str(default_pool))
            or default_pool
        )
    except ValueError:
        pool = default_pool
    score_k = max(top_k, min(pool, max(len(merged), top_k)))
    numeric_query = is_numeric_data_query(str(query), dec)
    comparative_query = is_comparative_bq_query(str(query), dec)
    def _rank() -> tuple[list[dict[str, Any]], str]:
        ranked = rerank(
            ("\n".join(doc.question for doc in contract.documents) or query) if scoped_documents else query,
            documents,
            top_k=score_k,
            numeric_query=numeric_query,
            comparative_query=comparative_query,
        )
        return ranked, last_rerank_mode()

    scored, mode = documents[:score_k], "skipped_budget" if documents else "skipped_no_documents"
    with request_budget(reserve_s=_GENERATION_RESERVE_S):
        if documents and remaining_seconds() != 0:
            future = _RERANK_WORKERS.try_submit(run_with_tracing_context(_rank))
            if future is not None:
                try:
                    scored, mode = future.result(timeout=remaining_seconds())
                except FuturesTimeoutError:
                    future.cancel()
                except Exception:
                    mode = "skipped_error"
                    logger.exception("Document rerank failed")
    packed = diversify_context_pack(scored, top_k=None, task_mode=task_mode)
    pack = typed_context_pack(
        [*structured, *packed], contract, top_k=top_k if top_k > 0 else None
    )
    items, dropped = _pack_fields(pack)
    if document_scope_dropped:
        dropped["document_scope"] = document_scope_dropped
    return {
        "reranked_context": items,
        "evidence_drops": _with_place_drops(dropped, place_dropped),
        "rerank_mode": mode,
        "rerank_ms": trace_elapsed_ms(rerank_t0),
        "rerank_pool_size": pool,
        "retrieval_stage_counts": {
            "candidate": len(merged),
            "structured": len(structured),
            "reranked": len(scored),
            "packed": len(items),
        },
    }



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


def node_web_fallback(state: RAGGraphState) -> RAGGraphState:
    """Append Wikipedia / Tavily chunks when internal retrieval is weak.

    The turn routes to the gap answer only when no warehouse row or document
    survived, internal or web.
    """
    with observed_span("web_fallback"):
        reranked = list(state.get("reranked_context") or [])
        bq_results = state.get("bq_results") or []
        has_bq = any(
            is_usable_structured_bq_row(r) for r in bq_results if isinstance(r, dict)
        )
        merged: list[dict[str, Any]] = []
        if not needs_web_fallback(
            reranked,
            task_mode=str(state.get("task_mode") or ""),
            has_usable_bq=has_bq,
            bq_warehouse_attempted=warehouse_blocks_web(dict(state)),
            heavy_path=bool(state.get("heavy_path")),
        ):
            update_current_span_metadata({"web_fallback_status": "skipped"})
            out = {"merged_context": merged}
            return RAGGraphState(**out)

        _pt = str(state.get("plan_type") or "").strip()
        if _pt.lower() == "free":
            update_current_span_metadata({"web_fallback_status": "disabled"})
            out = {"web_fallback_status": "disabled", "web_fallback_reason": "Free plan - web fallback not available"}
            return RAGGraphState(**out)

        # Prefer English enrich for wiki/Tavily; fall back to original user text.
        q = (
            str(state.get("language_enrich") or "").strip()
            or (state.get("query") or "").strip()
        )
        dec = state.get("decomposition") if isinstance(state.get("decomposition"), dict) else None
        ts = str((dec or {}).get("time_start") or "").strip()[:10]
        te = str((dec or {}).get("time_end") or "").strip()[:10]
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
            if decide_outcome(filter_context_items(reranked)).is_gap:
                out["insufficient_context"] = True
            update_current_span_metadata(
                {
                    "web_fallback_status": out["web_fallback_status"],
                    "insufficient_context": bool(out.get("insufficient_context")),
                }
            )
            return RAGGraphState(**out)

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
            return RAGGraphState(**out)

        if decide_outcome(filter_context_items(reranked)).is_gap:
            logger.info(
                "Web fallback status=%s and no rows or documents — routing to insufficient_context. reason=%s",
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
        return RAGGraphState(**out)


def node_insufficient_context(state: RAGGraphState) -> RAGGraphState:
    """Name the gap: no warehouse row and no document survived retrieval."""
    query = str(state.get("query") or "")
    answer_lang = str(state.get("answer_lang") or detect_answer_language(query))
    dec = state.get("decomposition")
    tc_raw = state.get("turn_contract")
    warehouse_failed = any(state.get(flag) for flag in (
        "structured_bq_validation_failed", "structured_bq_compile_error",
        "structured_bq_never_executed", "structured_bq_timed_out", "structured_bq_execution_failed",
    ))
    documents_failed = bool(state.get("retrieval_partial"))
    incomplete: Literal["warehouse", "documents", "both"] | None = (
        "both" if warehouse_failed and documents_failed else
        "warehouse" if warehouse_failed else "documents" if documents_failed else None
    )
    answer = gap_answer(
        query,
        dec if isinstance(dec, dict) else None,
        TurnContract.from_dict(tc_raw if isinstance(tc_raw, dict) else None),
        incomplete_source=incomplete,
    )
    with observed_span(
        "insufficient_context",
        input_data={"web_fallback_status": str(state.get("web_fallback_status") or "")[:80]},
    ):
        update_current_span_metadata(
            {
                "web_fallback_status": str(state.get("web_fallback_status") or ""),
                "answer_lang": answer_lang,
                "warehouse_attempted": warehouse_was_attempted(dict(state)),
            }
        )
    return {
        "answer": answer,
        "citations": [],
        "insufficient_context": True,
        "answer_outcome": "gap",
        "answer_lang": answer_lang,
        **acf_result_to_state(no_evidence_acf()),
    }


def _route_after_web_fallback(state: RAGGraphState) -> str:
    """Route post-web_fallback: either explicit 'insufficient' branch or normal generate."""
    if state.get("insufficient_context"):
        return "insufficient_context"
    return "generate"


def node_generate_language_help(state: RAGGraphState) -> RAGGraphState:
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


def node_generate(state: RAGGraphState) -> RAGGraphState:
    query = state.get("query") or ""
    dec = state.get("decomposition")
    dec_dict = dec if isinstance(dec, dict) else None
    context = drop_price_off_topic(
        filter_context_items(state.get("reranked_context") or []),
        str(query),
        dec_dict,
    )
    raw_turn = state.get("turn_contract")
    turn = TurnContract.from_dict(raw_turn if isinstance(raw_turn, dict) else None)
    raw_plan = state.get("bq_sql_plan")
    plan_dict = raw_plan if isinstance(raw_plan, dict) else {}
    requirement_plan = plan_dict.get("plan_source") == "turn_requirements"
    bq_results = state.get("bq_results") or []
    if requirement_plan:
        context = [item for item in context if not is_warehouse_item(item) or matches_bound_fact(item, turn, plan_dict)]
        bq_results = [row for row in bq_results if matches_bound_fact(row, turn, plan_dict)]
    usable_bq = [r for r in bq_results if is_usable_structured_bq_row(r)]
    analytical = bool(state.get("analytical_mode"))
    task_mode = str(state.get("task_mode") or ("analytical" if analytical else "chat"))
    if (
        analytical
        or task_mode in ("fact_lookup", "data_export_only")
        or should_elevate_bq_context(str(query), dec_dict, usable_bq=bool(usable_bq))
    ):
        context = pin_bq_context_first(context)
    gkw: dict[str, Any] = {
        "decomposition": dec_dict,
        "task_mode": task_mode,
        "retrieval_partial": bool(state.get("retrieval_partial")),
    }
    if not turn.documents and not state.get("export_intent"):
        gkw["price_list_rows"] = usable_bq
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
    state_queries = list(state.get("bq_sql_queries") or [])
    pre_queries = collect_pre_queries(
        plan_dict,
        state_queries=state_queries,
        bq_sql_debug=bq_debug,
    )
    warehouse_attempted = plan_indicates_warehouse_attempt(plan_dict) or bool(
        pre_queries or bq_debug
    )
    mergeable_bq = [
        r
        for r in (bq_results if isinstance(bq_results, list) else [])
        if isinstance(r, dict) and is_mergeable_bq_evidence(r)
    ]
    # Also admit filtered context BQ items (post-merge) for numeric availability.
    context_bq = [
        it for it in context if isinstance(it, dict) and is_mergeable_bq_evidence(it)
    ]
    has_mergeable_bq = bool(usable_bq or mergeable_bq or context_bq)
    exec_flags = bq_execute_flags(
        bq_debug,
        pre_queries=pre_queries,
        usable_bq=bool(usable_bq) or has_mergeable_bq,
        compile_error=bool(plan_dict.get("compile_error")),
        warehouse_attempted=warehouse_attempted,
    )
    gkw.update(exec_flags)
    gkw["pre_queries"] = pre_queries
    gkw["usable_bq"] = bool(usable_bq) or has_mergeable_bq
    gkw["bq_sql_debug"] = bq_debug
    gkw["warehouse_was_attempted"] = warehouse_attempted
    if not usable_bq and not has_mergeable_bq and not any(
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
    if (usable_bq or has_mergeable_bq) and is_numeric_data_query(str(query), dec_dict):
        gkw["structured_bq_numeric_available"] = True
    elif (usable_bq or has_mergeable_bq) and is_comparative_bq_query(str(query), dec_dict):
        gkw["structured_bq_comparative_available"] = True
    reranked = context
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
    topics = [doc.question for doc in turn.documents]
    fact_coverage: list[dict[str, Any]] = []
    corpus_selection = state.get("corpus_selection")
    corpus_results = corpus_selection.get("results") if isinstance(corpus_selection, dict) else None
    document_coverage = (
        document_requirement_coverage(
            turn, context, corpus_results=corpus_results if isinstance(corpus_results, dict) else None,
        )
        if _has_explicit_document_requirements(turn) else []
    )
    if topics:
        gkw["composer_addendum"] = (
            "Address each requested aspect separately and state any evidence gap for that aspect. "
            "Requested aspects: " + "; ".join(topics)
        )
    if document_coverage:
        gkw["require_document_citations"] = True
        missing_documents = [
            f"{entry['question']} ({entry['status'].replace('_', ' ')})"
            for entry in document_coverage if entry["status"] != "candidate_available"
        ]
        instructions = (
            "Cite each document-based claim inline using its source number. "
            "Use the passage tagged for the requirement and verify its actual wording. "
            "Describe an association as such unless the cited passage establishes causation."
        )
        if missing_documents:
            instructions += " State these document evidence gaps: " + "; ".join(missing_documents)
            if len(missing_documents) == len(document_coverage):
                instructions += (
                    " No accepted passage supports the requested explanation. Report the verified comparison and "
                    "explain the limits on attribution. Omit any proposed causes or intervention recommendations."
                )
        gkw["composer_addendum"] += "\n\n" + instructions
    if requirement_plan:
        # Use only rows that actually reach generation and carry their original observation dates.
        numeric_context = [item for item in context if is_usable_structured_bq_row(item)]
        fact_coverage = fact_requirement_coverage(turn, plan_dict, numeric_context, bq_debug)
        gkw["fact_requirement_coverage"] = fact_coverage
        if not turn.documents and fact_coverage and all(entry["status"] == "satisfied" for entry in fact_coverage):
            context = numeric_context
    gaps = [entry for entry in fact_coverage if entry["status"] != "satisfied"]
    if any(fact.enumerate_members for fact in turn.facts):
        # Numeric discovery still admits matching reported figures. Unrelated
        # passages cannot turn a query that never ran into a claim of absence.
        context = [item for item in context if is_warehouse_item(item)] + requirement_document_candidates(turn, context)
        if not context:
            gen_plan = replace(gen_plan, evidence_tier="empty", answer_shape="gap_ack", has_usable_spine=False)
            gkw["generation_plan"] = gen_plan.to_dict()
        elif any(is_document_item(item) for item in context):
            gkw["require_document_citations"] = True
    onto_for_gen = state.get("ontology_plan")
    if isinstance(onto_for_gen, dict):
        gkw["ontology_plan"] = onto_for_gen
    elif state.get("source_basis"):
        gkw["ontology_plan"] = {"source_basis": state.get("source_basis")}
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
        gkw["composer_addendum"] = "\n\n".join(filter(None, (
            gkw.get("composer_addendum"), composer_addendum(reasoner, rows_by),
        )))

    # Advisory metacog: diagnosis addendum. Verify already ran in metacog_revise.
    mp_state = MetacogPlan.from_dict(
        state.get("metacog_plan") if isinstance(state.get("metacog_plan"), dict) else None
    )
    verified = MetacogVerify.from_dict(
        state.get("metacog_verify") if isinstance(state.get("metacog_verify"), dict) else None
    )
    if mp_state is not None and metacog_enabled():
        diag = metacog_diagnosis_addendum(mp_state)
        if diag:
            existing = str(gkw.get("composer_addendum") or "")
            gkw["composer_addendum"] = (existing + "\n\n" + diag).strip()[:1800]
        if verified is not None and verified.addendum:
            existing = str(gkw.get("composer_addendum") or "")
            gkw["composer_addendum"] = (existing + "\n\n" + verified.addendum).strip()[:2000]

    tc_raw = state.get("turn_contract")
    if isinstance(tc_raw, dict):
        gkw["turn_contract"] = tc_raw
    if gen_plan.effective_category:
        gkw["category"] = gen_plan.effective_category
    if requirement_plan:
        requirements = [
            "Compare each requested indicator (" + ", ".join(f.measure_id for f in turn.facts)
            + ") across the requested places and periods.",
            "Use only matching evidence and do not substitute a related metric. "
            "Explain relationships with cited assessments and distinguish causation from association. "
            "Do not treat partial rows as complete series or average across sources or scales.",
        ]
        gkw["comparison_plan"] = plan_dict
        if gaps:
            requirements.append(
                "Explain these limits briefly in the user's language: "
                + " ".join(fact_gap_description(gap) for gap in gaps)
                + " Keep and cite the supported findings. A query failure or missing mapping is not evidence of data absence."
            )
        if document_coverage:
            requirements.append("Verify the exact indicator, place, period and unit of any figure cited from a document.")
        gkw["composer_addendum"] = "\n\n".join(filter(None, (
            gkw.get("composer_addendum"), " ".join(requirements),
        )))
    if any((item.get("metadata") or {}).get("price_report") for item in usable_bq):
        report_guidance = (
            "This is a multi-year food-price report. Present an analytical report with an annual comparison "
            "table for each requested country. Use annual_prices for the historical figures. Separate national "
            "food indices and inflation rates from the sampled commodity prices. The commodity averages are "
            "unweighted market observations, not a cost-of-living index or constant-market panel. "
            "Report each series' actual ending year and missing baseline or intermediate years. "
            "Partial current-year averages are not full-year comparisons. Use the SQL change_pct or "
            "change_percentage_points only for their stated first_year and year, with inline citations. "
            "Retain the provided units and currencies; compare relative changes across currencies, not nominal levels. "
            "Explain drivers or give recommendations only when a cited document supports them."
        )
        if not any(is_document_item(item) for item in context):
            report_guidance += (
                " Document retrieval was incomplete, so attribution is not available; say that plainly, "
                "without describing the retrieval process."
                if state.get("retrieval_partial") else
                " No verified document explains the drivers; report this limit without inventing causes."
            )
        gkw["composer_addendum"] = "\n\n".join(filter(None, (gkw.get("composer_addendum"), report_guidance)))
    gen_t0 = time.perf_counter()
    gen_result = generate(query, context, **gkw)
    draft_answer = str(gen_result.answer or "")
    job = ""
    if isinstance(tc_raw, dict):
        job = str(tc_raw.get("job") or "")
    elif mp_state is not None:
        job = mp_state.advisory_mode

    revised_result = gen_result

    def _regenerate(addendum: str) -> str:
        nonlocal revised_result
        extra = str(gkw.get("composer_addendum") or "")
        gkw["composer_addendum"] = (extra + "\n\n" + addendum).strip()[:2000]
        revised_result = generate(query, context, **gkw)
        return str(revised_result.answer or "")

    final_answer, self_check = complete_self_check(
        draft_answer,
        plan=mp_state,
        verified=verified,
        query=str(query),
        job=job,
        evidence=evidence_sample(
            bq_results=list(bq_results) if isinstance(bq_results, list) else None,
            reranked_context=list(reranked) if isinstance(reranked, list) else None,
        ),
        regenerate=_regenerate,
    )
    gen_max = _generate_max_tokens(task_mode)
    if self_check.verdict != "skip":
        update_current_span_metadata(
            {
                "metacog_self_check_verdict": self_check.verdict,
                "metacog_self_check": self_check.to_dict(),
            }
        )

    # ACF Path B is computed post-cite inside generate/_finalize_generation_result.
    acf = revised_result.acf or no_evidence_acf()
    answer_lang = str(state.get("answer_lang") or detect_answer_language(query))
    dec_dict = dec if isinstance(dec, dict) else {}
    cited_topics = set(revised_result.cited_document_requirements)
    for entry in document_coverage:
        if entry["status"] == "candidate_available":
            entry["status"] = "cited_candidate" if entry["question"] in cited_topics else "candidate_not_cited"

    out_gen: dict[str, Any] = {
        "query": str(query),
        "decomposition": dec_dict,
        "answer": final_answer,
        "citations": revised_result.citations,
        "answer_lang": answer_lang,
        "answer_outcome": decide_outcome(filter_context_items(context)).kind,
        "generation_plan": gen_plan.to_dict(),
        "comparison_gaps": gaps,
        "fact_requirement_coverage": fact_coverage,
        "document_requirement_coverage": document_coverage,
        "generate_ms": trace_elapsed_ms(gen_t0),
        "generate_max_tokens": gen_max,
        "generate_input_chars": getattr(revised_result, "generate_input_chars", None),
        # ML-054: how many LLM calls this request stopped on the max_tokens
        # ceiling. Surfaced to Langfuse so truncation frequency is measurable
        # instead of invisible; tune RAG_GENERATE_MAX_TOKENS from real data.
        "generate_length_capped_calls": length_capped_call_count(purpose="generate"),
        "generate_finish_reason": last_finish_reason(purpose="generate"),
        "error": getattr(revised_result, "error", None),
        **acf_result_to_state(acf),
        **{k: True for k in exec_flags if exec_flags.get(k)},
    }
    if self_check.verdict != "skip":
        out_gen["metacog_self_check"] = self_check.to_dict()
        out_gen["metacog_self_check_verdict"] = self_check.verdict
    
    return RAGGraphState(**out_gen)


def node_export(state: RAGGraphState) -> RAGGraphState:
    """Build downloadable artifacts when export is enabled on the route."""
    export_intent = state.get("export_intent")
    export_enabled = bool(state.get("export_enabled"))
    out: dict[str, Any] = {"artifacts": []}

    if not export_intent:
        return RAGGraphState(**out)

    answer = str(state.get("answer") or "")

    if not export_enabled:
        if EXPORT_UPGRADE_MESSAGE not in answer:
            out["answer"] = f"{answer}\n\n{EXPORT_UPGRADE_MESSAGE}".strip()
        return RAGGraphState(**out)

    try:
        artifacts = run_exports(
            export_kind=export_intent,
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

    return RAGGraphState(**out)


def node_generate_clarify(state: RAGGraphState) -> RAGGraphState:
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


def node_generate_meta(state: RAGGraphState) -> RAGGraphState:
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


def node_generate_social(state: RAGGraphState) -> RAGGraphState:
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

    # ML-057 (Sprint 2 Week 2, beta finding #9): a greeting or an out-of-scope
    # refusal is a non-answer -- there is no cited evidence behind it, so it
    # must not carry "Strong confidence". curated_product_acf() is reserved for
    # the curated product/meta knowledge base, which is genuinely reliable.
    if kind == "greeting":
        social_acf = no_evidence_acf(
            explanation=(
                "Greeting — no agricultural claim was made, so there is no "
                "evidence to score."
            )
        )
    else:
        social_acf = no_evidence_acf(
            explanation=(
                "Out of scope — no OpenTrace evidence was retrieved for this "
                "request."
            )
        )

    return {
        "answer": answer,
        "citations": [],
        "answer_lang": answer_lang,
        "is_greeting_query": kind == "greeting",
        "is_out_of_scope_query": kind == "out_of_scope",
        **acf_result_to_state(social_acf),
    }


def node_generate_product(state: RAGGraphState) -> RAGGraphState:
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


def node_retrieve_legs(state: RAGGraphState) -> RAGGraphState:
    """Dispatch only the selected source legs and overlap work when both are needed."""
    with request_budget(reserve_s=_GENERATION_RESERVE_S):
        return _retrieve_selected_legs(state)


def _retrieve_selected_legs(state: RAGGraphState) -> RAGGraphState:
    reason: RAGGraphState = {}

    def _bq_leg() -> RAGGraphState:
        nonlocal reason
        reason = node_bq_reason(state)
        retrieve = node_bq_retrieve({**state, **reason})
        return {**reason, **retrieve}

    bq_selected = _source_selected(state, "bigquery")
    docs_selected = _source_selected(state, "qdrant")
    if not bq_selected:
        docs = node_parallel_retrieve(state) if docs_selected else {}
        return RAGGraphState(**{**docs, "bq_results": [], "bq_sql_plan": {"skip_bq": True}, "sql_source": "skipped"})
    future = None
    if remaining_seconds() != 0:
        future = _WAREHOUSE_LEGS.try_submit(run_with_tracing_context(_bq_leg))
    docs: RAGGraphState = node_parallel_retrieve(state) if docs_selected else {
        "vector_results": [],
        "corpus_selection": {"active": [], "boosts": {}, "rationale": "source_selection"},
        "vector_ms": 0.0,
        "corpus_count": 0,
    }
    bq_failure = "bq_error"
    if future is not None:
        try:
            return {**docs, **future.result(timeout=remaining_seconds())}
        except FuturesTimeoutError:
            future.cancel()
            bq_failure = "bq_timeout"
        except Exception:
            logger.exception("Warehouse retrieval leg failed")
    elif remaining_seconds() == 0:
        bq_failure = "bq_timeout"
    return {
        **docs,
        **reason,
        "bq_results": [],
        "bq_sql_debug": [{"status": bq_failure}],
    }


def _source_selected(state: RAGGraphState, source: Source) -> bool:
    sources = state.get("source_selection")
    return sources is None or sources[source]["selected"]


def _has_explicit_document_requirements(turn: TurnContract) -> bool:
    return bool(turn.documents)


def _route_after_evidence(state: RAGGraphState) -> str:
    sources = state.get("source_selection")
    if sources and (
        not sources["qdrant"]["selected"]
        or sources["qdrant"]["reason"] in ("document_content_requested", "indicator_context")
    ):
        return "generate"
    return route_after_rerank(dict(state))


def _evidence_empty(state: RAGGraphState) -> bool:
    reranked = state.get("reranked_context") or []
    if reranked:
        return False
    bq = state.get("bq_results") or []
    return not any(isinstance(row, dict) and is_usable_structured_bq_row(row) for row in bq)


def _profile_ui_scope_active(decomposition: dict[str, Any]) -> bool:
    geo_source = str(decomposition.get("geography_source") or "")
    time_source = str(decomposition.get("time_source") or "")
    return geo_source in {"profile", "ui"} or time_source == "ui"


def _profile_ui_scope_updates(state: RAGGraphState) -> RAGGraphState:
    """Return updates that clear profile/UI filters and keep query-derived scope."""
    dec_raw = state.get("decomposition")
    dec: dict[str, Any] = dec_raw if isinstance(dec_raw, dict) else {}
    geo_source = str(dec.get("geography_source") or "")
    time_source = str(dec.get("time_source") or "")
    cleared = clear_profile_ui_retrieval_scope(dec)
    updates: RAGGraphState = {
        "decomposition": cleared,
        "profile_scope_retry": 1,
    }
    if geo_source in {"profile", "ui"}:
        updates["geo_override"] = ""
    if time_source == "ui":
        updates["time_start_override"] = ""
        updates["time_end_override"] = ""
        contract = state.get("turn_contract")
        if isinstance(contract, dict):
            rewritten = dict(contract)
            spec = dict(rewritten.get("time_spec") or {})
            spec["start"] = ""
            spec["end"] = ""
            spec["hard_filter"] = False
            rewritten["time_spec"] = spec
            updates["turn_contract"] = rewritten
    return updates


def node_coverage_retry(state: RAGGraphState) -> RAGGraphState:
    """Retry an empty document search once when optional filters can be removed."""
    remaining = remaining_seconds()
    if remaining is not None and remaining <= _GENERATION_RESERVE_S + 1:
        return {}
    plan = state.get("bq_sql_plan")
    if isinstance(plan, dict) and plan.get("plan_source") == "turn_requirements":
        # The comparison plan has already searched its explicit periods and places.
        # Repeating whole retrieval cannot resolve a missing or unbound series.
        return {}
    dec_raw = state.get("decomposition")
    dec: dict[str, Any] = dec_raw if isinstance(dec_raw, dict) else {}
    if (
        int(state.get("profile_scope_retry") or 0) < 1
        and _source_selected(state, "qdrant")
        and not state.get("retrieval_partial")
        and _evidence_empty(state)
        and _profile_ui_scope_active(dec)
    ):
        updates = _profile_ui_scope_updates(state)
        with request_budget(reserve_s=_GENERATION_RESERVE_S), observed_span("profile_scope_retry", input_data={"attempt": 1}):
            updates.update(node_parallel_retrieve({**state, **updates}))
            updates.update(node_merge({**state, **updates}))
            updates.update(node_rerank({**state, **updates}))
            update_current_span_metadata({"profile_scope_retry": 1})
        return updates
    return {}


def node_metacog_revise(state: RAGGraphState) -> RAGGraphState:
    """Verify evidence, then at most one reasoning replan back into retrieve."""
    plan = MetacogPlan.from_dict(
        state.get("metacog_plan") if isinstance(state.get("metacog_plan"), dict) else None
    )
    already = int(state.get("metacog_revise") or 0)
    updates: RAGGraphState = {
        "metacog_next": "generate",
        "metacog_revise": already,
        "metacog_revise_action": "skip",
    }
    if plan is None or not metacog_enabled():
        return updates

    with observed_span(
        "metacog_revise",
        input_data={"already_revised": already, "query": str(state.get("query") or "")[:200]},
    ):
        dec_raw = state.get("decomposition") if isinstance(state.get("decomposition"), dict) else {}
        return _metacog_revise_body(state, plan, dec_raw, already, updates)


def _metacog_revise_body(
    state: RAGGraphState,
    plan: MetacogPlan,
    dec_raw: Any,
    already: int,
    updates: RAGGraphState,
) -> RAGGraphState:
    dec = dec_raw if isinstance(dec_raw, dict) else {}
    bq_results = state.get("bq_results") if isinstance(state.get("bq_results"), list) else []
    reranked = state.get("reranked_context") if isinstance(state.get("reranked_context"), list) else []
    if metacog_verify_enabled() and not plan.skip_verify:
        verified = verify_metacog_evidence(
            plan,
            query=str(state.get("query") or ""),
            decomposition=dec,
            bq_results=bq_results,
            reranked_context=reranked,
        )
    else:
        verified = MetacogVerify(status="ok", evidence_tier_hint="strong")
    updates["metacog_verify"] = verified.to_dict()
    update_current_span_metadata(
        {
            "metacog_verify_status": verified.status,
            "metacog_verify_tier": verified.evidence_tier_hint,
            "metacog_failed_checks": list(verified.failed_checks)[:8],
        }
    )

    decision = propose_metacog_revise(
        plan,
        verified,
        query=str(state.get("query") or ""),
        decomposition=dec,
        already_revised=already,
        bq_results=bq_results,
        reranked_context=reranked,
    )
    updates["metacog_revise_action"] = decision.action
    if decision.action != "replan":
        return updates

    turn = TurnContract.from_dict(
        state.get("turn_contract") if isinstance(state.get("turn_contract"), dict) else None
    )
    updated_turn, revised_plan, counter = apply_revise_decision(turn, plan, decision)
    updates["turn_contract"] = updated_turn.to_dict()
    updates["metacog_plan"] = revised_plan.to_dict()
    updates["metacog_revise"] = counter
    if decision.geography:
        next_dec = dict(dec)
        next_dec["geography"] = list(decision.geography)
        updates["decomposition"] = next_dec
    updates["metacog_next"] = "retrieve_legs"
    update_current_span_metadata(
        {
            "metacog_revise": counter,
            "metacog_revise_action": decision.action,
            "metacog_mode": revised_plan.advisory_mode,
        }
    )
    return updates


def _route_after_metacog_revise(state: RAGGraphState) -> str:
    if str(state.get("metacog_next") or "") == "retrieve_legs":
        return "retrieve_legs"
    return "generate"


def build_graph():
    """Build and compile the LangGraph RAG graph. Requires langgraph."""
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("Install langgraph: pip install langgraph") from None

    graph = StateGraph(RAGGraphState)

    def _decompose_with_budget(state: RAGGraphState) -> RAGGraphState:
        with request_budget(reserve_s=_GENERATION_RESERVE_S):
            return node_decompose(state)

    graph.add_node("decompose", _decompose_with_budget)
    graph.add_node("retrieve_legs", node_retrieve_legs)
    graph.add_node("merge", node_merge)
    graph.add_node("rerank", node_rerank)
    graph.add_node("coverage_retry", node_coverage_retry)
    graph.add_node("metacog_revise", node_metacog_revise)
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
    graph.add_edge("retrieve_legs", "merge")
    graph.add_edge("merge", "rerank")
    graph.add_edge("rerank", "coverage_retry")
    graph.add_edge("generate", "export")
    graph.add_edge("export", END)
    graph.add_edge("insufficient_context", END)
    graph.add_edge("generate_meta", END)
    graph.add_edge("generate_product", END)
    graph.add_edge("generate_social", END)
    graph.add_edge("generate_language_help", END)
    graph.add_edge("generate_clarify", END)

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
        return "retrieve_legs"

    graph.add_conditional_edges(
        "decompose",
        _route_after_decompose,
        {
            "generate_meta": "generate_meta",
            "generate_product": "generate_product",
            "generate_social": "generate_social",
            "generate_language_help": "generate_language_help",
            "generate_clarify": "generate_clarify",
            "retrieve_legs": "retrieve_legs",
        },
    )
    graph.add_conditional_edges(
        "coverage_retry",
        _route_after_evidence,
        {
            "web_fallback": "web_fallback",
            "generate": "metacog_revise",
        },
    )
    graph.add_conditional_edges(
        "web_fallback",
        _route_after_web_fallback,
        {
            "generate": "metacog_revise",
            "insufficient_context": "insufficient_context",
        },
    )
    graph.add_conditional_edges(
        "metacog_revise",
        _route_after_metacog_revise,
        {
            "retrieve_legs": "retrieve_legs",
            "generate": "generate",
        },
    )

    return graph.compile()


# Compile on first use in each process.
_compiled_graph = None

def _get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_rag(query: str, **kwargs: Any) -> dict[str, Any]:
    """Run the RAG pipeline and return the state (including answer)."""
    reset_llm_usage()
    reset_llm_truncation()  # ML-054
    graph = _get_compiled_graph()
    initial: RAGGraphState = {
        "query": query,
        "decomposition": {},
        "bq_table_candidates": [],
        "bq_sql_plan": {},
        "merged_context": [],
        "reranked_context": [],
        "answer_outcome": "",
    }
    raw_profile = kwargs.get("user_profile") if isinstance(kwargs.get("user_profile"), dict) else None
    user_profile = dict(raw_profile) if raw_profile is not None else None
    if user_profile is not None:
        raw_country = user_profile.get("country")
        user_profile["country"] = sanitize_profile_country(
            str(raw_country) if raw_country is not None else None
        )
    plan_type = kwargs.get("plan_type")
    profile_geo = effective_geo_override(plan_type, user_profile)
    # Flag off: no hard override. Flag on: still do not replace a country the query named.
    if profile_geo and not _extract_countries(query):
        initial["geo_override"] = profile_geo  # type: ignore[assignment]
    if plan_type:
        initial["plan_type"] = plan_type  # type: ignore[assignment]
    if kwargs.get("category"):
        initial["category"] = kwargs["category"]  # type: ignore[assignment]
    if user_profile is not None:
        initial["user_profile"] = user_profile  # type: ignore[assignment]
    if kwargs.get("reference_date"):
        initial["reference_date"] = str(kwargs["reference_date"]).strip()[:10]
    else:
        initial["reference_date"] = utc_today().isoformat()
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
    if kwargs.get("session_id"):
        initial["session_id"] = kwargs["session_id"]  # type: ignore[assignment]
    # Session state is loaded by the adapter (owner-bound) and handed in; the core never reads the store.
    if isinstance(kwargs.get("active_request"), dict):
        initial["active_request"] = kwargs["active_request"]  # type: ignore[assignment]
    if isinstance(kwargs.get("last_bq_facts"), list):
        initial["last_bq_facts"] = kwargs["last_bq_facts"]  # type: ignore[assignment]
    if "conversation_summary" in kwargs:
        initial["conversation_summary"] = kwargs["conversation_summary"]  # type: ignore[assignment]
    if "recent_turns" in kwargs:
        initial["recent_turns"] = kwargs["recent_turns"]  # type: ignore[assignment]
    if kwargs.get("export_enabled"):
        initial["export_enabled"] = True  # type: ignore[assignment]
    if kwargs.get("export_intent"):
        initial["export_intent"] = kwargs["export_intent"]  # type: ignore[assignment]
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
    timeout = stage_budget_seconds("RAG_REQUEST_TIMEOUT_S", default_s=120.0)
    if _answer_mode(kwargs) == "rarr":
        out = _run_rarr(query, graph, initial, cfg, timeout, kwargs)
    else:
        with request_budget(timeout):
            result = graph.invoke(initial, config=cfg)
        out = dict(result)
    out.setdefault("citations", [])
    out.setdefault("artifacts", [])
    out["usage"] = get_llm_usage().to_dict()
    return out


def _answer_mode(kwargs: dict[str, Any]) -> str:
    """Caller's answer_mode wins; otherwise RAG_ANSWER_MODE (evidence or rarr, default evidence)."""
    raw = kwargs.get("answer_mode") or os.environ.get("RAG_ANSWER_MODE", "")
    return "rarr" if str(raw).strip().lower() == "rarr" else "evidence"


_RARR_MIN_REVISE_S = 1.0


def _timed_draft(**kw: Any) -> tuple[str, float]:
    started = time.perf_counter()
    text = rarr.draft_answer(**kw)
    return text, round((time.perf_counter() - started) * 1000.0, 1)


def _draft_and_retrieve(
    query: str,
    graph: Any,
    initial: RAGGraphState,
    cfg: Any,
    kwargs: dict[str, Any],
    reference_date: date,
    stages: dict[str, Any],
) -> dict[str, Any]:
    """Run the draft beside the unchanged graph. Fills stages; returns the graph state."""
    recent = kwargs.get("recent_turns")
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rarr-draft")
    try:
        pending = pool.submit(
            contextvars.copy_context().run,
            _timed_draft,
            query=query,
            plan_type=kwargs.get("plan_type"),
            conversation_summary=str(kwargs.get("conversation_summary") or ""),
            recent_turns=recent if isinstance(recent, list) else None,
            reference_date=reference_date,
        )
        started = time.perf_counter()
        out = dict(graph.invoke(initial, config=cfg))
        stages["graph_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        # Meta, product, social and language-help routes answer without retrieval and keep
        # their answer; the unused draft finishes in the background. A clarify route would
        # leave the user without an answer, so RARR answers it from the draft instead.
        if not out.get("answer_outcome") and out.get("task_mode") != "clarify":
            stages["skip"] = "route did not retrieve evidence"
            return out
        try:
            stages["draft"], stages["draft_ms"] = pending.result(timeout=remaining_seconds(None))
        except Exception:
            logger.exception("rarr draft failed or ran out of time")
        return out
    finally:
        pool.shutdown(wait=False)


def _run_rarr(
    query: str,
    graph: Any,
    initial: RAGGraphState,
    cfg: Any,
    timeout: float,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Draft beside the unchanged graph, then revise the draft against the graph's evidence.

    One budget covers the whole request, so draft, graph and revision together never exceed
    RAG_REQUEST_TIMEOUT_S. The draft and graph run with RAG_RARR_REVISE_RESERVE_S held back,
    so the revision keeps that time however long retrieval takes.
    """
    reference_date = date.fromisoformat(str(initial["reference_date"]))
    stages: dict[str, Any] = {"draft": "", "draft_ms": None, "graph_ms": None}
    reserve = min(stage_budget_seconds("RAG_RARR_REVISE_RESERVE_S", default_s=40.0), timeout / 2)
    with request_budget(timeout):
        with request_budget(None, reserve_s=reserve):
            out = _draft_and_retrieve(query, graph, initial, cfg, kwargs, reference_date, stages)
        skip = stages.pop("skip", "")
        if not skip and not stages["draft"]:
            skip = "draft failed, empty, or out of time"
        left = remaining_seconds(None)
        if not skip and left is not None and left < _RARR_MIN_REVISE_S:
            skip = f"only {left:.1f}s left for revision"
        if skip:
            out["rarr"] = {**stages, "status": "skipped", "reason": skip}
            return out
        with observed_span("rarr_revise"):
            try:
                revision = rarr.revise_with_evidence(
                    query,
                    stages["draft"],
                    out,
                    plan_type=kwargs.get("plan_type"),
                    reference_date=reference_date,
                )
            except Exception as exc:
                logger.exception("rarr revise failed")
                out["rarr"] = {**stages, "status": "skipped", "reason": f"revise failed: {exc}"}
                return out
    if revision is None:
        out["rarr"] = {**stages, "status": "skipped", "reason": "reviser returned nothing after retries; evidence answer kept"}
        return out
    rarr.apply_revision(out, stages, revision)
    return out
