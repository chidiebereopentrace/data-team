"""
RAG graph: query → decompose → parallel retrieval (BQ table match + news + academic)
→ BigQuery lookup → merge → rerank → generate.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypedDict, cast

from ml.rag.chatbot.assistant_identity import is_meta_query
from ml.rag.chatbot.product_knowledge import is_product_query
from ml.rag.chatbot.bq_table_matcher import match_bq_tables_from_descriptions
from ml.rag.chatbot.generator import generate
from ml.rag.chatbot.query_decomposer import (
    decompose_query,
    normalize_geography_for_filter,
    resolve_retrieval_geographies,
)
from ml.rag.chatbot.reranker import rerank
from ml.rag.retrievers.bq_retriever import BQRetriever
from ml.rag.retrievers.vector_retriever import VectorRetriever

from langchain_core.runnables import RunnableConfig

from ml.rag.text_processors.preprocess.bibliographic_metadata import format_academic_citation
from ml.rag.observability import get_langfuse_callback

logger = logging.getLogger(__name__)


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
) -> list[dict[str, Any]]:
    """Try retrieval with full filters, then relax geo and/or time when results are empty."""
    attempts: list[dict[str, Any]] = [dict(base_kwargs)]
    has_geo = bool(countries)
    if has_geo and allow_geo_fallback and _env_on(geo_fallback_env):
        no_geo = dict(base_kwargs)
        no_geo.pop("geo_country", None)
        no_geo.pop("geo_countries", None)
        attempts.append(no_geo)
    if has_time and _env_on(time_fallback_env):
        no_time = dict(base_kwargs)
        no_time.pop("published_at_from", None)
        no_time.pop("published_at_to", None)
        attempts.append(no_time)
    if (
        has_geo
        and has_time
        and allow_geo_fallback
        and _env_on(geo_fallback_env)
        and _env_on(time_fallback_env)
    ):
        relaxed = dict(base_kwargs)
        relaxed.pop("geo_country", None)
        relaxed.pop("geo_countries", None)
        relaxed.pop("published_at_from", None)
        relaxed.pop("published_at_to", None)
        attempts.append(relaxed)

    seen: set[str] = set()
    base_key = _kwargs_attempt_key(base_kwargs)
    for kwargs in attempts:
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
            if key != base_key:
                logger.debug(
                    "Vector retrieve fallback for %s: %d hits (filters=%s)",
                    vr.collection_name,
                    len(raw),
                    {k: kwargs[k] for k in kwargs if k not in ("top_k", "vector_search_mode", "doc_kind", "doc_kinds")},
                )
            return raw
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
    dec = state.get("decomposition") or {}
    coll = os.environ.get(collection_env, default_collection).strip() or default_collection
    vr = VectorRetriever(collection_name=coll)

    countries = resolve_retrieval_geographies(
        geo_override=str(state.get("geo_override") or ""),
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )
    ts = (state.get("time_start_override") or dec.get("time_start") or "").strip()[:10]
    te = (state.get("time_end_override") or dec.get("time_end") or "").strip()[:10]

    kwargs = build_kwargs(state, dec)
    _apply_geo_to_kwargs(kwargs, countries)
    if ts:
        kwargs["published_at_from"] = ts
    if te:
        kwargs["published_at_to"] = te

    allow_geo_fb = not _strict_compare_filters(dec)
    return _retrieve_vector_cascade(
        vr,
        q,
        base_kwargs=kwargs,
        countries=countries,
        has_time=bool(ts or te),
        geo_fallback_env=geo_fallback_env,
        time_fallback_env=time_fallback_env,
        allow_geo_fallback=allow_geo_fb,
    )


class RAGGraphState(TypedDict, total=False):
    query: str
    decomposition: dict[str, Any]
    bq_table_candidates: list[dict[str, Any]]
    vector_news_results: list[dict[str, Any]]
    vector_academic_results: list[dict[str, Any]]
    vector_ota_results: list[dict[str, Any]]
    vector_results: list[dict[str, Any]]
    bq_results: list[dict[str, Any]]
    bq_sql_queries: list[str]
    merged_context: list[dict[str, Any]]
    reranked_context: list[dict[str, Any]]
    answer: str
    error: str | None
    # Optional UI / API overrides (see run_rag)
    geo_override: str | None
    time_start_override: str | None
    time_end_override: str | None
    news_top_k: int | None
    academic_top_k: int | None
    bq_top_k: int | None
    rerank_top_k: int | None
    # Generator memory: rolling summary + last N verbatim pairs (see ml.rag.chat_memory)
    conversation_summary: str | None
    recent_turns: list[dict[str, Any]] | None
    chat_history: list[dict[str, Any]] | None  # legacy: verbatim-only, no summary
    is_meta_query: bool | None
    is_product_query: bool | None


def node_decompose(state: RAGGraphState) -> dict[str, Any]:
    q = (state.get("query") or "").strip()
    dec = decompose_query(q)
    meta = is_meta_query(q)
    product = (not meta) and is_product_query(q, dec)
    return {"decomposition": dec, "is_meta_query": meta, "is_product_query": product}


def _tag_vector(item: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        **item,
        "source": kind,
        "_context_kind": kind,
    }


_RESEARCH_DOC_KINDS = ("academic_article", "policy_document", "public_report")


def _news_kwargs(state: RAGGraphState, dec: dict[str, Any]) -> dict[str, Any]:
    top_k = int(state.get("news_top_k") or 20)
    kwargs: dict[str, Any] = {
        "doc_kind": "news_article",
        "top_k": top_k,
        "vector_search_mode": "dense_named",
    }
    if os.environ.get("RAG_NEWS_DOMAIN_FILTER", "").strip().lower() in ("1", "true", "on", "yes"):
        domains = dec.get("domains") or []
        if domains:
            kwargs["domains_substring"] = str(domains[0])
    return kwargs


def _academic_kwargs(state: RAGGraphState, dec: dict[str, Any]) -> dict[str, Any]:
    return {
        "top_k": int(state.get("academic_top_k") or 20),
        "doc_kinds": list(_RESEARCH_DOC_KINDS),
        "vector_search_mode": "dense_named",
    }


def _retrieve_news(state: RAGGraphState) -> list[dict[str, Any]]:
    dec = state.get("decomposition") or {}
    q = (state.get("query") or "").strip()
    countries = resolve_retrieval_geographies(
        geo_override=str(state.get("geo_override") or ""),
        geography=dec.get("geography") if isinstance(dec.get("geography"), list) else None,
    )
    ts = (state.get("time_start_override") or dec.get("time_start") or "").strip()[:10]
    te = (state.get("time_end_override") or dec.get("time_end") or "").strip()[:10]
    top_k = int(state.get("news_top_k") or 20)
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


def _retrieve_academic(state: RAGGraphState) -> list[dict[str, Any]]:
    """Research retrieval (academic / policy / public report) with geo + year filters."""
    raw = _vector_retrieve_for_corpus(
        state,
        collection_env="QDRANT_COLLECTION_RESEARCH_PAPERS",
        default_collection="research_other_papers",
        build_kwargs=_academic_kwargs,
        geo_fallback_env="RAG_RESEARCH_GEO_FALLBACK",
        time_fallback_env="RAG_RESEARCH_TIME_FALLBACK",
    )
    return [_tag_vector(x, "research") for x in raw]


def _retrieve_ota(state: RAGGraphState) -> list[dict[str, Any]]:
    """Retrieve OTA insights (triple vectors) from the OTA_insights collection.

    The collection may be empty on initial deployment (analysts populate it later).
    Results are merged into the main context with clear OTA citations.
    """
    coll = os.environ.get("QDRANT_COLLECTION_OTA_INSIGHTS", "OTA_insights").strip() or "OTA_insights"
    vr = VectorRetriever(collection_name=coll)

    q = (state.get("query") or "").strip()
    top_k = int(state.get("ota_top_k") or 10)

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

    return [_tag_vector(x, "ota_insight") for x in (raw or [])]


def _research_context_label(meta: dict[str, Any]) -> tuple[str, str]:
    """Return (source tag, content prefix) for a research-corpus chunk."""
    dk = str(meta.get("doc_kind") or "").strip().lower()
    if dk == "policy_document":
        title = str(meta.get("section_title") or meta.get("label") or meta.get("source_file") or "").strip()
        prefix = f"[Policy | {title}]" if title else "[Policy]"
        return "policy", prefix
    if dk == "public_report":
        title = str(meta.get("section_title") or meta.get("label") or meta.get("source_file") or "").strip()
        prefix = f"[Public report | {title}]" if title else "[Public report]"
        return "public_report", prefix
    cite = format_academic_citation(meta)
    prefix = f"[Academic | {cite}]" if cite else "[Academic]"
    return "academic", prefix


def _retrieve_bq_tables(state: RAGGraphState) -> list[dict[str, Any]]:
    q = (state.get("query") or "").strip()
    return match_bq_tables_from_descriptions(q, top_k=10)


def node_parallel_retrieve(state: RAGGraphState) -> dict[str, Any]:
    """Run BQ table-description match, news, academic, and OTA retrieval in parallel."""
    bq_cands: list[dict[str, Any]] = []
    news_out: list[dict[str, Any]] = []
    academic_out: list[dict[str, Any]] = []
    ota_out: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(_retrieve_bq_tables, state): "bq_tables",
            ex.submit(_retrieve_news, state): "news",
            ex.submit(_retrieve_academic, state): "academic",
            ex.submit(_retrieve_ota, state): "ota",
        }
        for fut in as_completed(futs):
            kind = futs[fut]
            try:
                res = fut.result()
            except Exception:
                logger.exception("Parallel retrieval failed for %s; returning empty list", kind)
                res = []
            if kind == "bq_tables":
                bq_cands = res
            elif kind == "news":
                news_out = res
            elif kind == "academic":
                academic_out = res
            else:
                ota_out = res

    combined = list(news_out) + list(academic_out) + list(ota_out)
    return {
        "bq_table_candidates": bq_cands,
        "vector_news_results": news_out,
        "vector_academic_results": academic_out,
        "vector_ota_results": ota_out,
        "vector_results": combined,
    }


def node_bq_retrieve(state: RAGGraphState) -> dict[str, Any]:
    q = (state.get("query") or "").strip()
    dec = state.get("decomposition") or {}
    cands = state.get("bq_table_candidates") or []
    max_sql = max(1, int(os.environ.get("RAG_BQ_MAX_SQL_QUERIES", "10") or 10))
    hints = [str(c.get("content") or "") for c in cands[:max_sql] if c.get("content")]
    top_k = int(state.get("bq_top_k") or 15)
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
    results = retriever.retrieve(
        q,
        top_k=top_k,
        table_hints=hints,
        time_start=ts or None,
        time_end=te or None,
        entities=entities,
        domains=domains,
        **bq_geo,
    )
    sql_seen: set[str] = set()
    bq_sql_queries: list[str] = []
    for row in results:
        sql = str((row.get("metadata") or {}).get("sql") or "").strip()
        if sql and sql not in sql_seen:
            sql_seen.add(sql)
            bq_sql_queries.append(sql)
    return {"bq_results": results, "bq_sql_queries": bq_sql_queries}


def node_merge(state: RAGGraphState) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    for r in state.get("bq_results") or []:
        text = str(r.get("content") or "").strip()
        merged.append(
            {
                **r,
                "content": f"[Structured data] {text}" if text else "[Structured data]",
                "source": r.get("source", "bigquery"),
                "_context_kind": "bigquery",
            }
        )
    for item in state.get("vector_news_results") or []:
        text = item.get("content") or ""
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        merged.append(
            {
                "content": f"[News] {text}",
                "source": "news",
                "_context_kind": "news",
                "metadata": meta,
                "score": item.get("score"),
            }
        )
    for item in state.get("vector_academic_results") or []:
        text = item.get("content") or ""
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source_tag, label = _research_context_label(meta if isinstance(meta, dict) else {})
        merged.append(
            {
                "content": f"{label} {text}",
                "source": source_tag,
                "_context_kind": source_tag,
                "metadata": meta,
                "score": item.get("score"),
            }
        )

    # OTA insights (from OTA_insights collection, ota_triple vectors)
    # Merged into main context (per spec) with clear OTA citations retained.
    for item in state.get("vector_ota_results") or []:
        text = item.get("content") or ""
        meta = _chunk_metadata(item)
        # Choose nice prefix based on common OTA payload fields
        if meta.get("recommendation_text") or "recommendation" in str(meta.get("doc_kind") or "").lower():
            label = "[OTA Recommendation]"
        elif meta.get("metric_text") or "metric" in str(meta.get("doc_kind") or "").lower():
            label = "[OTA Metric]"
        else:
            label = "[OTA Insight]"
        merged.append(
            {
                "content": f"{label} {text}",
                "source": "ota_insight",
                "_context_kind": "ota_insight",
                "metadata": meta,
                "score": item.get("score"),
            }
        )

    return {"merged_context": merged}


def node_rerank(state: RAGGraphState) -> dict[str, Any]:
    query = state.get("query") or ""
    merged = state.get("merged_context") or []
    top_k = int(state.get("rerank_top_k") or 20)
    top = rerank(query, merged, top_k=top_k)
    return {"reranked_context": top}


def node_generate(state: RAGGraphState) -> dict[str, Any]:
    query = state.get("query") or ""
    context = state.get("reranked_context") or []
    dec = state.get("decomposition")
    gkw: dict[str, Any] = {"decomposition": dec if isinstance(dec, dict) else None}
    cs = state.get("conversation_summary")
    rt = state.get("recent_turns")
    has_mem = (isinstance(cs, str) and cs.strip()) or (
        isinstance(rt, list) and len(rt) > 0
    )
    if has_mem:
        gkw["conversation_summary"] = cs if isinstance(cs, str) else ""
        gkw["recent_turns"] = list(rt) if isinstance(rt, list) else []
    elif state.get("chat_history"):
        gkw["chat_history"] = state.get("chat_history")
    answer = generate(query, context, **gkw)
    return {"answer": answer}


def node_generate_meta(state: RAGGraphState) -> dict[str, Any]:
    """Short-circuit node for identity meta questions. No retrieval."""
    from ml.rag.chatbot.assistant_identity import generate_meta_answer

    query = state.get("query") or ""
    gkw: dict[str, Any] = {}
    cs = state.get("conversation_summary")
    rt = state.get("recent_turns")
    has_mem = (isinstance(cs, str) and cs.strip()) or (isinstance(rt, list) and len(rt) > 0)
    if has_mem:
        gkw["conversation_summary"] = cs if isinstance(cs, str) else ""
        gkw["recent_turns"] = list(rt) if isinstance(rt, list) else []
    elif state.get("chat_history"):
        gkw["chat_history"] = state.get("chat_history")
    if state.get("stakeholder_type"):
        gkw["stakeholder_type"] = state.get("stakeholder_type")
    if state.get("audience_instructions"):
        gkw["audience_instructions"] = state.get("audience_instructions")
    answer = generate_meta_answer(query, **gkw)
    return {"answer": answer}


def node_generate_product(state: RAGGraphState) -> dict[str, Any]:
    """Short-circuit node for OpenTrace product questions. Uses product KB, no retrieval."""
    from ml.rag.chatbot.product_knowledge import generate_product_answer

    query = state.get("query") or ""
    gkw: dict[str, Any] = {}
    cs = state.get("conversation_summary")
    rt = state.get("recent_turns")
    has_mem = (isinstance(cs, str) and cs.strip()) or (isinstance(rt, list) and len(rt) > 0)
    if has_mem:
        gkw["conversation_summary"] = cs if isinstance(cs, str) else ""
        gkw["recent_turns"] = list(rt) if isinstance(rt, list) else []
    elif state.get("chat_history"):
        gkw["chat_history"] = state.get("chat_history")
    if state.get("stakeholder_type"):
        gkw["stakeholder_type"] = state.get("stakeholder_type")
    if state.get("audience_instructions"):
        gkw["audience_instructions"] = state.get("audience_instructions")
    answer = generate_product_answer(query, **gkw)
    return {"answer": answer}


def build_graph():
    """Build and compile the LangGraph RAG graph. Requires langgraph."""
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("Install langgraph: pip install langgraph") from None

    graph = StateGraph(RAGGraphState)

    graph.add_node("decompose", node_decompose)
    graph.add_node("parallel_retrieve", node_parallel_retrieve)
    graph.add_node("bq_retrieve", node_bq_retrieve)
    graph.add_node("merge", node_merge)
    graph.add_node("rerank", node_rerank)
    graph.add_node("generate", node_generate)
    graph.add_node("generate_meta", node_generate_meta)

    graph.add_edge(START, "decompose")

    def _route_after_decompose(state: RAGGraphState) -> str:
        return "generate_meta" if state.get("is_meta_query") else "parallel_retrieve"

    graph.add_conditional_edges("decompose", _route_after_decompose)
    graph.add_edge("parallel_retrieve", "bq_retrieve")
    graph.add_edge("bq_retrieve", "merge")
    graph.add_edge("merge", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("generate_meta", END)
    graph.add_edge("generate_product", END)

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
    graph = _get_compiled_graph()
    initial: RAGGraphState = {"query": query}
    for key in (
        "geo_override",
        "time_start_override",
        "time_end_override",
        "news_top_k",
        "academic_top_k",
        "bq_top_k",
        "rerank_top_k",
        "chat_history",
        "stakeholder_type",
        "audience_instructions",
    ):
        if key in kwargs and kwargs[key] is not None:
            initial[key] = kwargs[key]  # type: ignore[assignment]
    if "conversation_summary" in kwargs:
        initial["conversation_summary"] = kwargs["conversation_summary"]  # type: ignore[assignment]
    if "recent_turns" in kwargs:
        initial["recent_turns"] = kwargs["recent_turns"]  # type: ignore[assignment]
    cfg: RunnableConfig | None = kwargs.get("config")
    handler = get_langfuse_callback()
    if handler:
        base: dict[str, Any] = dict(cfg) if cfg else {}
        cbs = list(base.get("callbacks") or [])
        if handler not in cbs:
            cbs.append(handler)
        cfg = cast(RunnableConfig, {**base, "callbacks": cbs})
    result = graph.invoke(initial, config=cfg)
    return dict(result)
