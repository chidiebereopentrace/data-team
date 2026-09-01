"""YAML-index SQL reasoner: pick mart_dev tables and query intents (no Qdrant).

Reasoner-first with ontology-scoped prompts. Fail closed after retries, then
ontology fallback_plan last resort (never invent tables outside the catalog).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import (
    fallback_plan,
    reasoner_scope,
    resolve_measure,
)
from ml.rag.chatbot.analytical_bq_plan import (
    build_analytical_bq_plan,
)
from ml.rag.chatbot.bq_sql_patterns import normalize_pattern_name
from ml.rag.chatbot.bq_table_schema_yaml import (
    format_mart_reasoner_index,
    list_mart_table_index,
    pack_mart_table_hints,
)
from ml.rag.chatbot.fact_bq_plan import build_fact_bq_plan
from ml.rag.chatbot.plan_policy import model_for_plan
from ml.rag.chatbot.query_decomposer import (
    _AGRI_SCOPE_RE,
    _RANKING_SCOPE_RE,
    _extract_year_range,
    wants_africa_default_scope,
)
from ml.rag.chatbot.ontology_context import build_ontology_context
from ml.rag.chatbot.retrieval_contract import build_retrieval_contract, contract_to_bq_plan
from ml.rag.llm_chat import llm_chat_complete, llm_default_timeout_s, llm_model_id
from ml.rag.observability import observed_span, update_current_span_metadata

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TERM_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def _max_tables() -> int:
    try:
        return max(1, int(os.environ.get("RAG_BQ_MAX_TABLES", "6") or 6))
    except ValueError:
        return 6


def _reasoner_retries() -> int:
    try:
        return max(1, min(int(os.environ.get("RAG_BQ_REASONER_RETRIES", "3") or 3), 5))
    except ValueError:
        return 3


def _reasoner_model(plan_type: str | None) -> str:
    """Prefer dedicated reasoner model when set."""
    dedicated = os.environ.get("RAG_BQ_REASONER_MODEL_ID", "").strip()
    if dedicated:
        return dedicated
    return model_for_plan(plan_type) or llm_model_id()


def _known_table_ids() -> set[str]:
    return {str(r["table_id"]) for r in list_mart_table_index()}


def _slot_reasoner_active() -> bool:
    return os.environ.get("RAG_SLOT_REASONER", "").strip().lower() in ("1", "true", "yes", "on")


def _extract_json_obj(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _empty_plan(*, rationale: str) -> dict[str, Any]:
    return {
        "selected_tables": [],
        "query_intents": [],
        "skip_bq": True,
        "rationale": rationale,
        "table_hints": [],
        "index_truncated": False,
        "hints_truncated": False,
    }


def _query_terms_for_packing(query: str, decomposition: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    entities = decomposition.get("entities")
    if isinstance(entities, list):
        for ent in entities:
            t = str(ent).strip()
            if t:
                terms.append(t)
    for token in _TERM_TOKEN_RE.findall(query or ""):
        if len(token) >= 3:
            terms.append(token)
    return terms


def _normalize_intent_grain(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(g).strip() for g in raw if str(g).strip()]
    if isinstance(raw, str) and raw.strip():
        return [g.strip() for g in raw.split(",") if g.strip()]
    return []


def _normalize_intent(
    intent: dict[str, Any],
    *,
    known: set[str],
    selected: list[str],
) -> dict[str, Any]:
    intent_tables: list[str] = []
    for t in intent.get("tables") or []:
        tid = str(t).strip().split(".")[-1]
        if tid in known and tid not in intent_tables:
            intent_tables.append(tid)
    return {
        "goal": str(intent.get("goal") or "").strip() or "answer",
        "tables": intent_tables or selected[:1],
        "filters": str(intent.get("filters") or "").strip(),
        "notes": str(intent.get("notes") or "").strip(),
        "pattern": normalize_pattern_name(intent.get("pattern")),
        "metric": str(intent.get("metric") or "value").strip() or "value",
        "grain": _normalize_intent_grain(intent.get("grain")),
        "order_by": str(intent.get("order_by") or "").strip(),
    }


def _has_year_signal(query: str, decomposition: dict[str, Any]) -> bool:
    if decomposition.get("time_start") or decomposition.get("time_end"):
        return True
    ts, te = _extract_year_range(query or "")
    return bool(ts or te)


def _finalize_selected_plan(
    *,
    selected: list[str],
    intents: list[dict[str, Any]],
    rationale: str,
    query: str,
    decomposition: dict[str, Any],
    index_truncated: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not intents:
        intents = [
            {
                "goal": "answer_user_question",
                "tables": selected[:2],
                "filters": "",
                "notes": "",
                "pattern": "custom",
                "metric": "value",
                "grain": [],
                "order_by": "",
            }
        ]
    plan: dict[str, Any] = {
        "selected_tables": selected,
        "query_intents": intents,
        "skip_bq": False,
        "rationale": rationale,
    }
    if extra:
        plan.update(extra)
    hints, hints_truncated = pack_mart_table_hints(
        list(plan.get("selected_tables") or []),
        query_terms=_query_terms_for_packing(query, decomposition),
    )
    plan["table_hints"] = hints
    plan["index_truncated"] = index_truncated
    plan["hints_truncated"] = hints_truncated
    return plan


def _call_reasoner_llm(
    *,
    system: str,
    user: str,
    model: str,
    timeout: float,
) -> tuple[str, str]:
    """
    Return (raw_text, failure_cause).

    failure_cause is empty on success; otherwise timeout / empty_response / etc.
    Retries with short backoff are handled by the caller.
    """
    raw = llm_chat_complete(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=int(os.environ.get("RAG_BQ_REASONER_MAX_TOKENS", "800") or 800),
        temperature=0.0,
        timeout_s=timeout,
        purpose="bq.sql_reasoner",
    )
    if not raw:
        return "", "empty_or_soft_fail"
    return raw, ""


def reason_bq_sql_plan(
    query: str,
    *,
    decomposition: dict[str, Any] | None = None,
    plan_type: str | None = None,
    category: str | None = None,
    analytical_mode: bool = False,
    task_mode: str | None = None,
) -> dict[str, Any]:
    """
    Decide which mart_dev tables and SQL intents to run.

    Forced modes (analytical / fact / export) use ontology-aware builders.
    Otherwise: ontology-scoped LLM reasoner with retries; ontology fallback last.
    """
    known = _known_table_ids()
    dec = decomposition if isinstance(decomposition, dict) else {}
    if _slot_reasoner_active() and (dec.get("reasoner_job") or dec.get("matched_bundles")):
        return {
            "selected_tables": [],
            "query_intents": [],
            "skip_bq": True,
            "rationale": "slot_reasoner_owned",
        }
    max_tables = _max_tables()
    mode = (task_mode or ("analytical" if analytical_mode else "chat")).strip().lower()
    hit = resolve_measure(query, dec)
    scope = reasoner_scope(hit) if hit else None

    preferred_tables = set(scope["candidate_tables"]) if scope else set()
    iclasses = list(scope.get("indicator_classes") or []) if scope else []
    index_text, index_truncated = format_mart_reasoner_index(
        table_ids=list(preferred_tables) if preferred_tables else None,
        indicator_classes=iclasses or None,
    )

    if analytical_mode or mode == "analytical":
        analytical = build_analytical_bq_plan(query, decomposition=dec, known_tables=known)
        if analytical is not None:
            analytical["index_truncated"] = index_truncated
            if hit:
                analytical.setdefault("measure_id", hit.measure.id)
            update_current_span_metadata(
                {
                    "selected_tables": analytical.get("selected_tables"),
                    "skip_bq": False,
                    "analytical_mode": True,
                    "task_mode": "analytical",
                    "intent_count": len(analytical.get("query_intents") or []),
                    "rationale": analytical.get("rationale"),
                    "measure_id": analytical.get("measure_id"),
                }
            )
            return analytical

    # Entity/domain contract: multi-measure tables + intents before LLM freelancing.
    # Specialized builders (e.g. food_security) run only when that measure is activated.
    contract = build_retrieval_contract(query, decomposition=dec, known_tables=known)
    if contract.bq_tables and contract.bq_intents:
        plan = contract_to_bq_plan(
            contract,
            query=query,
            decomposition=dec,
            index_truncated=index_truncated,
        )
        if plan is not None and plan.get("selected_tables") and plan.get("query_intents"):
            update_current_span_metadata(
                {
                    "selected_tables": plan.get("selected_tables"),
                    "skip_bq": bool(plan.get("skip_bq")),
                    "task_mode": mode,
                    "intent_count": len(plan.get("query_intents") or []),
                    "rationale": plan.get("rationale"),
                    "measure_id": plan.get("measure_id"),
                    "primary_measures": plan.get("primary_measures"),
                    "corpus_domain_tags": plan.get("corpus_domain_tags"),
                }
            )
            # Soft opinion / corpus-only: allow skip_bq and continue without LLM pad.
            if plan.get("skip_bq"):
                return plan
            if plan.get("selected_tables") and plan.get("query_intents"):
                return plan

    if mode in ("fact_lookup", "data_export_only"):
        fact = build_fact_bq_plan(
            query,
            decomposition=dec,
            known_tables=known,
            task_mode=mode,
        )
        if fact is not None:
            fact["index_truncated"] = index_truncated
            if hit:
                fact.setdefault("measure_id", hit.measure.id)
            update_current_span_metadata(
                {
                    "selected_tables": fact.get("selected_tables"),
                    "skip_bq": False,
                    "task_mode": mode,
                    "intent_count": len(fact.get("query_intents") or []),
                    "rationale": fact.get("rationale"),
                    "measure_id": fact.get("measure_id"),
                }
            )
            return fact

    def _ontology_fallback(cause: str) -> dict[str, Any]:
        if hit is not None:
            fb = fallback_plan(
                hit,
                query=query,
                decomposition=dec,
                known_tables=known,
                task_mode=mode if mode != "chat" else hit.measure.default_task_mode,
            )
            if fb is not None:
                fb["index_truncated"] = index_truncated
                fb["rationale"] = f"ontology_fallback_after_{cause}"
                return fb
        heur = _heuristic_faostat_production_rank(query, decomposition=dec, known=known)
        if heur is not None:
            plan = _finalize_selected_plan(
                selected=list(heur["selected_tables"]),
                intents=list(heur["query_intents"]),
                rationale=str(heur.get("rationale") or "heuristic_africa_production_rank"),
                query=query,
                decomposition=dec,
                index_truncated=index_truncated,
            )
            return plan
        plan = _empty_plan(rationale=cause)
        plan["index_truncated"] = index_truncated
        return plan

    filter_hints = (scope or {}).get("filter_hints") or ""
    ontology = build_ontology_context(query, dec)
    ontology_block = ontology.to_reasoner_block()
    measure_line = ""
    if scope:
        measure_line = (
            f"Resolved measure: {scope.get('measure_id')} "
            f"(child={scope.get('child_measure_id')}). "
            f"Prefer ONLY these tables when possible: {scope.get('candidate_tables')}. "
            f"Filter hints: {filter_hints}. "
            f"crop_required={scope.get('crop_required')}; "
            f"country_is_answer={scope.get('country_is_answer')}; "
            f"recency_tier={scope.get('recency_tier')}.\n"
        )

    system = (
        "You are the OpenTrace BigQuery SQL planner for the mart_dev dataset only. "
        "OpenTrace is Africa-first. Use the measure ontology and indicator class hints — "
        "do NOT default every question to fct_production production_grain='physical'. "
        "Yield → fct_yield; exports → fct_trade; retail prices → fct_prices + price_source; "
        "IPC → fct_food_security measure_type; soil → fct_soil_health; climate → fct_climate. "
        "Select the minimum set of fct_* / agg_* tables (plus dim_* for joins when needed). "
        "Never invent table names outside the provided index. "
        "When selecting 2+ tables, use ONLY pairs listed in semantic_relationships "
        "(rels=) joins_with with explicit on= keys; never invent joins. "
        "Each query_intent should state goal, tables, filters, and join keys when multi-table. "
        "Set pattern to rank_by_sum, yoy_delta, share_of_total, or time_series when the "
        "question clearly matches; otherwise use custom. Include metric, grain, and "
        "order_by when using a non-custom pattern. "
        "If decomposition.africa_panel is true, plan a full African country panel "
        "(GROUP BY country_iso3), not a which-country clarify. "
        "If decomposition.africa_default is true, plan continental country rankings. "
        "If structured tables cannot help, set skip_bq=true. "
        "Respond with JSON only, no markdown."
    )
    user = (
        f"{measure_line}"
        f"{ontology_block}\n\n"
        f"Max tables: {max_tables}\n"
        f"Plan type: {plan_type or '-'}\n"
        f"Category: {category or '-'}\n"
        f"Task mode: {mode}\n"
        f"Decomposition: {json.dumps(dec, ensure_ascii=False)[:2000]}\n"
        f"Question: {query}\n\n"
        f"Mart table index (scoped when measure/class known):\n{index_text}\n\n"
        "Return JSON with keys:\n"
        '  "selected_tables": ["fct_..."],\n'
        '  "query_intents": [{"goal":"...","tables":["fct_..."],"filters":"...",'
        '"pattern":"rank_by_sum|yoy_delta|share_of_total|time_series|custom",'
        '"metric":"value","grain":["country_iso3"],"order_by":"total DESC","notes":"..."}],\n'
        '  "skip_bq": false,\n'
        '  "rationale": "short reason"\n'
    )

    with observed_span(
        "retrieval.bq.reason",
        input_data={
            "query": query[:200],
            "index_truncated": index_truncated,
            "measure_id": hit.measure.id if hit else None,
        },
    ):
        model = _reasoner_model(plan_type)
        timeout = float(os.environ.get("RAG_BQ_REASONER_TIMEOUT_S", "0") or 0) or llm_default_timeout_s()
        retries = _reasoner_retries()
        raw = ""
        last_cause = "reasoner_unavailable"
        parsed: dict[str, Any] | None = None

        for attempt in range(retries):
            raw, cause = _call_reasoner_llm(
                system=system, user=user, model=model, timeout=timeout
            )
            if cause:
                last_cause = cause if attempt == 0 else f"{cause}_retry{attempt}"
                time.sleep(min(0.4 * (2**attempt), 2.0))
                continue
            parsed = _extract_json_obj(raw)
            if parsed:
                break
            last_cause = "invalid_plan"
            time.sleep(min(0.4 * (2**attempt), 2.0))

        if not parsed:
            plan = _ontology_fallback(last_cause)
            update_current_span_metadata(
                {
                    "selected_tables": plan.get("selected_tables"),
                    "skip_bq": bool(plan.get("skip_bq")),
                    "index_truncated": index_truncated,
                    "reasoner_model": model,
                    "rationale": plan.get("rationale"),
                    "measure_id": hit.measure.id if hit else None,
                    "reasoner_failure": last_cause,
                }
            )
            return plan

        selected_raw = parsed.get("selected_tables")
        selected: list[str] = []
        if isinstance(selected_raw, list):
            for item in selected_raw:
                tid = str(item).strip().split(".")[-1]
                if tid in known and tid not in selected:
                    selected.append(tid)
                if len(selected) >= max_tables:
                    break

        intents_raw = parsed.get("query_intents")
        intents: list[dict[str, Any]] = []
        if isinstance(intents_raw, list):
            for intent in intents_raw[:max_tables]:
                if not isinstance(intent, dict):
                    continue
                intents.append(_normalize_intent(intent, known=known, selected=selected))
        skip = bool(parsed.get("skip_bq"))
        if not selected:
            plan = _ontology_fallback("skip_bq" if skip else "invalid_plan")
            update_current_span_metadata(
                {
                    "selected_tables": plan.get("selected_tables"),
                    "skip_bq": bool(plan.get("skip_bq")),
                    "index_truncated": index_truncated,
                    "reasoner_model": model,
                    "rationale": plan.get("rationale"),
                    "measure_id": hit.measure.id if hit else None,
                }
            )
            return plan

        plan = _finalize_selected_plan(
            selected=selected,
            intents=intents,
            rationale=str(parsed.get("rationale") or "").strip() or "llm",
            query=query,
            decomposition=dec,
            index_truncated=index_truncated,
            extra={"measure_id": hit.measure.id if hit else None},
        )
        update_current_span_metadata(
            {
                "selected_tables": list(plan.get("selected_tables") or []),
                "skip_bq": bool(plan.get("skip_bq")),
                "index_truncated": index_truncated,
                "hints_truncated": bool(plan.get("hints_truncated")),
                "reasoner_model": model,
                "rationale": plan.get("rationale"),
                "measure_id": plan.get("measure_id"),
            }
        )
        return plan


# Kept for tests that monkeypatch / import the old heuristic name.
def _heuristic_faostat_production_rank(
    query: str,
    *,
    decomposition: dict[str, Any],
    known: set[str],
) -> dict[str, Any] | None:
    """Deprecated production-rank heuristic — prefer ontology fallback."""
    if "fct_production" not in known:
        return None
    q = query or ""
    if not _RANKING_SCOPE_RE.search(q):
        return None
    africa_default = bool(decomposition.get("africa_default")) or wants_africa_default_scope(q)
    if not (_AGRI_SCOPE_RE.search(q) or africa_default):
        return None
    if not _has_year_signal(q, decomposition):
        return None
    hit = resolve_measure(q, decomposition)
    if hit is not None:
        return fallback_plan(
            hit, query=q, decomposition=decomposition, known_tables=known, task_mode="fact_lookup"
        )
    year_hint = str(decomposition.get("time_start") or "")[:4] or "year from question"
    want_yield = bool(re.search(r"\byields?\b", q, re.IGNORECASE))
    grain = "physical"
    table = "fct_production"
    if want_yield:
        table = "fct_yield"
        grain = "season_key/harvest_year"
    return {
        "selected_tables": [table],
        "query_intents": [
            {
                "goal": f"Africa country {'yield' if want_yield else 'production'} ranking",
                "tables": [table],
                "filters": (
                    f"{'production_grain=' + repr(grain) + '; ' if not want_yield else ''}"
                    f"year≈{year_hint}; Africa continental (country_iso3)"
                ),
                "notes": "heuristic_africa_rank",
                "pattern": "rank_by_sum",
                "metric": "value",
                "grain": ["country_iso3"],
                "order_by": "total DESC",
            }
        ],
        "skip_bq": False,
        "rationale": "heuristic_africa_rank",
    }
