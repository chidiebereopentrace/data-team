"""YAML-index SQL reasoner: pick staging_dev tables and query intents (no Qdrant).

Fail closed: if the LLM is unavailable or returns an invalid plan, skip BQ
(no heuristic table inventing) — except Africa-default production rankings.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from ml.rag.chatbot.bq_sql_patterns import normalize_pattern_name
from ml.rag.chatbot.bq_table_schema_yaml import (
    format_reasoner_index,
    list_staging_table_index,
    pack_selected_table_hints,
)
from ml.rag.chatbot.plan_policy import model_for_plan
from ml.rag.chatbot.query_decomposer import (
    _AGRI_SCOPE_RE,
    _RANKING_SCOPE_RE,
    _extract_year_range,
    wants_africa_default_scope,
)
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


def _reasoner_model(plan_type: str | None) -> str:
    """Same chat model as the rest of the pipeline (8B by default)."""
    dedicated = os.environ.get("RAG_BQ_REASONER_MODEL_ID", "").strip()
    if dedicated:
        return dedicated
    return model_for_plan(plan_type) or llm_model_id()


def _known_table_ids() -> set[str]:
    return {str(r["table_id"]) for r in list_staging_table_index()}


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
    """Entity + question tokens used to prefer matching FAOSTAT enum samples."""
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


def _heuristic_faostat_production_rank(
    query: str,
    *,
    decomposition: dict[str, Any],
    known: set[str],
) -> dict[str, Any] | None:
    """Deterministic Africa production ranking when LLM would skip."""
    if "stg_faostat_production" not in known:
        return None
    q = query or ""
    if not _RANKING_SCOPE_RE.search(q):
        return None
    africa_default = bool(decomposition.get("africa_default")) or wants_africa_default_scope(q)
    if not (_AGRI_SCOPE_RE.search(q) or africa_default):
        return None
    if not _has_year_signal(q, decomposition):
        return None
    year_hint = str(decomposition.get("time_start") or "")[:4] or "year from question"
    return {
        "selected_tables": ["stg_faostat_production"],
        "query_intents": [
            {
                "goal": "Africa country production ranking",
                "tables": ["stg_faostat_production"],
                "filters": (
                    f"element='Production'; year≈{year_hint}; "
                    "Africa continental (no geo dim)"
                ),
                "notes": "heuristic_africa_production_rank",
                "pattern": "rank_by_sum",
                "metric": "value",
                "grain": ["country_name"],
                "order_by": "total DESC",
            }
        ],
        "skip_bq": False,
        "rationale": "heuristic_africa_production_rank",
    }


def _finalize_selected_plan(
    *,
    selected: list[str],
    intents: list[dict[str, Any]],
    rationale: str,
    query: str,
    decomposition: dict[str, Any],
    index_truncated: bool,
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
    hints, hints_truncated = pack_selected_table_hints(
        list(plan.get("selected_tables") or []),
        query_terms=_query_terms_for_packing(query, decomposition),
    )
    plan["table_hints"] = hints
    plan["index_truncated"] = index_truncated
    plan["hints_truncated"] = hints_truncated
    return plan


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
    Decide which staging_dev tables and SQL intents to run.

    Fail closed on LLM failure / invalid JSON / no valid tables in index,
    except a deterministic FAOSTAT production-rank heuristic for Africa-default
    which-country questions, and forced plans for analytical / fact / export-only modes.
    """
    known = _known_table_ids()
    index_text, index_truncated = format_reasoner_index()
    dec = decomposition if isinstance(decomposition, dict) else {}
    max_tables = _max_tables()
    mode = (task_mode or ("analytical" if analytical_mode else "chat")).strip().lower()

    if analytical_mode or mode == "analytical":
        from ml.rag.chatbot.analytical_bq_plan import build_analytical_bq_plan

        analytical = build_analytical_bq_plan(query, decomposition=dec, known_tables=known)
        if analytical is not None:
            analytical["index_truncated"] = index_truncated
            update_current_span_metadata(
                {
                    "selected_tables": analytical.get("selected_tables"),
                    "skip_bq": False,
                    "analytical_mode": True,
                    "task_mode": "analytical",
                    "intent_count": len(analytical.get("query_intents") or []),
                    "rationale": analytical.get("rationale"),
                }
            )
            return analytical

    if mode in ("fact_lookup", "data_export_only"):
        from ml.rag.chatbot.fact_bq_plan import build_fact_bq_plan

        fact = build_fact_bq_plan(
            query,
            decomposition=dec,
            known_tables=known,
            task_mode=mode,
        )
        if fact is not None:
            fact["index_truncated"] = index_truncated
            update_current_span_metadata(
                {
                    "selected_tables": fact.get("selected_tables"),
                    "skip_bq": False,
                    "task_mode": mode,
                    "intent_count": len(fact.get("query_intents") or []),
                    "rationale": fact.get("rationale"),
                }
            )
            return fact

    heuristic = _heuristic_faostat_production_rank(query, decomposition=dec, known=known)

    system = (
        "You are the OpenTrace BigQuery SQL planner for the staging_dev dataset only. "
        "OpenTrace is Africa-first: unscoped which-country / ranking questions about "
        "agriculture, production, or agricultural activity default to African "
        "stg_faostat_production rankings (element='Production'), not global web trivia. "
        "Select the minimum set of stg_* tables that fully answers the question — "
        "one table when sufficient, more when the question requires enrichment or "
        "comparison across datasets. "
        "Never invent table names outside the provided index. "
        "When selecting 2+ tables, use ONLY pairs listed in semantic_relationships "
        "(rels=) joins_with with explicit on= keys; never invent joins. "
        "Each query_intent should state goal, tables, filters, and join keys when multi-table. "
        "Set pattern to rank_by_sum, yoy_delta, share_of_total, or time_series when the "
        "question clearly matches; otherwise use custom. Include metric, grain, and "
        "order_by when using a non-custom pattern. "
        "Do not add macro tables (GDP, HDI) unless the question asks for macro context "
        "or a documented join requires them. "
        "Respect geography and time constraints from the decomposition. "
        "If decomposition.africa_default is true, prefer stg_faostat_production and "
        "set skip_bq=false for production/agricultural ranking questions. "
        "If structured tables cannot help, set skip_bq=true. "
        "Respond with JSON only, no markdown."
    )
    user = (
        f"Max tables: {max_tables}\n"
        f"Plan type: {plan_type or '-'}\n"
        f"Category: {category or '-'}\n"
        f"Decomposition: {json.dumps(dec, ensure_ascii=False)[:2000]}\n"
        f"Question: {query}\n\n"
        f"Staging table index:\n{index_text}\n\n"
        "Return JSON with keys:\n"
        '  "selected_tables": ["stg_..."],\n'
        '  "query_intents": [{"goal":"...","tables":["stg_..."],"filters":"...",'
        '"pattern":"rank_by_sum|yoy_delta|share_of_total|time_series|custom",'
        '"metric":"value","grain":["country_name"],"order_by":"total DESC","notes":"..."}],\n'
        '  "skip_bq": false,\n'
        '  "rationale": "short reason"\n'
    )

    with observed_span(
        "retrieval.bq.reason",
        input_data={"query": query[:200], "index_truncated": index_truncated},
    ):
        model = _reasoner_model(plan_type)
        timeout = float(os.environ.get("RAG_BQ_REASONER_TIMEOUT_S", "0") or 0) or llm_default_timeout_s()
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
        parsed = _extract_json_obj(raw) if raw else None

        def _use_heuristic() -> dict[str, Any]:
            assert heuristic is not None
            plan = _finalize_selected_plan(
                selected=list(heuristic["selected_tables"]),
                intents=list(heuristic["query_intents"]),
                rationale=str(heuristic["rationale"]),
                query=query,
                decomposition=dec,
                index_truncated=index_truncated,
            )
            update_current_span_metadata(
                {
                    "selected_tables": plan["selected_tables"],
                    "skip_bq": False,
                    "index_truncated": index_truncated,
                    "hints_truncated": plan.get("hints_truncated"),
                    "reasoner_model": model,
                    "rationale": plan["rationale"],
                }
            )
            return plan

        if not parsed:
            if heuristic:
                return _use_heuristic()
            plan = _empty_plan(rationale="reasoner_unavailable" if not raw else "invalid_plan")
            plan["index_truncated"] = index_truncated
            update_current_span_metadata(
                {
                    "selected_tables": [],
                    "skip_bq": True,
                    "index_truncated": index_truncated,
                    "hints_truncated": False,
                    "reasoner_model": model,
                    "rationale": plan["rationale"],
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
            if heuristic:
                return _use_heuristic()
            plan = _empty_plan(
                rationale="skip_bq" if skip else "invalid_plan",
            )
            plan["index_truncated"] = index_truncated
            if skip and str(parsed.get("rationale") or "").strip():
                plan["rationale"] = str(parsed.get("rationale")).strip()
            update_current_span_metadata(
                {
                    "selected_tables": [],
                    "skip_bq": True,
                    "index_truncated": index_truncated,
                    "hints_truncated": False,
                    "reasoner_model": model,
                    "rationale": plan["rationale"],
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
        )
        update_current_span_metadata(
            {
                "selected_tables": list(plan.get("selected_tables") or []),
                "skip_bq": bool(plan.get("skip_bq")),
                "index_truncated": index_truncated,
                "hints_truncated": bool(plan.get("hints_truncated")),
                "reasoner_model": model,
                "rationale": plan.get("rationale"),
            }
        )
        return plan
