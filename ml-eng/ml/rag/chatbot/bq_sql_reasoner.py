"""YAML-index SQL reasoner: pick staging_dev tables and query intents (no Qdrant).

Fail closed: if the LLM is unavailable or returns an invalid plan, skip BQ
(no heuristic table inventing).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import (
    format_reasoner_index,
    list_staging_table_index,
    pack_selected_table_hints,
)
from ml.rag.chatbot.plan_policy import model_for_plan
from ml.rag.llm_chat import llm_chat_complete, llm_default_timeout_s, llm_model_id
from ml.rag.observability import observed_span, update_current_span_metadata

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _max_tables() -> int:
    try:
        return max(1, int(os.environ.get("RAG_BQ_MAX_TABLES", "4") or 4))
    except ValueError:
        return 4


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


def reason_bq_sql_plan(
    query: str,
    *,
    decomposition: dict[str, Any] | None = None,
    plan_type: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """
    Decide which staging_dev tables and SQL intents to run.

    Fail closed on LLM failure / invalid JSON / no valid tables in index.
    """
    known = _known_table_ids()
    index_text, index_truncated = format_reasoner_index()
    dec = decomposition if isinstance(decomposition, dict) else {}
    max_tables = _max_tables()

    system = (
        "You are the OpenTrace BigQuery SQL planner for the staging_dev dataset only. "
        "Select the minimum set of stg_* tables that fully answers the question — "
        "one table when sufficient, more when the question requires enrichment or "
        "comparison across datasets. "
        "Never invent table names outside the provided index. "
        "When selecting 2+ tables, use ONLY pairs listed in semantic_relationships "
        "(rels=) joins_with with explicit on= keys; never invent joins. "
        "Each query_intent should state goal, tables, filters, and join keys when multi-table. "
        "Do not add macro tables (GDP, HDI) unless the question asks for macro context "
        "or a documented join requires them. "
        "Respect geography and time constraints from the decomposition. "
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
        '  "query_intents": [{"goal":"...","tables":["stg_..."],"filters":"...","notes":"..."}],\n'
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
        if not parsed:
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
                intent_tables = []
                for t in intent.get("tables") or []:
                    tid = str(t).strip().split(".")[-1]
                    if tid in known and tid not in intent_tables:
                        intent_tables.append(tid)
                intents.append(
                    {
                        "goal": str(intent.get("goal") or "").strip() or "answer",
                        "tables": intent_tables or selected[:1],
                        "filters": str(intent.get("filters") or "").strip(),
                        "notes": str(intent.get("notes") or "").strip(),
                    }
                )
        skip = bool(parsed.get("skip_bq"))
        if not selected:
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

        if not intents:
            intents = [
                {
                    "goal": "answer_user_question",
                    "tables": selected[:2],
                    "filters": "",
                    "notes": "",
                }
            ]
        plan = {
            "selected_tables": selected,
            "query_intents": intents,
            "skip_bq": False,
            "rationale": str(parsed.get("rationale") or "").strip() or "llm",
        }

        hints, hints_truncated = pack_selected_table_hints(list(plan.get("selected_tables") or []))
        plan["table_hints"] = hints
        plan["index_truncated"] = index_truncated
        plan["hints_truncated"] = hints_truncated
        update_current_span_metadata(
            {
                "selected_tables": list(plan.get("selected_tables") or []),
                "skip_bq": bool(plan.get("skip_bq")),
                "index_truncated": index_truncated,
                "hints_truncated": hints_truncated,
                "reasoner_model": model,
            }
        )
        return plan
