"""Plan and track evidence for explicit country-level, whole-year comparisons."""
from __future__ import annotations

import re
from math import isfinite
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import MEASURES
from ml.rag.chatbot.agri_entities import CROP_ENTITY_RE
from ml.rag.chatbot.bq_context_enrich import table_from_sql
from ml.rag.chatbot.bq_execute_state import EVIDENCE_STATUSES, execution_status
from ml.rag.chatbot.bq_table_schema_yaml import load_mart_table_schema
from ml.rag.chatbot.class_engine_runner import engine_results_to_bq_plan
from ml.rag.chatbot.class_engines.shared import build_planned_engine_result
from ml.rag.chatbot.geo_iso3 import name_to_iso3
from ml.rag.chatbot.turn_contract import TurnContract


_EMPLOYMENT_SHARE = re.compile(
    r"\b(?:agricultural employment share|(?:share|percentage|proportion) of (?:total )?"
    r"employment (?:in|from) agriculture)\b",
    re.I,
)


def is_fact_comparison(turn: TurnContract) -> bool:
    windows = turn.time_spec.windows
    return (
        turn.job == "compare" and turn.geo_grain == "country" and bool(turn.geo)
        and bool(turn.facts) and bool(windows) and (len(windows) >= 2 or len(turn.geo) >= 2) and not turn.breakdown
        and all(name_to_iso3(geo) for geo in turn.geo)
        and all(
            w.strict and not w.forecast and not w.from_report
            and (w.start.month, w.start.day) == (1, 1) and (w.end.month, w.end.day) == (12, 31)
            for w in windows
        )
    )


def plan_fact_comparison(
    turn: TurnContract, *, query: str, facets: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind one serving table per requirement, with the same explicit scope."""
    if not is_fact_comparison(turn):
        return None
    windows = turn.time_spec.windows
    tables: dict[str, str] = {}
    periods = [w.as_dict() for w in windows]
    start = min(w.start for w in windows).isoformat()
    end = max(w.end for w in windows).isoformat()
    crops = list(dict.fromkeys(match.group().lower() for match in CROP_ENTITY_RE.finditer(query)))
    results = []
    for fact in turn.facts:
        spec = MEASURES.get(fact.measure_id)
        if spec is None or not spec.indicator_classes or (spec.crop_required and len(crops) != 1):
            continue
        if spec.id == "employment_share" and not _EMPLOYMENT_SHARE.search(query):
            continue
        # The national aggregate stores yield as t/ha. fct_yield is subnational and seasonal.
        candidates = (
            ("agg_production_country_year",) if spec.id == "yield" else spec.candidate_tables
        )
        for table_id in candidates:
            if table_id in tables.values() or not load_mart_table_schema(table_id):
                continue
            result = build_planned_engine_result(
                class_code=spec.indicator_classes[0], table_id=table_id,
                query=f"{query} yield_recomputed" if spec.id == "yield" else query,
                facets={
                    **facets, "primary_measures": [fact.measure_id],
                    "geography": turn.geo, "entities": turn.entities,
                    "time_start": start, "time_end": end,
                    "time_windows": periods, "comparison_windows": periods,
                },
                card={}, iso_list=turn.geo, measure_id=fact.measure_id,
                value_hits={"product_name": crops} if spec.crop_required else None,
            )
            bind = result.bind_contract
            if not bind or not bind.get("time_sql") or "geo" not in bind["filter_provenance"]:
                continue
            if spec.crop_required and not bind.get("product_literals"):
                continue
            metric_columns = {column for column, _ in bind["measure_filters"]}
            if spec.id == "climate" and "metric" not in metric_columns:
                continue
            if spec.id == "food_security_ipc" and not {"metric", "scenario_name"} <= metric_columns:
                continue
            if spec.id == "yield" and bind.get("measure_columns") != ["yield_recomputed"]:
                continue
            if spec.id == "employment_share" and not {"indicator", "sex", "unit"} <= metric_columns:
                continue
            tables[fact.measure_id] = table_id
            results.append(result)
            break
    return {
        **engine_results_to_bq_plan(results, rationale="explicit_fact_comparison"),
        "fact_tables": tables, "nl2sql_fallback": False,
        "max_sql_queries": len(tables),
    }


def _usable_observation(raw: dict[str, Any], bind: dict[str, Any]) -> bool:
    columns = ["value", *bind.get("measure_columns", [])]
    value = next((raw[col] for col in columns if col in raw), None)
    try:
        if isinstance(value, bool) or not isfinite(float(value)):
            return False
    except (TypeError, ValueError, OverflowError):
        return False

    for column, expected in bind.get("measure_filters", []):
        if column in raw and str(raw[column]).strip().casefold() != str(expected).strip().casefold():
            return False
    product = bind.get("product_column")
    products = bind.get("product_literals") or []
    label = raw.get("product_name")
    if label is None and isinstance(product, str) and not product.endswith("_key"):
        label = raw.get(product)
    if label is not None and products:
        return str(label).strip().casefold() in {str(p).strip().casefold() for p in products}
    return True


def _bound_observation(item: dict[str, Any], binds: dict[str, Any]) -> tuple[str, str, int] | None:
    meta = item.get("metadata")
    if not isinstance(meta, dict):
        return None
    table = str(meta.get("table_id") or "").split(".")[-1]
    bind = binds.get(table)
    raw = meta.get("raw_row")
    status = execution_status(str(meta.get("status") or ""))
    if not isinstance(raw, dict) or not bind or status not in EVIDENCE_STATUSES or not _usable_observation(raw, bind):
        return None
    country = name_to_iso3(str(raw.get("country_iso3") or raw.get("country_name") or ""))
    if not country or any(raw.get(key) for key in ("admin_1", "admin_2", "market_name")):
        return None
    time_column = bind.get("time_column")
    time_value = raw.get("year") or raw.get("harvest_year") or raw.get(time_column)
    if time_column == "as_of_date" and raw.get("as_of_date_basis") == "loaded_at" and not (raw.get("year") or raw.get("harvest_year")):
        return None
    year = str(time_value or "")[:4]
    if len(year) == 4 and year.isdecimal():
        return table, country, int(year)
    return None


def matches_bound_fact(item: dict[str, Any], turn: TurnContract, plan: dict[str, Any]) -> bool:
    observed = _bound_observation(item, plan.get("bind_contracts") or {})
    if observed is None:
        return False
    table, country, year = observed
    return (
        table in (plan.get("fact_tables") or {}).values()
        and country in {name_to_iso3(geo) for geo in turn.geo}
        and any(period.start.year <= year <= period.end.year for period in turn.time_spec.windows)
    )


def fact_requirement_coverage(
    turn: TurnContract,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    debug: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate the bound indicator at each requested country and year window."""
    tables = plan.get("fact_tables") or {}
    binds = plan.get("bind_contracts") or {}
    attempts: dict[str, set[str]] = {}
    for entry in debug:
        table = str(entry.get("table_id") or table_from_sql(str(entry.get("sql") or ""))).split(".")[-1]
        status = str(entry.get("status") or "unknown")
        attempts.setdefault(table, set()).add(status)

    observations: dict[tuple[str, str], set[int]] = {}
    for item in rows:
        if observation := _bound_observation(item, binds):
            table, country, year = observation
            observations.setdefault((table, country), set()).add(year)

    coverage: list[dict[str, Any]] = []
    for fact in turn.facts:
        table = tables.get(fact.measure_id, "")
        raw_statuses = (attempts.get(table) or attempts.get("") or set()) - {"unknown", "planned_not_executed"}
        statuses = {execution_status(status) for status in raw_statuses}
        if not table or table not in binds:
            missing_status = "unsupported"
        elif "timeout" in statuses:
            missing_status = "timed_out"
        elif statuses & {"exec_failed", "result_too_large"}:
            missing_status = "unavailable"
        elif "validation_failed" in statuses:
            missing_status = "rejected"
        elif statuses & {"ok", "empty_ok", "partial"}:
            missing_status = "no_match"
        else:
            missing_status = "unavailable"

        for geo in turn.geo:
            observed = observations.get((table, name_to_iso3(geo) or ""), set())
            for period in turn.time_spec.windows:
                years = set(range(period.start.year, period.end.year + 1))
                found = years & observed
                coverage.append({
                    "measure": fact.measure_id,
                    "geo": geo,
                    "grain": turn.geo_grain,
                    "start": period.start.isoformat(),
                    "end": period.end.isoformat(),
                    "status": "satisfied" if found == years else missing_status,
                    "observed_years": sorted(found),
                    "missing_years": sorted(years - found),
                })
    return coverage
