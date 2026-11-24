"""Plan and track independently scoped country-level annual fact requirements."""
from __future__ import annotations

import re
from decimal import Decimal
from math import isfinite
from typing import Any

from ml.rag.chatbot.agri_measure_ontology import MEASURES
from ml.rag.chatbot.bq_context_enrich import table_from_sql
from ml.rag.chatbot.bq_execute_state import EVIDENCE_STATUSES, execution_status
from ml.rag.chatbot.bq_table_schema_yaml import load_mart_table_schema, match_product_samples, product_group_prefix
from ml.rag.chatbot.class_engine_runner import engine_results_to_bq_plan
from ml.rag.chatbot.class_engines.shared import build_planned_engine_result
from ml.rag.chatbot.geo_iso3 import name_to_iso3
from ml.rag.chatbot.turn_contract import TurnContract, fact_scope

_EMPLOYMENT_SHARE = re.compile(
    r"\b(?:agricultural employment share|agriculture['’]s (?:share|percentage|proportion) of (?:total )?employment|"
    r"(?:share|percentage|proportion) of (?:total )?"
    r"employment (?:in|from) agriculture)\b",
    re.I,
)
_OPEN_MEMBER_SCOPE = re.compile(
    r"\b(?:crops?|products?|commodities|cultures?|produits?|agricultural (?:imports?|exports?|trade))\b", re.I,
)


def is_fact_comparison(turn: TurnContract) -> bool:
    return turn.job == "compare" and _has_annual_fact_scope(turn)


def _has_annual_fact_scope(turn: TurnContract) -> bool:
    scopes = [fact_scope(turn, fact) for fact in turn.facts]
    return (
        turn.job not in ("help", "social", "clarify")
        and turn.geo_grain == "country" and bool(scopes) and not (set(turn.breakdown) - {"product"})
        # Price observations have market/package/currency grain and use the
        # bounded price-series compiler, not the annual national-row compiler.
        and all(fact.measure_id != "market_price" for fact in turn.facts)
        and all(scope.geo and all(name_to_iso3(geo) for geo in scope.geo) and scope.time_spec.windows for scope in scopes)
        and all(
            w.strict and not w.forecast and not w.from_report
            and (w.start.month, w.start.day) == (1, 1) and (w.end.month, w.end.day) == (12, 31)
            for scope in scopes for w in scope.time_spec.windows
        )
    )


def plan_fact_requirements(
    turn: TurnContract, *, query: str, facets: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind each requirement separately, including different uses of the same table."""
    if not _has_annual_fact_scope(turn):
        return None
    tables: dict[str, str] = {}
    bindings: dict[str, Any] = {}
    results = []
    for fact in turn.facts:
        scope = fact_scope(turn, fact)
        question = fact.question or query
        windows = scope.time_spec.windows
        periods = [w.as_dict() for w in windows]
        start = min(w.start for w in windows).isoformat()
        end = max(w.end for w in windows).isoformat()
        spec = MEASURES.get(fact.measure_id)
        if spec is None or not spec.indicator_classes:
            continue
        if spec.id == "socio_economic":
            # A broad domain or the word "population" does not request HDI.
            # Only an explicitly identified indicator may bind a macro series.
            continue
        if spec.id == "employment_share" and not _EMPLOYMENT_SHARE.search(query):
            continue
        # The national aggregate stores yield as t/ha. fct_yield is subnational and seasonal.
        candidates = (
            ("agg_production_country_year",) if spec.id in ("yield", "production") else spec.candidate_tables
        )
        for table_id in candidates:
            if not load_mart_table_schema(table_id):
                continue
            member_query = fact.member_query or question
            resolved_products = match_product_samples(table_id, member_query) if spec.crop_required else []
            if fact.enumerate_members:
                # "List all crops" requests members, not the dimension's one
                # aggregate label "Crops" alongside its constituent products.
                resolved_products = [label for label in resolved_products if label.casefold() not in ("crop", "crops")]
            group_prefix = product_group_prefix(member_query) if spec.crop_required and not resolved_products else ""
            discovery = bool(spec.crop_required and fact.enumerate_members and not resolved_products
                             and (group_prefix or _OPEN_MEMBER_SCOPE.search(member_query)))
            if spec.crop_required and not resolved_products and not discovery:
                continue
            bind_query = f"{spec.id} quantities" if discovery else member_query
            result = build_planned_engine_result(
                class_code=spec.indicator_classes[0], table_id=table_id,
                query=(f"{bind_query} yield_recomputed" if spec.id == "yield" else bind_query),
                facets={
                    **facets, "primary_measures": [fact.measure_id],
                    "geography": scope.geo, "entities": [] if discovery else scope.entities,
                    "entity_roles": {} if discovery else facets.get("entity_roles", {}),
                    "product_discovery": discovery,
                    "product_group_prefix": group_prefix,
                    "time_start": start, "time_end": end,
                    "time_windows": periods, "comparison_windows": periods,
                },
                card={}, iso_list=scope.geo, measure_id=fact.measure_id,
                value_hits={"product_name": resolved_products} if spec.crop_required else None,
            )
            bind = result.bind_contract
            if not bind or not bind.get("time_sql") or "geo" not in bind["filter_provenance"]:
                continue
            if spec.crop_required and not bind.get("product_literals") and not discovery:
                continue
            if spec.id == "production" and table_id == "agg_production_country_year":
                bind["measure_columns"] = ["production_qty"]
            metric_columns = {column for column, _ in bind["measure_filters"]}
            if spec.id == "climate" and "metric" not in metric_columns:
                continue
            if spec.id == "food_security_ipc" and not {"metric", "scenario_name"} <= metric_columns:
                continue
            if spec.id == "yield" and bind.get("measure_columns") != ["yield_recomputed"]:
                continue
            if spec.id in ("employment_share", "labour_productivity") and not {"indicator", "sex", "unit"} <= metric_columns:
                continue
            bind["requirement_id"] = fact.key
            bind["enumerate_members"] = fact.enumerate_members
            bind["member_selection"] = "discover" if discovery else "explicit"
            bind["request_scope"] = {
                "question": question, "geography": scope.geo, "entities": scope.entities,
                "time_start": start, "time_end": end, "time_windows": periods,
                "primary_measures": [fact.measure_id],
            }
            bindings[fact.key] = bind
            tables[fact.key] = table_id
            results.append(result)
            break
    return {
        **engine_results_to_bq_plan(results, rationale="scoped_fact_requirements"),
        "bind_contracts": bindings, "fact_tables": tables, "nl2sql_fallback": False,
        "max_sql_queries": len(tables),
    }


def _usable_observation(raw: dict[str, Any], bind: dict[str, Any]) -> bool:
    columns = ["value", *bind.get("measure_columns", [])]
    value = next((raw[col] for col in columns if col in raw), None)
    try:
        if value is None or isinstance(value, bool) or not isfinite(float(value)):
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


def _bound_observation(item: dict[str, Any], binds: dict[str, Any]) -> tuple[str, str, int, str] | None:
    meta = item.get("metadata")
    if not isinstance(meta, dict):
        return None
    table = str(meta.get("table_id") or "").split(".")[-1]
    key = str(meta.get("requirement_id") or "")
    if not key:
        candidates = [key for key, value in binds.items() if value.get("table_id", key) == table]
        if len(candidates) != 1:
            return None
        key = candidates[0]
    bind = binds.get(key)
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
        scope = bind.get("request_scope") or {}
        allowed = {name_to_iso3(place) for place in scope.get("geography", [])}
        if allowed and country not in allowed:
            return None
        if scope.get("time_start") and not int(scope["time_start"][:4]) <= int(year) <= int(scope["time_end"][:4]):
            return None
        member = str(raw.get("product_name") or raw.get("item_name") or "").strip()
        if bind.get("enumerate_members") and bind.get("product_column") and not member:
            return None
        return key, country, int(year), member
    return None


def matches_bound_fact(item: dict[str, Any], turn: TurnContract, plan: dict[str, Any]) -> bool:
    observed = _bound_observation(item, plan.get("bind_contracts") or {})
    if observed is None:
        return False
    key, country, year, _ = observed
    for fact in turn.facts:
        scope = fact_scope(turn, fact)
        if key in (fact.key, (plan.get("fact_tables") or {}).get(fact.key)):
            return (country in {name_to_iso3(geo) for geo in scope.geo}
                    and any(period.start.year <= year <= period.end.year for period in scope.time_spec.windows))
    return False


def prioritize_bound_facts(
    rows: list[dict[str, Any]], turn: TurnContract, plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep one observation for each requested series, country and period ahead of extras."""
    binds = plan.get("bind_contracts") or {}
    seen: set[tuple[str, str, int, str]] = set()
    first, extra = [], []
    for item in rows:
        observation = _bound_observation(item, binds)
        if observation is None:
            continue
        key, country, year, member = observation
        periods = set()
        for fact in turn.facts:
            scope = fact_scope(turn, fact)
            if key not in (fact.key, (plan.get("fact_tables") or {}).get(fact.key)):
                continue
            if country not in {name_to_iso3(geo) for geo in scope.geo}:
                continue
            periods.update((key, country, i, member) for i, window in enumerate(scope.time_spec.windows)
                           if window.start.year <= year <= window.end.year)
        if not periods:
            continue
        (first if periods - seen else extra).append(item)
        seen.update(periods)
    return [*first, *extra]


def render_verified_facts(
    turn: TurnContract, plan: dict[str, Any], sources: list[tuple[int, dict[str, Any]]],
    *, coverage: list[dict[str, Any]] | None = None, endpoints_only: bool = False,
) -> str:
    """Render annual observations and source-consistent endpoint changes independently of the model."""
    binds = plan.get("bind_contracts") or {}
    if any(bind.get("enumerate_members") for bind in binds.values()):
        return _render_member_facts(turn, plan, sources, coverage=coverage)
    values: dict[tuple[str, str, int], set[tuple[Decimal, str, str]]] = {}
    citations: dict[tuple[str, str, int], int] = {}
    for source_id, item in sources:
        observation = _bound_observation(item, binds)
        if observation is None:
            continue
        key, _, _, _ = observation
        bind = binds[key]
        meta = item["metadata"]
        raw = meta["raw_row"]
        value = next((raw[col] for col in ["value", *bind.get("measure_columns", [])] if col in raw), None)
        schema = load_mart_table_schema(str(meta.get("table_id") or "")) or {}
        column_units = schema.get("column_units") or {}
        unit = str(raw.get("unit") or (meta.get("value_semantics") or {}).get("unit")
                   or dict(bind.get("measure_filters") or []).get("unit")
                   or next((column_units[col] for col in bind.get("measure_columns", []) if col in column_units), ""))
        if not unit:
            continue
        source = str(raw.get("source_key") or raw.get("organisation_name") or meta.get("organisation_name") or "")
        # A single-member comparison has one label fixed by its binding.
        key_without_member = observation[:3]
        values.setdefault(key_without_member, set()).add((Decimal(str(value)), source, unit))
        citations.setdefault(key_without_member, source_id)
    if not values:
        return ""

    lines = ["| Country | Indicator | Verified observations | Change |", "|---|---|---|---|"]
    notes = []
    for fact in turn.facts:
        scope = fact_scope(turn, fact)
        key = fact.key if fact.key in binds else (plan.get("fact_tables") or {}).get(fact.key, "")
        bind = binds.get(key) or {}
        label = {"yield": "Crop yield", "employment_share": "Agricultural employment share",
                 "labour_productivity": "Agriculture value added per worker (constant 2015 US$)"}.get(
            fact.measure_id, MEASURES[fact.measure_id].aliases[0] if fact.measure_id in MEASURES else "Requested indicator",
        )
        products = bind.get("product_literals") or []
        if len(products) == 1:
            label = f"{products[0]} {label.lower()}"
        for geo in scope.geo:
            country = name_to_iso3(geo) or ""
            years = {year for window in scope.time_spec.windows for year in range(window.start.year, window.end.year + 1)}
            observed = []
            for year in sorted(years):
                observation = (key, country, year)
                candidates = values.get(observation, set())
                if len(candidates) == 1:
                    value, source, unit = next(iter(candidates))
                    observed.append((year, value, source, unit, citations[observation]))
            if not observed:
                continue
            display = [observed[0], observed[-1]] if endpoints_only and len(observed) > 2 else observed
            cells = [f"{year}: {value:,.2f} {unit} [{cite}]" for year, value, _, unit, cite in display]
            change = "—"
            first, last = observed[0], observed[-1]
            # No blending publishers, units, scenarios or unrelated denominator definitions.
            if len(observed) > 1 and first[2] and first[2:4] == last[2:4]:
                delta = last[1].quantize(Decimal("0.01")) - first[1].quantize(Decimal("0.01"))
                unit = "percentage points" if first[3] == "%" else first[3]
                change = f"{delta:+,.2f} {unit} ({first[0]}–{last[0]})"
            lines.append(f"| {geo} | {label} | {'; '.join(cells)} | {change} |")
            missing = sorted(years - {row[0] for row in observed})
            if missing and coverage is None:
                notes.append(f"{geo}, {label.lower()}: no figures available for {', '.join(map(str, missing))}.")
    if len(lines) == 2:
        return ""
    if coverage is not None:
        notes.extend(fact_gap_description(gap) for gap in coverage if gap["status"] != "satisfied")
    return "\n".join([*lines, "", *notes]).strip()


def _render_member_facts(
    turn: TurnContract, plan: dict[str, Any], sources: list[tuple[int, dict[str, Any]]],
    *, coverage: list[dict[str, Any]] | None,
) -> str:
    """Itemized presentation of the same validated observations, never a cross-product total."""
    binds = plan.get("bind_contracts") or {}
    facts = {fact.key: fact for fact in turn.facts}
    lines = ["| Country | Item | Indicator | Year | Value | Unit | Source |",
             "|---|---|---|---|---:|---|---|"]
    seen = set()
    for source_id, item in sources:
        observation = _bound_observation(item, binds)
        if observation is None:
            continue
        key, country, year, member = observation
        bind = binds[key]
        meta = item["metadata"]
        raw = meta["raw_row"]
        value = next((raw[c] for c in ["value", *bind.get("measure_columns", [])] if c in raw), None)
        units = (load_mart_table_schema(bind["table_id"]) or {}).get("column_units") or {}
        unit = str(raw.get("unit") or (meta.get("value_semantics") or {}).get("unit")
                   or next((units[c] for c in bind.get("measure_columns", []) if c in units), ""))
        source = str(raw.get("organisation_name") or meta.get("organisation_name") or raw.get("source_key") or "")
        identity = (observation, str(value), unit, source)
        if not unit or identity in seen:
            continue
        seen.add(identity)
        fact = facts[key]
        spec = MEASURES[fact.measure_id]
        scope = fact_scope(turn, fact)
        place = next((geo for geo in scope.geo if name_to_iso3(geo) == country), country)
        # Render labels as table cells, without allowing a label to create rows or columns.
        cells = [place, member or str(raw.get("indicator") or raw.get("metric") or spec.aliases[0]),
                 spec.aliases[0], str(year), f"{Decimal(str(value)):,.2f}", unit, f"{source} [{source_id}]"]
        lines.append("| " + " | ".join(cell.replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    if len(lines) == 2:
        return ""
    notes = [fact_gap_description(gap) for gap in coverage or [] if gap["status"] != "satisfied"]
    if any(bind.get("member_selection") == "discover" for bind in binds.values()):
        notes.append("The table lists the available observations returned for the requested scope; it is not an exhaustive list of crops grown in the country.")
    if prefixes := sorted({bind["product_group_prefix"] for bind in binds.values() if bind.get("product_group_prefix")}):
        notes.append("Product groups use the catalog's CPC classification (" + ", ".join(prefixes) + "); unclassified products are excluded.")
    return "\n".join([*lines, "", *notes]).strip()


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
        key = str(entry.get("requirement_id") or table)
        status = str(entry.get("status") or "unknown")
        attempts.setdefault(key, set()).add(status)

    observations: dict[tuple[str, str, str], set[int]] = {}
    for item in rows:
        if observation := _bound_observation(item, binds):
            table, country, year, member = observation
            observations.setdefault((table, country, member), set()).add(year)

    coverage: list[dict[str, Any]] = []
    for fact in turn.facts:
        scope = fact_scope(turn, fact)
        table = tables.get(fact.key, "")
        bind_key = fact.key if fact.key in binds else table
        raw_statuses = (attempts.get(fact.key) or attempts.get(table) or attempts.get("") or set()) - {"unknown", "planned_not_executed"}
        statuses = {execution_status(status) for status in raw_statuses}
        if not table or bind_key not in binds:
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
            missing_status = "not_queried"

        for geo in scope.geo:
            country = name_to_iso3(geo) or ""
            enumerated = bool((binds.get(bind_key) or {}).get("enumerate_members"))
            members = list((binds.get(bind_key) or {}).get("product_literals") or []) if enumerated else []
            found_members = {member for key, iso, member in observations if key == bind_key and iso == country}
            members = members or (sorted(found_members) if enumerated else [""]) or [""]
            for member in members:
                observed = (observations.get((bind_key, country, member), set()) if enumerated else
                            set().union(*(years for (key, iso, _), years in observations.items()
                                          if key == bind_key and iso == country)))
                for period in scope.time_spec.windows:
                    years = set(range(period.start.year, period.end.year + 1))
                    found = years & observed
                    coverage.append({
                        "measure": fact.measure_id,
                        "requirement_id": fact.key,
                        "geo": geo,
                        "grain": scope.geo_grain,
                        "start": period.start.isoformat(),
                        "end": period.end.isoformat(),
                        "status": "satisfied" if found == years else missing_status,
                        "observed_years": sorted(found),
                        "missing_years": sorted(years - found),
                        **({"member": member, "member_selection": binds[bind_key].get("member_selection")} if enumerated else {}),
                    })
    return coverage


def fact_gap_description(gap: dict[str, Any]) -> str:
    """Client-readable limitation without turning execution failures into data absence."""
    measure = MEASURES.get(str(gap.get("measure") or ""))
    label = measure.aliases[0] if measure else "requested indicator"
    reason = {
        "unsupported": "a verified query for this indicator could not be compiled",
        "not_queried": "the planned query did not run",
        "timed_out": "the query timed out",
        "rejected": "the query did not pass validation",
        "unavailable": "the query could not complete",
        "no_match": "the completed query did not verify all requested years",
    }.get(str(gap.get("status") or ""), "the requested figures could not be verified")
    member = f" ({gap['member']})" if gap.get("member") else ""
    return f"{label}{member} in {gap['geo']} ({gap['start'][:4]}–{gap['end'][:4]}): not available -- {reason}."
