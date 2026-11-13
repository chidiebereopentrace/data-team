"""Deterministic single-table SQL from TableBindContract.

The question supplies the shape. A single year is a point fact. A year span, or a
trend, is a series on the table's time grain. A discriminator the statement does
not filter is selected so rows stay labeled. An aggregate that omits one still
has to filter it, which the validator enforces.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from ml.rag.chatbot.bq_mart_sql import mart_dataset
from ml.rag.chatbot.bq_sql_validate import metric_discriminator_names
from ml.rag.chatbot.bq_table_schema_yaml import (
    TableBindContract,
    dim_label_joins,
    discriminator_columns,
    grain_columns,
    load_mart_table_schema,
    load_table_schema,
    year_column,
)
from ml.rag.chatbot.coverage_ledger import operation_supported
from ml.rag.helpers.mart_semantic_relationships import relationships_for

logger = logging.getLogger(__name__)

_YEAR_EQ_RE = re.compile(
    r"\b(?:year|time_key|harvest_year|observation_year|mp_year)\s*=\s*\d{4}\b",
    re.IGNORECASE,
)
_YEAR_BETWEEN_RE = re.compile(
    r"\b(?:year|time_key|harvest_year|observation_year|mp_year)\s+BETWEEN\s+(\d{4})\s+AND\s+(\d{4})",
    re.IGNORECASE,
)
_DATE_BETWEEN_RE = re.compile(
    r"\bBETWEEN\s+DATE\s+'(\d{4})-\d{2}-\d{2}'\s+AND\s+DATE\s+'(\d{4})-\d{2}-\d{2}'",
    re.IGNORECASE,
)
_FILTERED_COL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|\bIN\b)",
    re.IGNORECASE,
)
_TIME_GRAIN = (
    "year",
    "time_key",
    "harvest_year",
    "observation_year",
    "mp_year",
    "month",
    "as_of_date",
)
_FINE_GRAIN = ("as_of_date", "month")
ROW_BUDGET = 60
# Series and comparisons fetch the complete shape; one row over the budget signals result_too_large.
RESULT_ROW_BUDGET = 500
_OPEN_GRAIN_POINT_ROWS = 50


def bind_contract_for_table(
    bind_contracts: dict[str, Any] | None,
    table_id: str,
) -> dict[str, Any] | None:
    """Resolve a bind map entry; table ids match on the bare, case-insensitive name."""
    if not isinstance(bind_contracts, dict) or not bind_contracts:
        return None
    needle = str(table_id or "").strip().split(".")[-1].lower()
    if not needle:
        return None
    raw = bind_contracts.get(table_id)
    if isinstance(raw, dict):
        return raw
    raw = bind_contracts.get(needle)
    if isinstance(raw, dict):
        return raw
    for key, value in bind_contracts.items():
        if str(key or "").strip().split(".")[-1].lower() == needle and isinstance(value, dict):
            return value
    return None


def _schema_column_names(table_id: str) -> set[str]:
    bare = str(table_id or "").strip().split(".")[-1].lower()
    schema = load_mart_table_schema(bare) or load_table_schema(bare) or {}
    names: set[str] = set()
    for col in schema.get("columns") or []:
        name = str(col.get("name") or "").strip().lower()
        if name:
            names.add(name)
    return names


def _span_bounds(text: str) -> tuple[int, int] | None:
    for pattern in (_YEAR_BETWEEN_RE, _DATE_BETWEEN_RE):
        match = pattern.search(text or "")
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _is_trend(job: str, pattern: str) -> bool:
    tokens = f"{job} {pattern}".strip().lower().replace("-", "_")
    return "time_series" in tokens or any(part == "trend" for part in tokens.split())


def _job_kind(job: str, pattern: str) -> str:
    tokens = f"{job} {pattern}".strip().lower().replace("-", "_")
    parts = set(tokens.split())
    if "rank" in tokens or "ranking" in parts:
        return "rank"
    if "compare" in parts:
        return "compare"
    if "breakdown" in parts:
        return "breakdown"
    if "region" in parts or "list" in parts:
        return "region"
    return ""


def _aggregate_fn(column: str) -> str:
    low = column.lower()
    if "yield" in low or low.endswith("_avg") or "pct_" in low:
        return "AVG"
    if low.startswith("avg_"):
        return "AVG"
    return "SUM"


def _is_series(text: str, *, job: str, pattern: str) -> bool:
    bounds = _span_bounds(text)
    if bounds is not None and bounds[0] != bounds[1]:
        return True
    return _is_trend(job, pattern)


def _filtered_columns(sql: str) -> set[str]:
    return {match.group(1).lower() for match in _FILTERED_COL_RE.finditer(sql or "")}


def _unnamed_discriminators(table_id: str, where_sql: str, names: set[str]) -> list[str]:
    """Discriminators the statement does not filter. Selecting them keeps rows labeled."""
    filtered = _filtered_columns(where_sql)
    wanted = metric_discriminator_names()
    out: list[str] = []
    for col in discriminator_columns(table_id):
        low = col.lower()
        if low in wanted and low in names and low not in filtered and low not in out:
            out.append(low)
    return out


def _append_columns(select_list: list[str], extras: list[str]) -> None:
    selected = {col.lower() for col in select_list}
    for extra in extras:
        if extra.lower() not in selected:
            select_list.append(extra)
            selected.add(extra.lower())


def _is_unit_column(name: str) -> bool:
    low = name.lower()
    return low in ("unit", "currency") or low.endswith(("_unit", "_currency"))


_LABEL_PREFERENCE = (
    "country_name",
    "product_name",
    "item_name",
    "element_name",
    "indicator_name",
    "organisation_name",
    "market_name",
    "unit_code",
    "season_name",
    "land_use_code",
    "sex_name",
)


def _label_column(dim: str) -> str | None:
    cols = _schema_column_names(dim)
    for cand in _LABEL_PREFERENCE:
        if cand in cols:
            return cand
    for col in sorted(cols):
        if col.endswith("_name") or col.endswith("_code"):
            return col
    return None


def _declared_label_joins(
    bare: str, names: set[str], joined: set[str]
) -> list[tuple[str, str, str, str]]:
    """Dim and bridge joins declared for this table, skipping keys already joined."""
    out: list[tuple[str, str, str, str]] = []
    for spec in relationships_for(bare).get("joins_with") or []:
        if not isinstance(spec, dict):
            continue
        dim = str(spec.get("table") or "").strip()
        if not dim.startswith(("dim_", "bridge_")):
            continue
        on_cols = [str(col).strip() for col in (spec.get("on") or []) if str(col).strip()]
        fk = next((col for col in on_cols if col.lower() in names), "")
        dim_key = fk
        if not fk and any(col.lower() == "geography_key" for col in on_cols) and "geo_key" in names:
            fk, dim_key = "geo_key", "geography_key"
        if not fk or fk.lower() in joined:
            continue
        label = _label_column(dim)
        dim_cols = _schema_column_names(dim)
        if not label or label not in dim_cols or dim_key.lower() not in dim_cols:
            continue
        out.append((fk, dim, dim_key, label))
        joined.add(fk.lower())
    return out


def _grain_labels(
    bare: str, grain: list[str], names: set[str]
) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    """Readable labels for hash-key grain columns: fact columns that carry them, else dim joins."""
    direct: list[str] = []
    joins: list[tuple[str, str, str, str]] = []
    joined: set[str] = set()
    for fk, (dim, key, label) in dim_label_joins(bare).items():
        low_fk, low_label = fk.lower(), label.lower()
        if low_fk not in grain and low_fk != "unit_key":
            continue
        if low_label in names:
            direct.append(low_label)
            joined.add(low_fk)
            continue
        dim_cols = _schema_column_names(dim)
        if key.lower() in dim_cols and low_label in dim_cols:
            joins.append((fk, dim, key, label))
            joined.add(low_fk)
    joins.extend(_declared_label_joins(bare, names, joined))
    return direct, joins


def _grain_is_open(grain: list[str], filtered: set[str], *, series: bool) -> bool:
    """True when a non-time grain column is unfiltered, so several rows answer one point."""
    for col in grain:
        if col in _TIME_GRAIN and series:
            continue
        if col not in filtered and col not in _TIME_GRAIN:
            return True
    return False


def compile_sql_from_bind(
    contract: TableBindContract | dict[str, Any],
    *,
    project_id: str | None = None,
    dataset: str | None = None,
    limit: int = 1,
    job: str = "",
    pattern: str = "",
    breakdown: list[str] | None = None,
) -> str | None:
    """Compile one fact SELECT from bind filters and the question shape."""
    if isinstance(contract, dict):
        parsed = TableBindContract.from_dict(contract)
        if parsed is None:
            return None
        contract = parsed

    # An unfiltered scan is only a question shape on untimed snapshot tables (rank the continent).
    unfiltered_rank = _job_kind(job, pattern) == "rank" and not year_column(contract.table_id)
    if not str(contract.required_filters_sql or "").strip() and not unfiltered_rank:
        return None
    unproven = contract.unproven_filters()
    if unproven:
        logger.warning("bind contract %s refused: filters without provenance %s", contract.table_id, unproven)
        return None

    cols = [c for c in contract.measure_columns if str(c).strip()]
    if not cols:
        return None

    proj = (project_id or os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4")).strip()
    ds = (dataset or mart_dataset()).strip()
    bare = str(contract.table_id or "").strip().split(".")[-1].lower()
    if not bare:
        return None

    fqn = f"`{proj}.{ds}.{bare}`"
    where_sql = str(contract.required_filters_sql or "").strip()
    if where_sql.upper().startswith("AND "):
        where_sql = where_sql[4:].strip()
    if not where_sql:
        if not unfiltered_rank:
            return None
        where_sql = "TRUE"

    names = _schema_column_names(bare)
    shape_text = f"{where_sql} {contract.time_sql or ''}"
    kind = _job_kind(job, pattern)
    series = kind in ("",) and _is_series(shape_text, job=job, pattern=pattern)
    undated = (
        not series
        and not str(contract.time_sql or "").strip()
        and _YEAR_EQ_RE.search(where_sql) is None
        and _span_bounds(shape_text) is None
    )

    grain = [col.lower() for col in grain_columns(bare)]
    filtered = _filtered_columns(where_sql)
    hash_keys = {fk.lower() for fk in dim_label_joins(bare)}
    direct_labels, joins = _grain_labels(bare, grain, names)
    join_keys = {fk.lower() for fk, _, _, _ in joins}

    primary_measure = str(cols[0]).strip()
    alias_value = primary_measure.lower() != "value"

    def _output_name(col: str) -> str:
        if alias_value and col.lower() == primary_measure.lower():
            return "value"
        return col

    def _projected(col: str) -> str:
        out = _output_name(col)
        if out != col:
            return f"{col} AS {out}"
        return col

    select_list = [c for c in cols[:4]]
    _append_columns(select_list, [col for col in _TIME_GRAIN if col in names])
    _append_columns(select_list, [col for col in grain if col in names and col not in hash_keys])
    _append_columns(select_list, direct_labels)
    _append_columns(select_list, _unnamed_discriminators(bare, where_sql, names))
    _append_columns(select_list, sorted(col for col in names if _is_unit_column(col)))
    if kind == "breakdown":
        for dim in breakdown or []:
            if dim.lower() in names:
                _append_columns(select_list, [dim.lower()])

    open_grain = _grain_is_open(grain, filtered, series=series or kind in ("rank", "compare", "region", "breakdown"))
    if kind == "rank":
        order_cols = [_output_name(cols[0])] if cols else []
        order_sql = f" ORDER BY {order_cols[0]} DESC" if order_cols else ""
        cap = min(ROW_BUDGET, max(10, int(limit or 10)))
    elif kind in ("compare", "region", "breakdown"):
        order_cols = [col for col in _TIME_GRAIN if col in names]
        order_sql = (" ORDER BY " + ", ".join(order_cols)) if order_cols else ""
        cap = RESULT_ROW_BUDGET + 1
    elif series:
        order_cols = [col for col in _TIME_GRAIN if col in names]
        order_sql = (" ORDER BY " + ", ".join(f"{col} DESC" for col in order_cols)) if order_cols else ""
        cap = RESULT_ROW_BUDGET + 1
    else:
        if undated:
            order_cols = [col for col in ("year", "month") if col in names]
        else:
            order_cols = [col for col in _FINE_GRAIN if col in names]
        order_sql = (" ORDER BY " + ", ".join(f"{col} DESC" for col in order_cols)) if order_cols else ""
        cap = max(1, int(limit or 1))
        if open_grain:
            cap = max(cap, _OPEN_GRAIN_POINT_ROWS)

    group_sql = ""
    grouped: list[str] = []
    inner_select = list(select_list)
    if kind in ("rank", "breakdown"):
        discriminators = metric_discriminator_names()
        measure_set = {c.lower() for c in cols[:4] if c.lower() not in discriminators}
        grouped = [c for c in select_list if c.lower() not in measure_set]
        agg = [
            f"{_aggregate_fn(c)}({c}) AS {_output_name(c)}"
            for c in select_list
            if c.lower() in measure_set
        ]
        if grouped and agg:
            inner_select = agg + grouped
            group_sql = " GROUP BY " + ", ".join(grouped)
            if kind == "rank" and cols:
                order_sql = f" ORDER BY {_output_name(cols[0])} DESC"

    shown = inner_select if group_sql else [_projected(col) for col in inner_select]
    if not joins:
        return (
            f"SELECT {', '.join(shown)} FROM {fqn} "
            f"WHERE {where_sql}{group_sql}{order_sql} LIMIT {cap}"
        )

    inner_cols = list(shown)
    _append_columns(inner_cols, [fk for fk, _, _, _ in joins])
    if group_sql:
        group_sql = group_sql + ", " + ", ".join(
            fk for fk, _, _, _ in joins if fk.lower() not in {c.lower() for c in grouped}
        )
    outer_cols = [
        f"f.{_output_name(col)}" for col in select_list if col.lower() not in join_keys
    ]
    join_sql = ""
    for i, (fk, dim, key, label) in enumerate(joins):
        alias = f"d{i}"
        outer_cols.append(f"{alias}.{label}")
        join_sql += f" LEFT JOIN `{proj}.{ds}.{dim}` AS {alias} ON {alias}.{key} = f.{fk}"
    outer_order = order_sql.replace(" ORDER BY ", " ORDER BY f.", 1).replace(", ", ", f.")
    return (
        f"SELECT {', '.join(outer_cols)} FROM (SELECT {', '.join(inner_cols)} FROM {fqn} "
        f"WHERE {where_sql}{group_sql}{order_sql} LIMIT {cap}) AS f{join_sql}{outer_order}"
    )


Operation = Literal["lookup", "latest", "series", "compare", "rank", "categorical"]

_RANK_RE = re.compile(
    r"\b(rank(?:ing|ed)?|highest|lowest|most|least|top\s+\d+|which\s+countr(?:y|ies)|what\s+countr(?:y|ies))\b",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"\b(compar(?:e|ed|ing|ison)|versus|vs\.?|differ(?:s|ence|ences)?)\b",
    re.IGNORECASE,
)
_CATEGORY_RE = re.compile(
    r"\b(?:break\s*down|broken\s+down|by\s+(?:sex|gender|crop|product|source|element|category|sector|item|indicator))\b",
    re.IGNORECASE,
)
_SERIES_RE = re.compile(
    r"\b(trends?|over\s+time|time\s+series|since\s+(?:19|20)\d{2}|from\s+(?:19|20)\d{2}\s+to|year[-\s]by[-\s]year|changed|change\s+over)\b",
    re.IGNORECASE,
)
_LATEST_RE = re.compile(
    r"\b(latest|most\s+recent|current|right\s+now|as\s+of\s+now)\b",
    re.IGNORECASE,
)
_NAMED_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def requested_operation(query: str) -> Operation:
    """The one warehouse operation the question asks for."""
    text = query or ""
    if _RANK_RE.search(text):
        return "rank"
    if _COMPARE_RE.search(text):
        return "compare"
    if _CATEGORY_RE.search(text):
        return "categorical"
    if _SERIES_RE.search(text):
        return "series"
    if _LATEST_RE.search(text):
        return "latest"
    return "lookup"


def question_names_year(query: str) -> bool:
    """True when the question itself names a calendar year."""
    return bool(_NAMED_YEAR_RE.search(query or ""))

# Operation -> (job, pattern) understood by compile_sql_from_bind.
_OPERATION_SHAPE: dict[Operation, tuple[str, str]] = {
    "lookup": ("fact", ""),
    "latest": ("fact", ""),
    "series": ("trend", "time_series"),
    "compare": ("compare", ""),
    "rank": ("rank", ""),
    "categorical": ("breakdown", ""),
}


@dataclass(frozen=True)
class CompiledOperation:
    table_id: str
    operation: Operation
    sql: str | None
    refusal: str = ""


def operation_for(contract: TableBindContract, *, job: str = "", pattern: str = "") -> Operation:
    """The registry operation a question shape asks of one bound table."""
    kind = _job_kind(job, pattern)
    if kind == "rank":
        return "rank"
    if kind in ("compare", "region"):
        return "compare"
    if kind == "breakdown":
        return "categorical"
    if _is_series(f"{contract.required_filters_sql} {contract.time_sql or ''}", job=job, pattern=pattern):
        return "series"
    if not str(contract.time_sql or "").strip() and year_column(contract.table_id):
        return "latest"
    return "lookup"


def compile_operation(
    contract: TableBindContract | dict[str, Any],
    *,
    job: str = "",
    pattern: str = "",
    operation: Operation | None = None,
    breakdown: list[str] | None = None,
    project_id: str | None = None,
    dataset: str | None = None,
    limit: int = 1,
) -> CompiledOperation | None:
    """Compile the operation the ledger declares for this table, or refuse it with the recorded reason."""
    parsed = TableBindContract.from_dict(contract) if isinstance(contract, dict) else contract
    if parsed is None:
        return None
    bare = parsed.table_id.strip().split(".")[-1].lower()
    chosen = operation or operation_for(parsed, job=job, pattern=pattern)
    supported, reason = operation_supported(bare, chosen)
    if not supported:
        return CompiledOperation(bare, chosen, None, f"{chosen} is not supported on {bare}: {reason}")
    op_job, op_pattern = _OPERATION_SHAPE[chosen]
    dims = breakdown or (discriminator_columns(bare) if chosen == "categorical" else [])
    sql = compile_sql_from_bind(
        parsed,
        project_id=project_id,
        dataset=dataset,
        limit=limit if chosen == "lookup" else max(limit, RESULT_ROW_BUDGET),
        job=op_job,
        pattern=op_pattern,
        breakdown=dims,
    )
    return CompiledOperation(bare, chosen, sql, "" if sql else "bind compiler produced no SQL")


__all__ = [
    "RESULT_ROW_BUDGET",
    "ROW_BUDGET",
    "CompiledOperation",
    "Operation",
    "bind_contract_for_table",
    "compile_operation",
    "compile_sql_from_bind",
    "operation_for",
    "question_names_year",
    "requested_operation",
]
