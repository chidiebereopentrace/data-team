"""Admit a query form only after live BigQuery accepts it and returns rows.

Slot values come from the target table. A country and indicator pair the table
does not hold is a coverage gap: that pair is not marked supported, and it does
not by itself reject the form. SQL BigQuery rejects, or SQL that returns nothing
for a combination the table does hold, stays out.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from google.cloud import bigquery

from ml.rag.chatbot.bq_mart_sql import mart_dataset
from ml.rag.chatbot.bq_sql_validate import (
    dry_run_sql,
    max_bytes_billed_for_source,
    validate_required_metric_filters,
    validate_sql_column_allowlist,
)
from ml.rag.chatbot.bq_table_schema_yaml import (
    geo_column,
    measure_columns,
    measure_columns_mart,
    year_column,
)
from ml.rag.chatbot.query_forms import QUERY_FORMS, QueryForm, compile_form, indicator_tables

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_YEAR_RE = re.compile(r"^\d{4}")
_HASH_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class HeldValues:
    geo: str
    year: int
    year_end: int


@dataclass(frozen=True)
class FormAdmission:
    form_id: str
    indicator: str
    table_id: str
    admitted: bool
    row_count: int
    reason: str


def _ident(name: str | None) -> str | None:
    text = str(name or "").strip()
    if _IDENT_RE.match(text):
        return text
    return None


def _year_number(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "year"):
        try:
            return int(value.year)
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    match = _YEAR_RE.match(text)
    if not match:
        return None
    year = int(match.group(0))
    if 1900 <= year <= 2100:
        return year
    return None


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _job_config() -> bigquery.QueryJobConfig:
    cap = max_bytes_billed_for_source("template")
    if cap > 0:
        return bigquery.QueryJobConfig(maximum_bytes_billed=cap)
    return bigquery.QueryJobConfig()


def _query_rows(client: bigquery.Client, sql: str) -> list[dict[str, Any]]:
    job = client.query(sql, job_config=_job_config())
    return [dict(row) for row in job.result(timeout=60)]


def discover_held_values(
    client: bigquery.Client,
    table_id: str,
    *,
    project_id: str,
    dataset: str,
) -> HeldValues | None:
    """One geography and year grain the table already stores."""
    bare = str(table_id or "").strip().split(".")[-1]
    geo = _ident(geo_column(bare))
    year_col = _ident(year_column(bare))
    if not geo or not year_col:
        return None
    fqn = f"`{project_id}.{dataset}.{bare}`"
    window = f"{year_col} BETWEEN 2015 AND 2024"
    probe = (
        f"SELECT {geo} AS geo, {year_col} AS yr FROM {fqn} "
        f"WHERE {geo} IS NOT NULL AND {window} LIMIT 1"
    )
    if dry_run_sql(client, probe):
        return None
    try:
        rows = _query_rows(client, probe)
    except Exception:
        rows = []
    if not rows:
        wide = (
            f"SELECT {geo} AS geo, {year_col} AS yr FROM {fqn} "
            f"WHERE {geo} IS NOT NULL AND {year_col} IS NOT NULL LIMIT 1"
        )
        if dry_run_sql(client, wide):
            return None
        try:
            rows = _query_rows(client, wide)
        except Exception:
            return None
    if not rows:
        return None
    geo_val = str(rows[0].get("geo") or "").strip()
    year = _year_number(rows[0].get("yr"))
    if not geo_val or year is None:
        return None
    year_end = year
    latest = (
        f"SELECT {year_col} AS yr FROM {fqn} "
        f"WHERE {geo} = {_quote(geo_val)} AND {window} "
        f"ORDER BY {year_col} DESC LIMIT 1"
    )
    if dry_run_sql(client, latest) is None:
        try:
            latest_rows = _query_rows(client, latest)
        except Exception:
            latest_rows = []
        if latest_rows:
            latest_year = _year_number(latest_rows[0].get("yr"))
            if latest_year is not None:
                year_end = max(year, latest_year)
    return HeldValues(geo=geo_val, year=year, year_end=year_end)


def _admit_sql(
    client: bigquery.Client,
    sql: str,
    *,
    table_id: str,
) -> tuple[bool, int, str]:
    invalid = validate_required_metric_filters(sql, {table_id}) or validate_sql_column_allowlist(
        sql, {table_id}
    )
    if invalid:
        return False, 0, invalid
    dry = dry_run_sql(client, sql)
    if dry:
        return False, 0, dry
    try:
        rows = _query_rows(client, sql)
    except Exception as exc:
        return False, 0, str(exc)[:500]
    if not rows:
        return False, 0, "query returned no rows for values the table holds"
    unlabeled = unlabeled_figures(rows, measure_columns_mart(table_id) or measure_columns(table_id))
    if unlabeled:
        return False, len(rows), unlabeled
    return True, len(rows), "ok"


def unlabeled_figures(rows: list[dict[str, Any]], measures: list[str]) -> str | None:
    """Reason when two rows share every label column but report different figures."""
    measure_set = {m.lower() for m in measures}
    seen: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for row in rows:
        labels = tuple(
            sorted(
                (k, str(v))
                for k, v in row.items()
                if k.lower() not in measure_set and not _HASH_RE.match(str(v))
            )
        )
        figures = tuple(str(row.get(k)) for k in sorted(row) if k.lower() in measure_set)
        prior = seen.setdefault(labels, figures)
        if prior != figures:
            return f"rows share labels {dict(labels)} but report different figures; a grain column is missing"
    return None


def _sql_signature(form: QueryForm) -> tuple[str, str, bool, bool]:
    """Compiler inputs that change the SQL. Wording inside a signature does not."""
    return (
        form.shape,
        form.year_shape,
        form.named_discriminator,
        "{crop}" in form.template,
    )


def shape_signature(shape: str, year_shape: str) -> str:
    """Runtime key for an admitted SQL shape on a table."""
    return f"{shape}:{year_shape}"


_FORMS_BY_ID = {form.form_id: form for form in QUERY_FORMS}


def shape_signature_for(form_id: str) -> str:
    form = _FORMS_BY_ID[form_id]
    return shape_signature(form.shape, form.year_shape)


def admit_catalog(
    client: bigquery.Client,
    *,
    project_id: str | None = None,
    dataset: str | None = None,
    forms: tuple[QueryForm, ...] | None = None,
) -> list[FormAdmission]:
    """Dry-run and execute each distinct SQL statement on its indicator table.

    Forms that share a shape, year shape, named discriminator, and crop slot
    compile to one statement. That statement is executed once, and every form
    in the group gets that result. A form is admitted only when the statement
    returns rows.
    """
    proj = (project_id or os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4")).strip()
    ds = (dataset or mart_dataset()).strip()
    catalog = forms if forms is not None else QUERY_FORMS
    held_by_table: dict[str, HeldValues | None] = {}
    executed: dict[str, tuple[bool, int, str]] = {}
    results: list[FormAdmission] = []

    sql_forms = [form for form in catalog if form.shape != "report"]
    for form in catalog:
        if form.shape == "report":
            results.append(
                FormAdmission(
                    form_id=form.form_id,
                    indicator="",
                    table_id="",
                    admitted=True,
                    row_count=0,
                    reason="report_only",
                )
            )

    for indicator, table_id in indicator_tables():
        if table_id not in held_by_table:
            held_by_table[table_id] = discover_held_values(
                client, table_id, project_id=proj, dataset=ds
            )
        held = held_by_table[table_id]
        grouped: dict[tuple[str, str, bool, bool], list[QueryForm]] = {}
        for form in sql_forms:
            grouped.setdefault(_sql_signature(form), []).append(form)
        for forms in grouped.values():
            if held is None:
                for form in forms:
                    results.append(
                        FormAdmission(
                            form_id=form.form_id,
                            indicator=indicator,
                            table_id=table_id,
                            admitted=False,
                            row_count=0,
                            reason="coverage_gap",
                        )
                    )
                continue
            representative = forms[0]
            sql = compile_form(
                representative,
                table_id=table_id,
                country=held.geo,
                indicator=indicator,
                year=held.year,
                year_end=held.year_end,
                project_id=proj,
                dataset=ds,
            )
            if not sql:
                verdict = (False, 0, "compile_failed")
            elif sql not in executed:
                verdict = _admit_sql(client, sql, table_id=table_id)
                executed[sql] = verdict
            else:
                verdict = executed[sql]
            admitted, row_count, reason = verdict
            for form in forms:
                results.append(
                    FormAdmission(
                        form_id=form.form_id,
                        indicator=indicator,
                        table_id=table_id,
                        admitted=admitted,
                        row_count=row_count,
                        reason=reason,
                    )
                )
    return results


def admitted_sql_form_ids(results: list[FormAdmission]) -> set[str]:
    """SQL forms with at least one warehouse table that returned rows."""
    return {row.form_id for row in results if row.admitted and row.reason == "ok"}


__all__ = [
    "FormAdmission",
    "HeldValues",
    "admit_catalog",
    "admitted_sql_form_ids",
    "discover_held_values",
    "shape_signature",
    "shape_signature_for",
    "unlabeled_figures",
]
