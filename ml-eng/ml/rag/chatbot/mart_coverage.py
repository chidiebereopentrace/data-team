"""What the mart actually holds: country by indicator table, with the year span.

Built from live BigQuery after each mart build and published as a JSON artifact.
The retriever reads it to route to a table that holds the place and period, to
clamp "latest" to the real newest year, and to name the nearest period in a gap.

Build:
  cd ml-eng && python -m ml.rag.chatbot.mart_coverage --build
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from google.cloud import bigquery

from ml.rag.chatbot.bq_mart_sql import mart_dataset
from ml.rag.chatbot.bq_sql_validate import max_bytes_billed_for_source
from ml.rag.chatbot.bq_table_schema_yaml import (
    columns_for_mart_tables,
    geo_column,
    year_column,
)
from ml.rag.chatbot.geo_iso3 import iso3_to_canonical_name, name_to_iso3
from ml.rag.chatbot.query_form_admission import admit_catalog, shape_signature_for
from ml.rag.chatbot.query_forms import indicator_tables

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "mart_coverage.json"
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ISO3_RE = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class CoverageEntry:
    table_id: str
    geo_column: str
    geo_value: str
    country: str
    iso3: str
    min_year: int
    max_year: int


@dataclass
class Coverage:
    entries: list[CoverageEntry] = field(default_factory=list)
    admitted_shapes: dict[str, list[str]] = field(default_factory=dict)
    built_at: str = ""

    def entry(self, table_id: str, iso3: str) -> CoverageEntry | None:
        bare = table_id.split(".")[-1].lower()
        code = iso3.upper()
        for item in self.entries:
            if item.table_id == bare and item.iso3 == code:
                return item
        return None

    def tables_holding(self, iso3: str, tables: list[str]) -> list[str]:
        """Candidate tables, in their given order, that hold rows for the country."""
        code = iso3.upper()
        held = {item.table_id for item in self.entries if item.iso3 == code}
        return [t for t in tables if t.split(".")[-1].lower() in held]

    def latest_year(self, table_id: str, iso3: str) -> int | None:
        hit = self.entry(table_id, iso3)
        return hit.max_year if hit else None

    def shape_admitted(self, table_id: str, signature: str) -> bool:
        shapes = self.admitted_shapes.get(table_id.split(".")[-1].lower())
        if shapes is None:
            return True
        return signature in shapes


def route_tables(tables: list[str] | set[str], iso3: str | None) -> list[str]:
    """Keep candidate tables that hold the country, in the given order."""
    ordered = [str(t).split(".")[-1].lower() for t in tables if str(t).strip()]
    if not iso3 or not ordered:
        return ordered
    cov = load_coverage()
    if not cov.entries:
        return ordered
    held = cov.tables_holding(iso3, ordered)
    return held or ordered


def clamp_year(table_id: str, iso3: str | None, year: int | None) -> int | None:
    """Cap a requested year to the newest year the table holds for the country."""
    if year is None or not iso3:
        return year
    latest = load_coverage().latest_year(table_id, iso3)
    if latest is None:
        return year
    return min(year, latest)


def _country_fields(column: str, value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if column == "country_iso3" or _ISO3_RE.match(text):
        code = text.upper()
        return iso3_to_canonical_name(code) or code, code
    code = name_to_iso3(text) or ""
    return text, code


def _ident(name: str | None) -> str | None:
    text = str(name or "").strip()
    return text if _IDENT_RE.match(text) else None


def _year_of(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "year"):
        return int(value.year)
    match = re.match(r"^\d{4}", str(value).strip())
    return int(match.group(0)) if match else None


def build_table_coverage(
    client: bigquery.Client,
    table_id: str,
    *,
    project_id: str,
    dataset: str,
) -> list[CoverageEntry]:
    bare = table_id.split(".")[-1].lower()
    columns = columns_for_mart_tables({bare}).get(bare) or set()
    geo = _ident(geo_column(bare))
    year_col = _ident(year_column(bare))
    joins: list[str] = []
    if geo in ("geo_key", "geography_key"):
        joins.append(
            f"JOIN `{project_id}.{dataset}.dim_geography` g ON g.geography_key = f.{geo}"
        )
        geo_expr, geo = "g.country_iso3", "country_iso3"
    elif geo:
        geo_expr = f"f.{geo}"
    else:
        return []
    if year_col == "time_key":
        joins.append(f"JOIN `{project_id}.{dataset}.dim_time` t ON t.time_key = f.time_key")
        year_expr = "t.year"
    elif year_col:
        year_expr = f"f.{year_col}"
    elif "as_of_date" in columns:
        year_expr = "EXTRACT(YEAR FROM f.as_of_date)"
    else:
        return []
    sql = (
        f"SELECT {geo_expr} AS geo, MIN({year_expr}) AS y0, MAX({year_expr}) AS y1 "
        f"FROM `{project_id}.{dataset}.{bare}` f {' '.join(joins)} "
        f"WHERE {geo_expr} IS NOT NULL AND {year_expr} IS NOT NULL GROUP BY geo"
    )
    cap = max_bytes_billed_for_source("template")
    config = bigquery.QueryJobConfig(maximum_bytes_billed=cap) if cap > 0 else bigquery.QueryJobConfig()
    try:
        rows = list(client.query(sql, job_config=config).result(timeout=120))
    except Exception:
        return []
    out: list[CoverageEntry] = []
    for row in rows:
        y0 = _year_of(row.get("y0"))
        y1 = _year_of(row.get("y1"))
        value = str(row.get("geo") or "").strip()
        if not value or y0 is None or y1 is None:
            continue
        country, iso3 = _country_fields(geo, value)
        out.append(
            CoverageEntry(
                table_id=bare,
                geo_column=geo,
                geo_value=value,
                country=country,
                iso3=iso3,
                min_year=y0,
                max_year=y1,
            )
        )
    return out


def build_coverage(client: bigquery.Client, *, admit: bool = True) -> Coverage:
    proj = os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4").strip()
    ds = mart_dataset()
    entries: list[CoverageEntry] = []
    seen: set[str] = set()
    for _indicator, table_id in indicator_tables():
        if table_id in seen:
            continue
        seen.add(table_id)
        entries.extend(build_table_coverage(client, table_id, project_id=proj, dataset=ds))
    shapes: dict[str, list[str]] = {}
    if admit:
        from_admission = _admitted_shapes(client)
        shapes.update(from_admission)
    return Coverage(
        entries=entries,
        admitted_shapes=shapes,
        built_at=datetime.now(UTC).isoformat(),
    )


def _admitted_shapes(client: bigquery.Client) -> dict[str, list[str]]:
    shapes: dict[str, set[str]] = {}
    for row in admit_catalog(client):
        if not row.table_id:
            continue
        bucket = shapes.setdefault(row.table_id, set())
        if row.admitted and row.reason == "ok":
            bucket.add(shape_signature_for(row.form_id))
    return {table: sorted(sigs) for table, sigs in shapes.items()}


def write_coverage(coverage: Coverage, path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "built_at": coverage.built_at,
        "entries": [asdict(e) for e in coverage.entries],
        "admitted_shapes": coverage.admitted_shapes,
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    load_coverage.cache_clear()
    return path


@lru_cache(maxsize=1)
def load_coverage(path: Path = ARTIFACT) -> Coverage:
    if not path.is_file():
        return Coverage()
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = [CoverageEntry(**item) for item in raw.get("entries") or []]
    shapes = {str(k): [str(s) for s in v] for k, v in (raw.get("admitted_shapes") or {}).items()}
    return Coverage(entries=entries, admitted_shapes=shapes, built_at=str(raw.get("built_at") or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the mart coverage artifact from live BigQuery")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--no-admit", action="store_true", help="skip live form admission")
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return 1
    client = bigquery.Client(project=os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4"))
    coverage = build_coverage(client, admit=not args.no_admit)
    path = write_coverage(coverage)
    print(f"{len(coverage.entries)} coverage rows, {len(coverage.admitted_shapes)} tables admitted -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "Coverage",
    "CoverageEntry",
    "build_coverage",
    "clamp_year",
    "load_coverage",
    "route_tables",
    "write_coverage",
]
