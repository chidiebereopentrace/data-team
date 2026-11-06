"""Count mart_dev fact rows by indicator class, country, and year, then write the report."""

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from google.cloud import bigquery

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from count_landing_datapoints import (  # noqa: E402
    REPO_ROOT,
    _load_dotenv,
    as_string,
    quote_bq,
    table_columns,
    year_expr,
)
from write_landing_report import (  # noqa: E402
    build_document,
    class_total,
    export_pdf,
    load_classes,
    load_country_lookup,
    load_counts,
    write_resolved_csv,
)

DATA_TEAM = REPO_ROOT.parent
MART_CLASSES = DATA_TEAM / "ml-eng" / "ml" / "rag" / "helpers" / "mart_indicator_classes.yaml"
TABLES_PATH = SCRIPT_DIR / "mart_dev_rag_tables.txt"
CATALOG_PATH = SCRIPT_DIR.parent / "out" / "dashboard" / "indicator_class_catalog.csv"
NAMES_PATH = SCRIPT_DIR / "country_names.yaml"
OUT_DIR = SCRIPT_DIR.parent / "out" / "dashboard"
RAW_PATH = OUT_DIR / "mart_datapoints_by_class_country_year.csv"
RESOLVED_PATH = OUT_DIR / "mart_datapoints_by_class_country_resolved.csv"
DOCX_PATH = OUT_DIR / "mart_datapoint_report.docx"
PDF_PATH = OUT_DIR / "mart_datapoint_report.pdf"

DATASET = "mart_dev"
UNCLASSIFIED = "unclassified"
COUNTRY_CANDIDATES = ("country_iso3", "country_name", "country_code", "place_scope")
YEAR_CANDIDATES = ("year", "harvest_year", "time_key")


def _key_file() -> None:
    if (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip():
        return
    key = REPO_ROOT / "data" / "local" / "keys" / "opentrace-bq-key.json"
    if key.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key)


def fact_tables(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name.startswith("fct_"):
            names.append(name)
    return names


def class_by_table(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assigned: dict[str, str] = {}
    for class_id, spec in (data.get("classes") or {}).items():
        for table in spec.get("primary_facts") or []:
            assigned.setdefault(str(table), str(class_id))
    return assigned


def pick_named(columns: list[tuple[str, str]], candidates: tuple[str, ...]) -> tuple[str, str] | None:
    by_lower = {name.lower(): (name, data_type) for name, data_type in columns}
    for candidate in candidates:
        hit = by_lower.get(candidate)
        if hit is not None:
            return hit
    return None


def year_sql(picked: tuple[str, str] | None) -> str:
    if picked is None:
        return "CAST(NULL AS STRING)"
    name, data_type = picked
    if name.lower() == "time_key":
        ident = quote_bq(name)
        return (
            "CASE WHEN SAFE_CAST("
            f"{ident} AS INT64) BETWEEN 1900 AND 2100 "
            f"THEN CAST(SAFE_CAST({ident} AS INT64) AS STRING) ELSE NULL END"
        )
    return year_expr(name, data_type)


def count_sql(project: str, table: str, columns: list[tuple[str, str]]) -> tuple[str, str, str]:
    country = pick_named(columns, COUNTRY_CANDIDATES)
    year = pick_named(columns, YEAR_CANDIDATES)
    country_sql = as_string(country[0], country[1]) if country else "CAST(NULL AS STRING)"
    sql = f"""
    SELECT {country_sql} AS country, {year_sql(year)} AS year, COUNT(*) AS n
    FROM {quote_bq(project)}.{quote_bq(DATASET)}.{quote_bq(table)}
    GROUP BY country, year
    """
    return sql, (country[0] if country else ""), (year[0] if year else "")


def query_table(
    client: bigquery.Client,
    project: str,
    table: str,
    columns: list[tuple[str, str]],
    class_id: str,
) -> tuple[str, list[dict], str, str, str]:
    sql, country_column, year_column = count_sql(project, table, columns)
    try:
        rows = [
            {
                "class_id": class_id,
                "table": table,
                "country": "" if row.country is None else str(row.country),
                "year": "" if row.year is None else str(row.year),
                "n": int(row.n),
            }
            for row in client.query(sql).result()
        ]
        return table, rows, country_column, year_column, ""
    except Exception as exc:  # noqa: BLE001 — one failed fact table must not hide the rest
        return table, [], country_column, year_column, str(exc)


def write_raw(path: Path, rows: list[dict]) -> None:
    fields = ["class_id", "table", "country", "year", "n"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    _load_dotenv()
    _key_file()
    project = (os.environ.get("BQ_PROJECT") or "").strip()
    if not project:
        print("Set BQ_PROJECT (e.g. in data/local/.env)", file=sys.stderr)
        return 1
    if not CATALOG_PATH.exists():
        print(f"Missing {CATALOG_PATH}", file=sys.stderr)
        return 1

    tables = fact_tables(TABLES_PATH)
    classes_for_table = class_by_table(MART_CLASSES)
    client = bigquery.Client(project=project)
    print(f"Reading columns for {DATASET}")
    columns_by_table = table_columns(client, project, DATASET)
    missing = [table for table in tables if table not in columns_by_table]
    if missing:
        print("Missing fact tables: " + ", ".join(missing), file=sys.stderr)
        return 1

    jobs = []
    for table in tables:
        class_id = classes_for_table.get(table, UNCLASSIFIED)
        jobs.append((table, columns_by_table[table], class_id))
        print(f"  {table} -> {class_id}")

    raw_rows: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(query_table, client, project, table, columns, class_id)
            for table, columns, class_id in jobs
        ]
        for future in as_completed(futures):
            table, rows, country_column, year_column, error = future.result()
            total = sum(row["n"] for row in rows)
            if error:
                errors.append(f"{table}: {error}")
                print(f"  {table} failed: {error}")
            else:
                raw_rows.extend(rows)
                print(
                    f"  {table}: {total} rows"
                    f" country={country_column or 'none'}"
                    f" year={year_column or 'none'}"
                )

    if errors:
        print(f"{len(errors)} fact tables failed", file=sys.stderr)
        return 1

    raw_rows.sort(key=lambda row: (row["class_id"], row["table"], row["country"], row["year"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_raw(RAW_PATH, raw_rows)
    grand = sum(row["n"] for row in raw_rows)
    print(f"Grand total: {grand}")
    print(f"Wrote {RAW_PATH}")

    lookup = load_country_lookup(NAMES_PATH)
    class_names = load_classes(CATALOG_PATH)
    counts = load_counts(RAW_PATH, lookup)
    resolved_total = sum(class_total(countries) for countries in counts.values())
    if resolved_total != grand:
        print(f"Resolved total {resolved_total} does not match {grand}", file=sys.stderr)
        return 1
    class_order = sorted(counts, key=lambda class_id: (-class_total(counts[class_id]), class_id))
    write_resolved_csv(RESOLVED_PATH, counts, class_order)
    print(f"Wrote {RESOLVED_PATH}")

    build_document(
        DOCX_PATH,
        counts,
        class_names,
        class_order,
        grand,
        title="Mart_dev datapoints by indicator class, country, and year",
        source_label="mart_dev fact tables",
        expected_total=None,
    )
    print(f"Wrote {DOCX_PATH}")
    export_pdf(DOCX_PATH, PDF_PATH)
    print(f"Wrote {PDF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
