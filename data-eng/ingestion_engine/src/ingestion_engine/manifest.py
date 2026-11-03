"""Read a directory of manifest CSVs. One row is one landing table."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

INDICATOR_CLASSES = frozenset({
    "EL", "GYI", "FVC", "FS", "PROD", "PRC", "CLIM", "SOIL",
    "AH", "VEG", "ENV", "INP", "HDI", "BIO", "RES",
})
CONTRACTS = frozenset({"json_api", "csv", "xlsx", "xls", "html_table"})
GEO_GRAINS = frozenset({"country", "admin1", "admin2", "market", "point"})
TEMPORAL_GRAINS = frozenset({"day", "month", "quarter", "season", "year"})
CODE_STANDARDS = frozenset({"iso2", "iso3", "name"})
LOAD_MODES = frozenset({"append_snapshot", "replace_partition", "replace_table"})
AUTH_TYPES = frozenset({"none", "api_key_header", "api_key_query", "bearer", "basic"})
PAGINATION = frozenset({"none", "page", "offset", "next_link"})
TABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,1023}")
SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

REQUIRED = (
    "source_id", "source_name", "source_url", "endpoint", "table_name",
    "indicator_class", "contract_type", "column_count", "column_names",
    "geo_grain", "geo_country_column", "geo_code_standard",
    "temporal_grain", "temporal_column", "temporal_format",
    "value_column", "load_mode", "natural_key", "enabled", "owner",
    "auth_required", "auth_type", "auth_ref",
)


@dataclass(frozen=True)
class Source:
    manifest_file: str
    source_id: str
    source_name: str
    source_url: str
    endpoint: str
    table_name: str
    indicator_class: str
    contract_type: str
    column_count: int
    column_names: tuple[str, ...]
    geo_grain: str
    geo_country_column: str
    geo_country_columns: tuple[str, ...]
    geo_country_constant: str
    geo_code_standard: str
    geo_admin1_column: str
    geo_admin2_column: str
    geo_market_column: str
    geo_lat_column: str
    geo_lon_column: str
    temporal_grain: str
    temporal_column: str
    temporal_format: str
    value_column: str
    unit_column: str
    unit_constant: str
    class_columns: tuple[str, ...]
    measure_kind: str
    freshness_days: int | None
    load_mode: str
    natural_key: tuple[str, ...]
    owner: str
    auth_required: bool
    auth_type: str
    auth_ref: str
    auth_header: str
    auth_query_param: str
    records_path: str
    pagination: str
    page_param: str
    page_size: int | None
    size_param: str
    next_path: str
    sheet_name: str
    header_row: int
    html_table_selector: int
    query: str
    xlsx_layout: str
    record_layout: str
    file_name: str
    row_filter: str = ""


@dataclass(frozen=True)
class ManifestProblem:
    manifest_file: str
    source_id: str
    source_name: str
    source_url: str
    endpoint: str
    table_name: str
    indicator_class: str
    contract_type: str
    auth_required: str
    auth_type: str
    owner: str
    error: str
    stage: str = "manifest"


def _cell(row: dict[str, str], name: str) -> str:
    return (row.get(name) or "").strip()


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _int(value: str) -> int | None:
    if not value.strip():
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    return number


def _problem(path: str, row: dict[str, str], error: str) -> ManifestProblem:
    return ManifestProblem(
        manifest_file=path,
        source_id=_cell(row, "source_id"),
        source_name=_cell(row, "source_name"),
        source_url=_cell(row, "source_url"),
        endpoint=_cell(row, "endpoint"),
        table_name=_cell(row, "table_name"),
        indicator_class=_cell(row, "indicator_class"),
        contract_type=_cell(row, "contract_type"),
        auth_required=_cell(row, "auth_required"),
        auth_type=_cell(row, "auth_type"),
        owner=_cell(row, "owner"),
        error=error,
    )


def _validate(row: dict[str, str]) -> str | None:
    missing = [name for name in REQUIRED if name not in row]
    if missing:
        return "missing columns: " + ", ".join(missing)
    if not SOURCE_ID.fullmatch(_cell(row, "source_id")):
        return "source_id must start with a letter or digit and use only letters, digits, dot, underscore, or hyphen"
    if not _cell(row, "source_name") or not _cell(row, "owner"):
        return "source_name and owner are required"
    if not _cell(row, "source_url").startswith("https://") or not _cell(row, "endpoint").startswith("https://"):
        return "source_url and endpoint must be https URLs"
    if not TABLE_NAME.fullmatch(_cell(row, "table_name")):
        return "table_name is not a legal BigQuery table name"
    if _cell(row, "indicator_class") not in INDICATOR_CLASSES:
        return "unknown indicator class"
    contract = _cell(row, "contract_type")
    if contract not in CONTRACTS:
        return "contract_type must be json_api, csv, xlsx, xls, or html_table"
    file_name = _cell(row, "file_name")
    if file_name and ("/" in file_name or "\\" in file_name or file_name in {".", ".."}):
        return "file_name must be a single file name"
    if file_name and contract not in {"csv", "xlsx", "xls"}:
        return "file_name is only used with csv, xlsx, or xls"
    names = _split(_cell(row, "column_names"))
    count = _int(_cell(row, "column_count"))
    if count is None or count != len(names) or count < 1:
        return "column_count must equal the number of column_names"
    if len(set(names)) != len(names):
        return "column_names contains a duplicate"
    filter_error = _check_row_filter(_cell(row, "row_filter"), set(names))
    if filter_error:
        return filter_error
    grain = _cell(row, "geo_grain")
    if grain not in GEO_GRAINS:
        return "unknown geo_grain"
    if not _cell(row, "geo_country_column") and not _cell(row, "geo_country_constant"):
        return "geo_country_column is required"
    if _cell(row, "geo_code_standard") not in CODE_STANDARDS:
        return "geo_code_standard must be iso2, iso3, or name"
    if grain in {"admin1", "admin2", "market"} and not _cell(row, "geo_admin1_column") and grain != "market":
        return "geo_admin1_column is required for this geo grain"
    if grain == "admin2" and not _cell(row, "geo_admin2_column"):
        return "geo_admin2_column is required for admin2"
    if grain == "market" and not _cell(row, "geo_market_column"):
        return "geo_market_column is required for market"
    if grain == "point" and (not _cell(row, "geo_lat_column") or not _cell(row, "geo_lon_column")):
        return "point grain requires geo_lat_column and geo_lon_column"
    if _cell(row, "temporal_grain") not in TEMPORAL_GRAINS:
        return "unknown temporal_grain"
    if not _cell(row, "temporal_column") or not _cell(row, "temporal_format"):
        return "temporal_column and temporal_format are required"
    if not _cell(row, "value_column"):
        return "value_column is required"
    if _cell(row, "load_mode") not in LOAD_MODES:
        return "load_mode must be append_snapshot, replace_partition, or replace_table"
    keys = _split(_cell(row, "natural_key"))
    if not keys:
        return "natural_key is required"
    declared = set(names)
    referenced = [_cell(row, "temporal_column"), _cell(row, "value_column"), *keys]
    extra_countries = _split(_cell(row, "geo_country_columns"))
    if len(set(extra_countries)) != len(extra_countries):
        return "geo_country_columns contains a duplicate"
    if _cell(row, "geo_country_column") and _cell(row, "geo_country_column") in extra_countries:
        return "geo_country_columns must not repeat geo_country_column"
    if _cell(row, "geo_country_column"):
        referenced.append(_cell(row, "geo_country_column"))
    referenced.extend(extra_countries)
    for optional in (
        "geo_admin1_column", "geo_admin2_column", "geo_market_column",
        "geo_lat_column", "geo_lon_column", "unit_column",
    ):
        if _cell(row, optional):
            referenced.append(_cell(row, optional))
    referenced.extend(_split(_cell(row, "class_columns")))
    unknown = sorted({name for name in referenced if name not in declared})
    if unknown:
        return "columns not in column_names: " + ", ".join(unknown)
    indicator = _cell(row, "indicator_class")
    class_columns = _split(_cell(row, "class_columns"))
    if not class_columns:
        return f"{indicator} requires a reported measure column in class_columns"
    if indicator == "PRC" and len(class_columns) < 2 and not _cell(row, "unit_constant"):
        return "PRC requires commodity and currency columns in class_columns"
    if indicator == "PROD":
        if not class_columns:
            return "PROD requires a commodity column in class_columns"
        if _cell(row, "measure_kind") not in {"quantity", "area", "yield"}:
            return "PROD measure_kind must be quantity, area, or yield"
    auth_required = _bool(_cell(row, "auth_required"))
    if auth_required is None:
        return "auth_required must be true or false"
    auth_type = _cell(row, "auth_type")
    if auth_type not in AUTH_TYPES:
        return "unknown auth_type"
    if auth_required:
        if auth_type == "none" or not _cell(row, "auth_ref"):
            return "a gated endpoint needs auth_type and auth_ref"
        if auth_type == "api_key_header" and not _cell(row, "auth_header"):
            return "api_key_header requires auth_header"
        if auth_type == "api_key_query" and not _cell(row, "auth_query_param"):
            return "api_key_query requires auth_query_param"
    elif auth_type != "none" or _cell(row, "auth_ref"):
        return "a public endpoint must use auth_type none and a blank auth_ref"
    pagination = _cell(row, "pagination") or "none"
    if pagination not in PAGINATION:
        return "pagination must be none, page, offset, or next_link"
    if pagination in {"page", "offset"}:
        size = _int(_cell(row, "page_size"))
        if not _cell(row, "page_param") or size is None or size < 1:
            return "page and offset pagination need page_param and a positive page_size"
    freshness = _int(_cell(row, "freshness_days")) if _cell(row, "freshness_days") else None
    if _cell(row, "freshness_days") and (freshness is None or freshness < 0):
        return "freshness_days must be a non-negative integer"
    header_row = _int(_cell(row, "header_row")) if _cell(row, "header_row") else 0
    if header_row is None or header_row < 0:
        return "header_row must be a non-negative integer"
    selector = _cell(row, "html_table_selector")
    if selector and _int(selector) is None:
        return "html_table_selector must be the zero-based table index"
    if _cell(row, "query") and _query(_cell(row, "query")) is None:
        return "query must be key=value pairs separated by &"
    if _cell(row, "xlsx_layout") not in {"", "wide_monthly"}:
        return "xlsx_layout must be wide_monthly or blank"
    if _cell(row, "record_layout") not in {"", "wide_commodities"}:
        return "record_layout must be wide_commodities or blank"
    return None


def _check_row_filter(value: str, names: set[str]) -> str | None:
    if not value:
        return None
    for part in value.split("&"):
        if "=" not in part:
            return "row_filter must be Column=value pairs separated by &"
        column, raw = part.split("=", 1)
        column = column.strip()
        choices = [item.strip() for item in raw.split("|") if item.strip()]
        if not column or not choices:
            return "row_filter must be Column=value pairs separated by &"
        if column not in names:
            return "row_filter columns not in column_names: " + column
    return None


def _query(value: str) -> dict[str, str] | None:
    params: dict[str, str] = {}
    for part in value.split("&"):
        if "=" not in part:
            return None
        key, item = part.split("=", 1)
        if not key.strip():
            return None
        params[key.strip()] = item.strip()
    return params


def _source(path: str, row: dict[str, str]) -> Source:
    freshness = _int(_cell(row, "freshness_days")) if _cell(row, "freshness_days") else None
    page_size = _int(_cell(row, "page_size")) if _cell(row, "page_size") else None
    return Source(
        manifest_file=path,
        source_id=_cell(row, "source_id"),
        source_name=_cell(row, "source_name"),
        source_url=_cell(row, "source_url"),
        endpoint=_cell(row, "endpoint"),
        table_name=_cell(row, "table_name"),
        indicator_class=_cell(row, "indicator_class"),
        contract_type=_cell(row, "contract_type"),
        column_count=int(_cell(row, "column_count")),
        column_names=_split(_cell(row, "column_names")),
        geo_grain=_cell(row, "geo_grain"),
        geo_country_column=_cell(row, "geo_country_column"),
        geo_country_columns=_split(_cell(row, "geo_country_columns")),
        geo_country_constant=_cell(row, "geo_country_constant"),
        geo_code_standard=_cell(row, "geo_code_standard"),
        geo_admin1_column=_cell(row, "geo_admin1_column"),
        geo_admin2_column=_cell(row, "geo_admin2_column"),
        geo_market_column=_cell(row, "geo_market_column"),
        geo_lat_column=_cell(row, "geo_lat_column"),
        geo_lon_column=_cell(row, "geo_lon_column"),
        temporal_grain=_cell(row, "temporal_grain"),
        temporal_column=_cell(row, "temporal_column"),
        temporal_format=_cell(row, "temporal_format"),
        value_column=_cell(row, "value_column"),
        unit_column=_cell(row, "unit_column"),
        unit_constant=_cell(row, "unit_constant"),
        class_columns=_split(_cell(row, "class_columns")),
        measure_kind=_cell(row, "measure_kind"),
        freshness_days=freshness,
        load_mode=_cell(row, "load_mode"),
        natural_key=_split(_cell(row, "natural_key")),
        owner=_cell(row, "owner"),
        auth_required=bool(_bool(_cell(row, "auth_required"))),
        auth_type=_cell(row, "auth_type"),
        auth_ref=_cell(row, "auth_ref"),
        auth_header=_cell(row, "auth_header"),
        auth_query_param=_cell(row, "auth_query_param"),
        records_path=_cell(row, "records_path"),
        pagination=_cell(row, "pagination") or "none",
        page_param=_cell(row, "page_param"),
        page_size=page_size,
        size_param=_cell(row, "size_param"),
        next_path=_cell(row, "next_path") or "next",
        sheet_name=_cell(row, "sheet_name"),
        header_row=_int(_cell(row, "header_row")) or 0,
        html_table_selector=_int(_cell(row, "html_table_selector")) or 0,
        query=_cell(row, "query"),
        xlsx_layout=_cell(row, "xlsx_layout"),
        record_layout=_cell(row, "record_layout"),
        file_name=_cell(row, "file_name"),
        row_filter=_cell(row, "row_filter"),
    )


def _read_csv(path: Path) -> tuple[list[Source], list[ManifestProblem], list[ManifestProblem]]:
    sources: list[Source] = []
    problems: list[ManifestProblem] = []
    skipped: list[ManifestProblem] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            problems.append(_problem(path.name, {}, "manifest has no header"))
            return sources, problems, skipped
        fieldnames = [name.strip() for name in reader.fieldnames]
        missing = [name for name in REQUIRED if name not in fieldnames]
        if missing:
            problems.append(_problem(path.name, {}, "missing columns: " + ", ".join(missing)))
            return sources, problems, skipped
        for raw in reader:
            row = {key.strip(): (value or "") for key, value in raw.items() if key}
            if _bool(_cell(row, "enabled")) is False:
                skipped.append(_problem(path.name, row, "disabled"))
                continue
            error = _validate(row)
            if error:
                problems.append(_problem(path.name, row, error))
            else:
                sources.append(_source(path.name, row))
    return sources, problems, skipped


def load_manifests(directory: Path) -> tuple[list[Source], list[ManifestProblem], list[ManifestProblem]]:
    """Load every CSV in a directory. Duplicate source ids fail every copy."""
    files = sorted(path for path in directory.glob("*.csv") if path.is_file())
    if not files:
        return [], [ManifestProblem(
            manifest_file=str(directory), source_id="", source_name="", source_url="",
            endpoint="", table_name="", indicator_class="", contract_type="",
            auth_required="", auth_type="", owner="", error="no manifest CSVs in directory",
        )], []
    sources: list[Source] = []
    problems: list[ManifestProblem] = []
    skipped: list[ManifestProblem] = []
    for path in files:
        found, failed, inactive = _read_csv(path)
        sources.extend(found)
        problems.extend(failed)
        skipped.extend(inactive)
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.source_id] = counts.get(source.source_id, 0) + 1
    unique: list[Source] = []
    for source in sources:
        if counts[source.source_id] > 1:
            problems.append(ManifestProblem(
                manifest_file=source.manifest_file,
                source_id=source.source_id,
                source_name=source.source_name,
                source_url=source.source_url,
                endpoint=source.endpoint,
                table_name=source.table_name,
                indicator_class=source.indicator_class,
                contract_type=source.contract_type,
                auth_required="true" if source.auth_required else "false",
                auth_type=source.auth_type,
                owner=source.owner,
                error="duplicate source_id across manifests",
            ))
        else:
            unique.append(source)
    return unique, problems, skipped
