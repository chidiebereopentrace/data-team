"""Fetch a declared contract: JSON API, CSV, workbook, or one HTML table."""

from __future__ import annotations

import base64
import csv
import io
import json
import re
import zipfile
from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from urllib.parse import urlsplit

from openpyxl import load_workbook
from python_calamine import WorksheetNotFound
from python_calamine import load_workbook as load_xls_workbook

from ingestion_engine.http import max_response_bytes, request_bytes
from ingestion_engine.manifest import Source

Get = Callable[[str, dict[str, str], dict[str, str]], bytes]


class FetchError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "fetch") -> None:
        super().__init__(message)
        self.stage = stage


class AuthError(FetchError):
    def __init__(self, message: str) -> None:
        super().__init__(message, stage="auth")


def auth_material(source: Source, env: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return request headers and query params. The secret stays in the environment."""
    if not source.auth_required:
        return {}, {}
    secret = env.get(source.auth_ref, "")
    if not secret:
        raise AuthError("AUTH_MISSING")
    if source.auth_type == "api_key_header":
        return {source.auth_header: secret}, {}
    if source.auth_type == "api_key_query":
        return {}, {source.auth_query_param: secret}
    if source.auth_type == "bearer":
        return {"Authorization": f"Bearer {secret}"}, {}
    if source.auth_type == "basic":
        token = base64.b64encode(secret.encode()).decode()
        return {"Authorization": f"Basic {token}"}, {}
    raise AuthError("unsupported auth_type")


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_records(header: list[str], rows: list[list[object]], expected: tuple[str, ...]) -> list[dict[str, str]]:
    if header != list(expected):
        raise FetchError(
            f"header mismatch: expected {', '.join(expected)}; got {', '.join(header)}",
            stage="shape",
        )
    records: list[dict[str, str]] = []
    for row in rows:
        values = [_text(cell) for cell in row]
        if len(values) != len(expected):
            records.append({**{name: "" for name in expected}, "_width_error": f"expected {len(expected)} columns, got {len(values)}"})
            continue
        records.append(dict(zip(expected, values, strict=True)))
    return records


def _unwrap_csv_row(row: list[str]) -> list[str]:
    """Parse a row that arrived as one quoted CSV line with doubled quotes."""
    if len(row) != 1 or '","' not in row[0]:
        return row
    inner = next(csv.reader(io.StringIO(row[0])), None)
    if inner is None or len(inner) <= 1:
        return row
    inner[-1] = inner[-1].rstrip(";")
    return inner


def _decode_csv(body: bytes) -> str:
    """Read CSV text. UTF-8 first, then Windows-1252 for bytes such as the micro sign."""
    try:
        return body.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return body.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise FetchError("csv is not UTF-8 or Windows-1252 text", stage="shape") from exc


def parse_csv(body: bytes, expected: tuple[str, ...]) -> list[dict[str, str]]:
    text = _decode_csv(body)
    lines = text.splitlines()
    if len(lines) >= 2 and lines[1].lstrip().startswith("#"):
        del lines[1]
    reader = csv.reader(io.StringIO("\n".join(lines)))
    header = next(reader, None)
    if header is None:
        raise FetchError("csv has no header", stage="shape")
    header = _unwrap_csv_row(header)
    rows = [_unwrap_csv_row(row) for row in reader if any(cell.strip() for cell in row)]
    if header and not header[0].strip():
        header = header[1:]
        rows = [row[1:] if row else row for row in rows]
    return _as_records([cell.strip() for cell in header], rows, expected)


def filter_records(records: list[dict[str, str]], row_filter: str) -> list[dict[str, str]]:
    """Keep rows that match every Column=a|b clause. A blank filter keeps every row."""
    if not row_filter:
        return records
    clauses: list[tuple[str, set[str]]] = []
    for part in row_filter.split("&"):
        column, raw = part.split("=", 1)
        clauses.append((column.strip(), {item.strip() for item in raw.split("|") if item.strip()}))
    kept: list[dict[str, str]] = []
    for record in records:
        if record.get("_width_error"):
            kept.append(record)
            continue
        if all(record.get(column, "").strip() in choices for column, choices in clauses):
            kept.append(record)
    return kept


def parse_xlsx(body: bytes, source: Source) -> list[dict[str, str]]:
    workbook = load_workbook(io.BytesIO(body), read_only=True, data_only=True)
    try:
        sheet = workbook[source.sheet_name] if source.sheet_name else workbook.worksheets[0]
        grid = [list(row) for row in sheet.iter_rows(values_only=True)]
    finally:
        workbook.close()
    if source.xlsx_layout == "wide_monthly":
        return _wide_monthly(grid, source.column_names)
    if source.header_row >= len(grid):
        raise FetchError("header_row is past the end of the sheet", stage="shape")
    header = [_text(cell) for cell in grid[source.header_row]]
    data = [row for row in grid[source.header_row + 1:] if any(_text(cell) for cell in row)]
    return _as_records(header, data, source.column_names)


def parse_xls(body: bytes, source: Source) -> list[dict[str, str]]:
    workbook = load_xls_workbook(io.BytesIO(body))
    try:
        try:
            sheet = workbook.get_sheet_by_name(source.sheet_name) if source.sheet_name else workbook.get_sheet_by_index(0)
        except WorksheetNotFound as exc:
            raise FetchError("sheet_name was not in the workbook", stage="shape") from exc
        grid = sheet.to_python()
    finally:
        workbook.close()
    if source.header_row >= len(grid):
        raise FetchError("header_row is past the end of the sheet", stage="shape")
    header = [_text(cell) for cell in grid[source.header_row]]
    data = [row for row in grid[source.header_row + 1:] if any(_text(cell) for cell in row)]
    return _as_records(header, data, source.column_names)


class _Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._depth = 0
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if self._depth == 0:
                self._rows = []
            self._depth += 1
        elif self._depth == 1 and tag == "tr":
            self._row = []
        elif self._depth == 1 and tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._depth == 1 and self._rows is not None:
            if any(self._row):
                self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._depth:
            self._depth -= 1
            if self._depth == 0 and self._rows is not None:
                self.tables.append(self._rows)
                self._rows = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def parse_html_table(body: bytes, source: Source) -> list[dict[str, str]]:
    parser = _Tables()
    parser.feed(body.decode("utf-8-sig", errors="replace"))
    if source.html_table_selector >= len(parser.tables):
        raise FetchError("html table index not found", stage="shape")
    table = parser.tables[source.html_table_selector]
    if not table:
        raise FetchError("html table is empty", stage="shape")
    header, *data = table
    return _as_records(header, data, source.column_names)


def _at(payload: object, path: str) -> object:
    current = payload
    for key in [part for part in path.split(".") if part]:
        if not isinstance(current, dict) or key not in current:
            raise FetchError(f"JSON path is missing: {path}", stage="shape")
        current = current[key]
    return current


_PERIOD = re.compile(r"\d{4}M\d{2}")
_WIDE_DIMS = frozenset({
    "ISO3", "country", "adm1_name", "adm2_name", "mkt_name", "lat", "lon", "geo_id",
    "price_date", "year", "month", "currency", "components", "start_dense_data",
    "last_survey_point", "data_coverage", "data_coverage_recent", "index_confidence_score",
    "spatially_interpolated",
})
_WIDE_PREFIXES = ("o_", "h_", "l_", "c_", "inflation_", "trust_")


def _wide_monthly(grid: list[list[object]], expected: tuple[str, ...]) -> list[dict[str, str]]:
    if list(expected) != ["period", "series", "unit", "value"]:
        raise FetchError(
            "wide_monthly expects columns period, series, unit, value",
            stage="shape",
        )
    data_start = None
    for index, row in enumerate(grid[:10]):
        cell = _text(row[0] if row else "")
        if _PERIOD.fullmatch(cell):
            data_start = index
            break
    if data_start is None or data_start < 2:
        raise FetchError("wide monthly sheet has no period row", stage="shape")
    series = [_text(cell) for cell in grid[data_start - 2][1:]]
    units = [_text(cell) for cell in grid[data_start - 1][1:]]
    records: list[dict[str, str]] = []
    for row in grid[data_start:]:
        period = _text(row[0] if row else "")
        if not _PERIOD.fullmatch(period):
            continue
        for index, name in enumerate(series):
            if not name:
                continue
            value = _text(row[index + 1] if index + 1 < len(row) else "")
            unit = units[index] if index < len(units) else ""
            records.append({"period": period, "series": name, "unit": unit, "value": value})
    return records


def _commodity_key(name: str) -> bool:
    if name in _WIDE_DIMS or name.startswith(_WIDE_PREFIXES):
        return False
    return True


def _melt_commodities(row: dict[str, object]) -> list[dict[str, str]]:
    currency = _text(row.get("currency"))
    country = _text(row.get("country"))
    market = _text(row.get("mkt_name"))
    observed = _text(row.get("price_date"))
    records: list[dict[str, str]] = []
    for name, raw in row.items():
        if not _commodity_key(name):
            continue
        price = _text(raw)
        if not price:
            continue
        records.append({
            "country": country,
            "market": market,
            "date": observed,
            "commodity": name,
            "price": price,
            "currency": currency,
        })
    return records


def _json_records(body: bytes, source: Source) -> tuple[list[dict[str, str]], object, int]:
    try:
        payload = json.loads(body.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FetchError("response is not JSON", stage="shape") from exc
    rows = _at(payload, source.records_path)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise FetchError("JSON records must be a list of objects", stage="shape")
    if source.record_layout == "wide_commodities":
        melted: list[dict[str, str]] = []
        for row in rows:
            melted.extend(_melt_commodities(row))
        return melted, payload, len(rows)
    records: list[dict[str, str]] = []
    expected = list(source.column_names)
    for row in rows:
        missing = [name for name in expected if name not in row]
        if missing:
            records.append({name: _text(row.get(name)) for name in expected} | {"_width_error": "columns differ from contract"})
            continue
        records.append({name: _text(row[name]) for name in expected})
    return records, payload, len(rows)


def _same_host(origin: str, target: str) -> str:
    if not target.startswith("https://"):
        raise FetchError("API next link must be https")
    if urlsplit(target).netloc != urlsplit(origin).netloc:
        raise FetchError("API next link left the endpoint host")
    return target


def _basename(name: str) -> str:
    return name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _table_bytes(body: bytes, source: Source, *, get: Get, headers: dict[str, str]) -> bytes:
    """Return the CSV or workbook bytes, following a file list or a ZIP when needed."""
    link = _file_link(body, source)
    if link is not None:
        body = get(link, {}, headers)
    return _member_bytes(body, source.file_name)


def _file_link(body: bytes, source: Source) -> str | None:
    payload = _json_object(body)
    if payload is None:
        return None
    dataverse = _dataverse_files(payload)
    zenodo = None if dataverse is not None else _zenodo_files(payload)
    files = dataverse if dataverse is not None else zenodo
    if files is None:
        return None
    if not source.file_name:
        raise FetchError("file_name is required for a dataset file list")
    matches = [item for item in files if _catalog_name(item, dataverse is not None) == source.file_name]
    if not matches:
        raise FetchError("file_name was not in the dataset")
    if len(matches) > 1:
        raise FetchError("file_name matched more than one file")
    if dataverse is not None:
        file_id = _dataverse_id(matches[0])
        if not file_id:
            raise FetchError("dataset file has no id")
        origin = urlsplit(source.endpoint)
        link = f"{origin.scheme}://{origin.netloc}/api/access/datafile/{file_id}"
    else:
        link = _zenodo_link(matches[0])
    return _same_host(source.endpoint, link)


def _json_object(body: bytes) -> dict[str, object] | None:
    stripped = body.lstrip()
    if not stripped.startswith(b"{") and not stripped.startswith(b"\xef\xbb\xbf{"):
        return None
    try:
        payload = json.loads(body.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _dataverse_files(payload: dict[str, object]) -> list[dict[str, object]] | None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    version = data.get("latestVersion")
    if not isinstance(version, dict):
        return None
    files = version.get("files")
    if not isinstance(files, list) or any(not isinstance(item, dict) for item in files):
        return None
    return files


def _zenodo_files(payload: dict[str, object]) -> list[dict[str, object]] | None:
    files = payload.get("files")
    if not isinstance(files, list) or any(not isinstance(item, dict) for item in files):
        return None
    if files and any("key" not in item and "links" not in item for item in files):
        return None
    return files


def _catalog_name(item: dict[str, object], dataverse: bool) -> str:
    if dataverse:
        data_file = item.get("dataFile")
        filename = ""
        if isinstance(data_file, dict):
            filename = str(data_file.get("filename") or "")
        if not filename:
            filename = str(item.get("label") or "")
        return _basename(filename)
    return _basename(str(item.get("key") or ""))


def _dataverse_id(item: dict[str, object]) -> str:
    data_file = item.get("dataFile")
    if not isinstance(data_file, dict) or data_file.get("id") is None:
        return ""
    return str(data_file["id"])


def _zenodo_link(item: dict[str, object]) -> str:
    links = item.get("links")
    if not isinstance(links, dict) or not links.get("self"):
        raise FetchError("dataset file has no https link")
    return str(links["self"])


def _member_bytes(body: bytes, file_name: str) -> bytes:
    if not body.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return body
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile as exc:
        raise FetchError("response is not a zip archive") from exc
    with archive:
        names = archive.namelist()
        if "[Content_Types].xml" in names and not file_name:
            return body
        members = [info for info in archive.infolist() if not info.is_dir()]
        chosen = _choose_member(members, file_name)
        return _read_member(archive, chosen)


def _choose_member(members: list[zipfile.ZipInfo], file_name: str) -> zipfile.ZipInfo:
    if file_name:
        matches = [info for info in members if _basename(info.filename) == file_name]
        if not matches:
            raise FetchError("file_name was not in the zip")
        if len(matches) > 1:
            raise FetchError("file_name matched more than one file")
        return matches[0]
    tables = [
        info for info in members
        if _basename(info.filename).lower().endswith((".csv", ".xlsx", ".xls"))
    ]
    if len(tables) == 1:
        return tables[0]
    if len(tables) > 1:
        raise FetchError("zip contains multiple files; set file_name")
    raise FetchError("zip has no csv or xlsx or xls")


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    limit = max_response_bytes()
    if info.file_size > limit:
        raise FetchError("response exceeds the size limit")
    with archive.open(info) as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise FetchError("response exceeds the size limit")
    return data


def iter_pages(source: Source, *, get: Get, env: dict[str, str]) -> Iterator[tuple[list[dict[str, str]], bytes]]:
    """Yield each fetched page as records plus the response body."""
    headers, params = auth_material(source, env)
    if source.query:
        for key, value in _static_query(source.query).items():
            params.setdefault(key, value)
    if source.contract_type in {"csv", "xlsx", "xls"}:
        body = _table_bytes(get(source.endpoint, params, headers), source, get=get, headers=headers)
        if source.contract_type == "csv":
            records = parse_csv(body, source.column_names)
        elif source.contract_type == "xlsx":
            records = parse_xlsx(body, source)
        else:
            records = parse_xls(body, source)
        yield filter_records(records, source.row_filter), body
        return
    if source.contract_type == "html_table":
        body = get(source.endpoint, params, headers)
        yield filter_records(parse_html_table(body, source), source.row_filter), body
        return
    if source.contract_type == "json_api":
        yield from _iter_json(source, get=get, headers=headers, params=params)
        return
    raise FetchError(f"unsupported contract {source.contract_type}")


def fetch_records(source: Source, *, get: Get, env: dict[str, str]) -> tuple[list[dict[str, str]], bytes]:
    records: list[dict[str, str]] = []
    pages: list[bytes] = []
    for page_records, body in iter_pages(source, get=get, env=env):
        records.extend(page_records)
        pages.append(body)
    return records, b"\n".join(pages)


def _iter_json(
    source: Source,
    *,
    get: Get,
    headers: dict[str, str],
    params: dict[str, str],
) -> Iterator[tuple[list[dict[str, str]], bytes]]:
    url = source.endpoint
    query = dict(params)
    position = 1 if source.pagination == "page" else 0
    for _page in range(10_000):
        page_query = dict(query)
        if source.pagination in {"page", "offset"}:
            page_query[source.page_param] = str(position)
            if source.size_param and source.page_size:
                page_query[source.size_param] = str(source.page_size)
        print(f"page\t{source.source_id}\t{position}", flush=True)
        body = get(url, page_query, headers)
        page_records, payload, api_count = _json_records(body, source)
        yield filter_records(page_records, source.row_filter), body
        if source.pagination == "none":
            break
        if source.pagination == "next_link":
            nxt = _at(payload, source.next_path) if source.next_path else None
            if not isinstance(nxt, str) or not nxt.strip():
                break
            url = _same_host(source.endpoint, nxt)
            query = {}
            continue
        if not api_count or api_count < int(source.page_size or 0):
            break
        position += 1 if source.pagination == "page" else api_count


def _static_query(value: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for part in value.split("&"):
        key, item = part.split("=", 1)
        params[key.strip()] = item.strip()
    return params


def default_get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
    return request_bytes(url, params=params or None, headers=headers or None)
