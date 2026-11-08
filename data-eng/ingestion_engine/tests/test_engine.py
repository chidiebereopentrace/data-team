"""Offline tests for manifest contracts, shaping, and the weekly report."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from openpyxl import Workbook

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from ingestion_engine.bq import attempted_source_ids, landing_destination, report_load_config, run_history, source_cursor
from ingestion_engine.cli import clear_staged, main, materialize_service_account, paired_raw_dataset
from ingestion_engine.countries import Countries, load_africa, load_countries
from ingestion_engine.fetch import parse_csv
from ingestion_engine.http import HttpError, request_bytes
from ingestion_engine.mail import send_report
from ingestion_engine.manifest import load_manifests
from ingestion_engine.run import run_manifests

COLUMNS = ["country", "market", "observed", "commodity", "price", "currency"]
HEADER = ",".join([
    "source_id", "source_name", "source_url", "endpoint", "table_name", "indicator_class",
    "contract_type", "column_count", "column_names", "geo_grain", "geo_country_column",
    "geo_code_standard", "geo_market_column", "temporal_grain", "temporal_column",
    "temporal_format", "value_column", "class_columns", "load_mode", "natural_key",
    "enabled", "owner", "auth_required", "auth_type", "auth_ref", "unit_constant",
])


def _row(**overrides: str) -> dict[str, str]:
    base = {
        "source_id": "ng_prices",
        "source_name": "Nigeria prices",
        "source_url": "https://example.test/prices",
        "endpoint": "https://example.test/prices.csv",
        "table_name": "ng_market_prices",
        "indicator_class": "PRC",
        "contract_type": "csv",
        "column_count": "6",
        "column_names": ",".join(COLUMNS),
        "geo_grain": "market",
        "geo_country_column": "country",
        "geo_code_standard": "name",
        "geo_market_column": "market",
        "temporal_grain": "day",
        "temporal_column": "observed",
        "temporal_format": "%Y-%m-%d",
        "value_column": "price",
        "class_columns": "commodity,currency",
        "load_mode": "append_snapshot",
        "natural_key": "market,observed,commodity",
        "enabled": "true",
        "owner": "data-eng",
        "auth_required": "false",
        "auth_type": "none",
        "auth_ref": "",
        "unit_constant": "NGN/kg",
        "html_table_selector": "",
    }
    base.update(overrides)
    return base


def _write_manifest(directory: Path, name: str, rows: list[dict[str, str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _places() -> Countries:
    places = Countries()
    places.names["NGA"] = "Nigeria"
    places.to_iso3["nigeria"] = "NGA"
    places.to_iso3["nga"] = "NGA"
    return places


def _clock() -> datetime:
    return datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)


def _csv_body() -> bytes:
    return (
        "country,market,observed,commodity,price,currency\n"
        "Nigeria,Lagos,2024-05-01,Maize,100,NGN\n"
        "Nigeria,Lagos,not-a-date,Maize,100,NGN\n"
    ).encode()


def test_gated_row_without_secret_name_is_rejected(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "sources.csv", [_row(auth_required="true", auth_type="none", auth_ref="")])
    sources, problems, skipped = load_manifests(tmp_path)
    assert sources == []
    assert skipped == []
    assert problems[0].error == "a gated endpoint needs auth_type and auth_ref"


def test_duplicate_source_id_fails_every_copy(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "a.csv", [_row()])
    _write_manifest(tmp_path, "b.csv", [_row(endpoint="https://example.test/other.csv")])
    sources, problems, _skipped = load_manifests(tmp_path)
    assert sources == []
    assert len(problems) == 2
    assert {item.error for item in problems} == {"duplicate source_id across manifests"}


def test_csv_run_lands_good_rows_and_reports_the_bad_date(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    out = tmp_path / "out"
    _write_manifest(manifests, "prices.csv", [_row()])

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        assert url == "https://example.test/prices.csv"
        assert headers == {}
        return _csv_body()

    code = run_manifests(manifests, out, get=get, env={}, countries=_places(), clock=_clock)
    assert code == 1
    landing = [json.loads(line) for line in next((out / "landing").glob("*/*.jsonl")).read_text(encoding="utf-8").splitlines()]
    prepared = [json.loads(line) for line in next((out / "prepared").glob("*/*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert len(landing) == 2
    assert "country_iso3" not in landing[0]
    rows = prepared
    assert rows[0]["country_iso3"] == "NGA"
    assert rows[0]["market_name"] == "Lagos"
    assert rows[0]["observation_date"] == "2024-05-01"
    assert rows[0]["year"] == 2024
    assert rows[0]["value_num"] == 100
    assert rows[0]["indicator_class"] == "PRC"
    coverage = list(csv.DictReader((out / "reports" / "ingest_coverage.csv").open(encoding="utf-8")))
    assert coverage[0]["indicator_class"] == "PRC"
    assert coverage[0]["country_iso3"] == "NGA"
    assert coverage[0]["year"] == "2024"
    assert coverage[0]["rows_loaded"] == "1"
    failures = list(csv.DictReader((out / "reports" / "ingest_failures.csv").open(encoding="utf-8")))
    assert failures[0]["status"] == "PARTIAL"
    assert failures[0]["failure_stage"] == "shape"
    sources = list(csv.DictReader((out / "reports" / "ingest_sources.csv").open(encoding="utf-8")))
    assert sources[0]["quarantine_reasons"] == "unparsed observation date=1"
    raw = next((out / "raw").glob("*/*.bin"))
    assert raw.read_bytes() == _csv_body()


def test_missing_secret_is_an_auth_failure(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "api.csv", [_row(
        contract_type="json_api",
        endpoint="https://example.test/api",
        auth_required="true",
        auth_type="bearer",
        auth_ref="PRICE_API_TOKEN",
        records_path="data",
    )])

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        raise AssertionError("gated row must not be fetched without a secret")

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 1
    failures = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_failures.csv").open(encoding="utf-8")))
    assert failures[0]["failure_stage"] == "auth"
    assert failures[0]["error"] == "AUTH_MISSING"


def test_disabled_row_is_skipped(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "off.csv", [_row(enabled="false", source_url="not a url")])
    code = run_manifests(manifests, tmp_path / "out", get=lambda *args: b"", env={}, countries=_places(), clock=_clock)
    assert code == 0
    summary = json.loads((tmp_path / "out" / "reports" / "run.json").read_text(encoding="utf-8"))
    assert summary["sources_skipped"] == 1
    assert summary["status"] == "FINISHED"
    failures = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_failures.csv").open(encoding="utf-8")))
    assert failures[0]["status"] == "SKIPPED"
    assert failures[0]["owner"] == "data-eng"
    sources = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_sources.csv").open(encoding="utf-8")))
    assert sources[0]["status"] == "SKIPPED"
    assert sources[0]["owner"] == "data-eng"


def test_already_done_source_is_not_fetched(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [
        _row(),
        _row(
            source_id="ke_prices",
            source_name="Kenya prices",
            endpoint="https://example.test/ke.csv",
            table_name="ke_market_prices",
        ),
    ])
    places = _places()
    places.names["KEN"] = "Kenya"
    places.to_iso3["kenya"] = "KEN"
    fetched: list[str] = []

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        fetched.append(url)
        return (
            "country,market,observed,commodity,price,currency\n"
            "Kenya,Nairobi,2024-05-01,Maize,100,KES\n"
        ).encode()

    code = run_manifests(
        manifests, tmp_path / "out", get=get, env={}, countries=places, clock=_clock, done={"ng_prices"},
    )
    assert code == 0
    assert fetched == ["https://example.test/ke.csv"]
    summary = json.loads((tmp_path / "out" / "reports" / "run.json").read_text(encoding="utf-8"))
    assert summary["sources_already_done"] == 1
    assert summary["status"] == "FINISHED"
    sources = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_sources.csv").open(encoding="utf-8")))
    assert [row["source_id"] for row in sources] == ["ke_prices"]


def test_run_finishes_when_every_source_is_already_done(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [_row()])

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        raise AssertionError("already ingested source must not be fetched")

    code = run_manifests(
        manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock, done={"ng_prices"},
    )
    assert code == 0
    summary = json.loads((tmp_path / "out" / "reports" / "run.json").read_text(encoding="utf-8"))
    assert summary["sources_already_done"] == 1
    assert summary["sources_ingested"] == 0
    assert summary["status"] == "FINISHED"


def test_http_error_fails_one_source_and_the_next_source_still_runs(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [
        _row(),
        _row(
            source_id="ke_prices",
            source_name="Kenya prices",
            endpoint="https://example.test/ke.csv",
            table_name="ke_market_prices",
        ),
    ])
    places = _places()
    places.names["KEN"] = "Kenya"
    places.to_iso3["kenya"] = "KEN"
    fetched: list[str] = []

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        fetched.append(url)
        if url.endswith("prices.csv"):
            raise HttpError("HTTP 404", status_code=404)
        return (
            "country,market,observed,commodity,price,currency\n"
            "Kenya,Nairobi,2024-05-01,Maize,100,KES\n"
        ).encode()

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=places, clock=_clock)
    assert code == 1
    assert fetched == ["https://example.test/prices.csv", "https://example.test/ke.csv"]
    sources = _loaded_rows(tmp_path / "out")
    assert [(row["source_id"], row["status"], row["error"]) for row in sources] == [
        ("ng_prices", "FAILED", "HTTP 404"),
        ("ke_prices", "LOADED", ""),
    ]


def test_unexpected_error_fails_one_source_and_the_next_source_still_runs(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [
        _row(),
        _row(
            source_id="ke_prices",
            source_name="Kenya prices",
            endpoint="https://example.test/ke.csv",
            table_name="ke_market_prices",
        ),
    ])
    places = _places()
    places.names["KEN"] = "Kenya"
    places.to_iso3["kenya"] = "KEN"

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        if url.endswith("prices.csv"):
            raise RuntimeError("disk full")
        return (
            "country,market,observed,commodity,price,currency\n"
            "Kenya,Nairobi,2024-05-01,Maize,100,KES\n"
        ).encode()

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=places, clock=_clock)
    assert code == 1
    sources = _loaded_rows(tmp_path / "out")
    assert sources[0]["error"] == "disk full"
    assert sources[1]["status"] == "LOADED"


def test_started_source_is_closed_without_a_download(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "a.csv", [_row()])
    _write_manifest(manifests, "b.csv", [_row(
        source_id="ke_prices",
        source_name="Kenya prices",
        endpoint="https://example.test/ke.csv",
        table_name="ke_market_prices",
    )])
    places = _places()
    places.names["KEN"] = "Kenya"
    places.to_iso3["kenya"] = "KEN"
    fetched: list[str] = []

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        fetched.append(url)
        return (
            "country,market,observed,commodity,price,currency\n"
            "Kenya,Nairobi,2024-05-01,Maize,100,KES\n"
        ).encode()

    code = run_manifests(
        manifests,
        tmp_path / "out",
        get=get,
        env={},
        countries=places,
        clock=_clock,
        stopped={"ng_prices"},
    )
    assert code == 1
    assert fetched == ["https://example.test/ke.csv"]
    sources = _loaded_rows(tmp_path / "out")
    assert [(row["source_id"], row["status"], row["error"]) for row in sources] == [
        ("ng_prices", "FAILED", "stopped before finish"),
        ("ke_prices", "LOADED", ""),
    ]


def test_missing_report_table_has_no_attempted_sources() -> None:
    class Missing:
        def query(self, query: str) -> object:
            raise NotFound("ingest_sources")

    assert attempted_source_ids(Missing(), "opentrace.landing_dev.ingest_sources") == set()


def test_cursor_keeps_finished_sources_and_names_unfinished_starts() -> None:
    class Row:
        def __init__(self, source_id: str, status: str) -> None:
            self.source_id = source_id
            self.status = status

    class Client:
        def query(self, query: str) -> "Client":
            assert "STARTED" in query
            return self

        def result(self) -> list[Row]:
            return [
                Row("ng_prices", "STARTED"),
                Row("ng_prices", "LOADED"),
                Row("ke_prices", "STARTED"),
                Row("gh_prices", "FAILED"),
                Row("", "STARTED"),
            ]

    finished, started = source_cursor(Client(), "opentrace.landing_dev.ingest_sources")
    assert finished == {"ng_prices", "gh_prices"}
    assert started == {"ke_prices"}


def test_missing_source_table_has_no_cursor() -> None:
    class Missing:
        def query(self, query: str) -> object:
            raise NotFound("ingest_sources")

    assert source_cursor(Missing(), "opentrace.landing_dev.ingest_sources") == (set(), set())


def test_missing_run_table_has_no_history() -> None:
    class Missing:
        def query(self, query: str) -> object:
            raise NotFound("ingest_runs")

    assert run_history(Missing(), "opentrace.landing_dev.ingest_runs") == []


def test_html_table_and_xlsx_use_the_declared_columns(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    book = Workbook()
    sheet = book.active
    sheet.append(COLUMNS)
    sheet.append(["Nigeria", "Kano", "2024-06-01", "Rice", "80", "NGN"])
    xlsx = tmp_path / "book.xlsx"
    book.save(xlsx)
    html = (
        "<html><body><table><tr><td>ignore</td></tr></table>"
        "<table><tr><th>country</th><th>market</th><th>observed</th><th>commodity</th><th>price</th><th>currency</th></tr>"
        "<tr><td>Nigeria</td><td>Abuja</td><td>2023-01-15</td><td>Yam</td><td>50</td><td>NGN</td></tr>"
        "</table></body></html>"
    ).encode()
    _write_manifest(manifests, "files.csv", [
        _row(source_id="ng_xlsx", endpoint="https://example.test/book.xlsx", contract_type="xlsx", table_name="ng_xlsx_prices"),
        _row(
            source_id="ng_html",
            endpoint="https://example.test/page",
            contract_type="html_table",
            table_name="ng_html_prices",
            html_table_selector="1",
        ),
    ])

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        if url.endswith(".xlsx"):
            return xlsx.read_bytes()
        return html

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 0
    coverage = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_coverage.csv").open(encoding="utf-8")))
    assert {row["year"] for row in coverage} == {"2024", "2023"}
    assert sum(int(row["rows_loaded"]) for row in coverage) == 2


TABLE = b"country,market,observed,commodity,price,currency\nNigeria,Lagos,2024-05-01,Maize,100,NGN\n"


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _loaded_rows(out: Path) -> list[dict[str, str]]:
    return list(csv.DictReader((out / "reports" / "ingest_sources.csv").open(encoding="utf-8")))


def _failure_text(out: Path) -> str:
    rows = list(csv.DictReader((out / "reports" / "ingest_failures.csv").open(encoding="utf-8")))
    return rows[0]["error"]


def test_direct_csv_without_file_name_still_parses(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [_row()])
    code = run_manifests(
        manifests, tmp_path / "out", get=lambda *args: TABLE, env={}, countries=_places(), clock=_clock,
    )
    assert code == 0
    assert _loaded_rows(tmp_path / "out")[0]["rows_loaded"] == "1"


def test_file_name_must_be_one_path_segment(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "sources.csv", [_row(file_name="folder/prices.csv")])
    sources, problems, _skipped = load_manifests(tmp_path)
    assert sources == []
    assert problems[0].error == "file_name must be a single file name"


def test_one_csv_inside_a_zip_is_the_table(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [_row(endpoint="https://example.test/pack.zip")])
    packed = _zip_bytes({"notes.txt": b"ignore", "prices.csv": TABLE})
    code = run_manifests(
        manifests, tmp_path / "out", get=lambda *args: packed, env={}, countries=_places(), clock=_clock,
    )
    assert code == 0
    assert _loaded_rows(tmp_path / "out")[0]["rows_loaded"] == "1"


def test_a_zip_with_two_tables_requires_file_name(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    packed = _zip_bytes({"a.csv": TABLE, "b.csv": TABLE})
    _write_manifest(manifests, "prices.csv", [
        _row(source_id="ng_open", endpoint="https://example.test/open.zip", file_name=""),
        _row(source_id="ng_named", endpoint="https://example.test/named.zip", file_name="b.csv", table_name="ng_named_prices"),
    ])

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        return packed

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 1
    rows = {row["source_id"]: row for row in _loaded_rows(tmp_path / "out")}
    assert rows["ng_open"]["status"] == "FAILED"
    assert rows["ng_open"]["error"] == "zip contains multiple files; set file_name"
    assert rows["ng_named"]["status"] == "LOADED"
    assert rows["ng_named"]["rows_loaded"] == "1"


def test_dataverse_file_list_downloads_the_named_file(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    endpoint = "https://dataverse.harvard.edu/api/datasets/:persistentId"
    _write_manifest(manifests, "prices.csv", [_row(endpoint=endpoint, file_name="prices.csv")])
    catalog = json.dumps({
        "data": {"latestVersion": {"files": [
            {"dataFile": {"id": 7, "filename": "other.csv"}},
            {"dataFile": {"id": 8, "filename": "prices.csv"}},
        ]}},
    }).encode()

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        if url == endpoint:
            return catalog
        assert url == "https://dataverse.harvard.edu/api/access/datafile/8"
        assert params == {}
        return TABLE

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 0
    assert _loaded_rows(tmp_path / "out")[0]["rows_loaded"] == "1"


def test_zenodo_file_list_downloads_the_named_file(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    endpoint = "https://zenodo.org/api/records/1"
    _write_manifest(manifests, "prices.csv", [_row(endpoint=endpoint, file_name="prices.csv")])
    catalog = json.dumps({
        "files": [{"key": "prices.csv", "links": {"self": "https://zenodo.org/api/records/1/files/prices.csv/content"}}],
    }).encode()

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        if url == endpoint:
            return catalog
        assert url == "https://zenodo.org/api/records/1/files/prices.csv/content"
        return TABLE

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 0
    assert _loaded_rows(tmp_path / "out")[0]["rows_loaded"] == "1"


def test_dataset_file_link_must_stay_on_the_endpoint_host(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    endpoint = "https://zenodo.org/api/records/1"
    _write_manifest(manifests, "prices.csv", [_row(endpoint=endpoint, file_name="prices.csv")])
    catalog = json.dumps({
        "files": [{"key": "prices.csv", "links": {"self": "https://files.example.test/prices.csv"}}],
    }).encode()

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        assert url == endpoint
        return catalog

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 1
    assert _failure_text(tmp_path / "out") == "API next link left the endpoint host"


def test_json_pages_follow_page_size(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "api.csv", [_row(
        source_id="ng_api",
        contract_type="json_api",
        endpoint="https://example.test/api",
        table_name="ng_api_prices",
        records_path="data",
        pagination="page",
        page_param="page",
        page_size="1",
        size_param="limit",
        auth_required="true",
        auth_type="api_key_header",
        auth_ref="PRICE_API_TOKEN",
        auth_header="X-Api-Key",
    )])
    seen: list[dict[str, str]] = []

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        seen.append(params)
        assert headers["X-Api-Key"] == "secret"
        page = params["page"]
        if page == "1":
            row = {"country": "Nigeria", "market": "Lagos", "observed": "2024-01-01", "commodity": "Maize", "price": "10", "currency": "NGN"}
            return json.dumps({"data": [row]}).encode()
        if page == "2":
            row = {"country": "Nigeria", "market": "Kano", "observed": "2024-02-01", "commodity": "Rice", "price": "12", "currency": "NGN"}
            return json.dumps({"data": [row]}).encode()
        return json.dumps({"data": []}).encode()

    code = run_manifests(
        manifests, tmp_path / "out", get=get, env={"PRICE_API_TOKEN": "secret"}, countries=_places(), clock=_clock,
    )
    assert code == 0
    assert [item["page"] for item in seen] == ["1", "2", "3"]
    landing = list(json.loads(line) for line in next((tmp_path / "out" / "landing").glob("*/*.jsonl")).read_text().splitlines())
    assert len(landing) == 2


def test_landing_destination_keeps_snapshots_or_replaces_one_day() -> None:
    appended, mode = landing_destination("opentrace", "landing_dev", "ng_market_prices", "append_snapshot", "2026-11-02")
    assert appended == "opentrace.landing_dev.ng_market_prices"
    assert mode == "WRITE_APPEND"
    replaced, write = landing_destination("opentrace", "landing_dev", "ng_market_prices", "replace_partition", "2026-11-02")
    assert replaced == "opentrace.landing_dev.ng_market_prices$20261102"
    assert write == "WRITE_TRUNCATE"


def test_header_mismatch_fails_the_ref(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "bad.csv", [_row()])

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        return b"a,b\n1,2\n"

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 1
    failures = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_failures.csv").open(encoding="utf-8")))
    assert failures[0]["failure_stage"] == "shape"
    assert "header mismatch" in failures[0]["error"]


def test_hxl_second_line_is_not_the_header(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    fixture = Path(__file__).resolve().parents[2] / "ingestion" / "tests" / "fixtures" / "hdx_wfp_nga_sample.csv"
    names = "date,market_id,market,commodity_id,commodity,pricetype,unit,price,currency,usdprice,latitude,longitude"
    _write_manifest(manifests, "wfp.csv", [_row(
        column_count="12",
        column_names=names,
        geo_country_column="",
        geo_country_constant="Nigeria",
        temporal_column="date",
        class_columns="commodity,currency",
        unit_column="unit",
        unit_constant="",
        natural_key="date,market,commodity,pricetype,unit",
    )])

    code = run_manifests(
        manifests, tmp_path / "out", get=lambda *args: fixture.read_bytes(), env={}, countries=_places(), clock=_clock,
    )
    assert code == 0
    prepared = [json.loads(line) for line in next((tmp_path / "out" / "prepared").glob("*/*.jsonl")).read_text().splitlines()]
    assert prepared[0]["country_iso3"] == "NGA"
    assert prepared[0]["commodity"] == "Maize"


def test_extra_json_keys_are_dropped(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "api.csv", [_row(
        contract_type="json_api",
        endpoint="https://example.test/api",
        records_path="data",
        query="DATABASE_ID=WB_HCP",
    )])
    seen: list[dict[str, str]] = []

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        seen.append(params)
        row = {
            "country": "Nigeria", "market": "Lagos", "observed": "2024-01-01",
            "commodity": "Maize", "price": "10", "currency": "NGN", "note": "extra",
        }
        return json.dumps({"data": [row]}).encode()

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 0
    assert seen[0]["DATABASE_ID"] == "WB_HCP"
    prepared = json.loads(next((tmp_path / "out" / "prepared").glob("*/*.jsonl")).read_text().splitlines()[0])
    assert "note" not in prepared


def test_blank_measure_stays_in_landing_and_out_of_raw(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [_row()])

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        return (
            "country,market,observed,commodity,price,currency\n"
            "Nigeria,Lagos,2024-05-01,,100,NGN\n"
            "Nigeria,Lagos,2024-05-02,Maize,100,NGN\n"
        ).encode()

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 1
    landing = [json.loads(line) for line in next((tmp_path / "out" / "landing").glob("*/*.jsonl")).read_text().splitlines()]
    prepared = [json.loads(line) for line in next((tmp_path / "out" / "prepared").glob("*/*.jsonl")).read_text().splitlines()]
    assert len(landing) == 2
    assert len(prepared) == 1
    assert prepared[0]["commodity"] == "Maize"


def test_duplicate_natural_key_is_kept_once(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [_row()])

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        line = "Nigeria,Lagos,2024-05-01,Maize,100,NGN\n"
        return ("country,market,observed,commodity,price,currency\n" + line + line).encode()

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=_places(), clock=_clock)
    assert code == 0
    landing = next((tmp_path / "out" / "landing").glob("*/*.jsonl")).read_text().splitlines()
    assert len(landing) == 1
    sources = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_sources.csv").open(encoding="utf-8")))
    assert sources[0]["status"] == "LOADED"
    assert sources[0]["owner"] == "data-eng"


def test_pink_sheet_melts_the_wide_month(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    fixture = Path(__file__).resolve().parents[2] / "ingestion" / "tests" / "fixtures" / "pink_sheet_sample.xlsx"
    _write_manifest(manifests, "pink.csv", [_row(
        source_id="wb_pink_sheet",
        contract_type="xlsx",
        endpoint="https://example.test/pink.xlsx",
        table_name="wb_pink_sheet",
        column_count="4",
        column_names="period,series,unit,value",
        geo_grain="country",
        geo_country_column="",
        geo_country_constant="Nigeria",
        geo_market_column="",
        temporal_grain="month",
        temporal_column="period",
        temporal_format="%YM%m",
        value_column="value",
        class_columns="series,unit",
        unit_column="unit",
        unit_constant="",
        natural_key="period,series",
        sheet_name="Monthly Prices",
        xlsx_layout="wide_monthly",
    )])
    code = run_manifests(
        manifests, tmp_path / "out", get=lambda *args: fixture.read_bytes(), env={}, countries=load_countries(), clock=_clock,
    )
    assert code == 0
    prepared = [json.loads(line) for line in next((tmp_path / "out" / "prepared").glob("*/*.jsonl")).read_text().splitlines()]
    assert prepared[0]["country_iso3"] == "NGA"
    assert prepared[0]["observation_date"] == "2024-01-01"
    assert prepared[0]["series"] == "Crude oil, average"


def test_report_append_allows_a_new_column() -> None:
    config = report_load_config(["sources_ingested", "status", "year"])
    assert config.autodetect is False
    assert bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION in config.schema_update_options
    kinds = {field.name: field.field_type for field in config.schema}
    assert kinds["sources_ingested"] == "INT64"
    assert kinds["year"] == "INT64"
    assert kinds["status"] == "STRING"


def test_replace_table_truncates_the_whole_table() -> None:
    destination, mode = landing_destination("opentrace", "landing_dev", "ng_market_prices", "replace_table", "2026-11-02")
    assert destination == "opentrace.landing_dev.ng_market_prices"
    assert mode == "WRITE_TRUNCATE"


def test_world_resolves_from_the_packaged_country_file() -> None:
    assert load_countries().resolve("World", "name") == "WLD"


def test_landing_pairs_with_raw() -> None:
    assert paired_raw_dataset("landing_dev") == "raw_dev"
    assert paired_raw_dataset("landing") == "raw"


def test_service_account_json_is_written_without_being_printed(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("BQ_SERVICE_ACCOUNT_JSON", '{"client_email":"ingest@example.test"}')
    path = materialize_service_account()
    captured = capsys.readouterr()
    assert path is not None
    assert "client_email" not in captured.out
    assert path.read_text(encoding="utf-8") == '{"client_email":"ingest@example.test"}'
    path.unlink()


def test_africa_list_has_the_54_and_leaves_out_the_rest() -> None:
    codes = load_africa()
    assert len(codes) == 54
    assert "NGA" in codes
    assert "FRA" not in codes
    assert "ESH" not in codes
    assert "WLD" not in codes


def test_country_outside_africa_is_left_out_and_does_not_fail_the_source(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [_row()])
    places = _places()
    places.names["FRA"] = "France"
    places.to_iso3["france"] = "FRA"

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        return (
            "country,market,observed,commodity,price,currency\n"
            "Nigeria,Lagos,2024-05-01,Maize,100,NGN\n"
            "France,Paris,2024-05-01,Maize,100,EUR\n"
        ).encode()

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=places, clock=_clock)
    assert code == 0
    landing = [json.loads(line) for line in next((tmp_path / "out" / "landing").glob("*/*.jsonl")).read_text().splitlines()]
    prepared = [json.loads(line) for line in next((tmp_path / "out" / "prepared").glob("*/*.jsonl")).read_text().splitlines()]
    assert [row["country"] for row in landing] == ["Nigeria"]
    assert prepared[0]["country_iso3"] == "NGA"
    sources = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_sources.csv").open(encoding="utf-8")))
    assert sources[0]["status"] == "LOADED"
    assert sources[0]["rows_outside_africa"] == "1"
    assert sources[0]["rows_loaded"] == "1"


def test_partner_country_column_is_kept(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "trade.csv", [_row(
        column_count="7",
        column_names="country,market,observed,commodity,price,currency,partner",
        geo_country_columns="partner",
        natural_key="market,observed,commodity,partner",
    )])
    places = _places()
    places.names["KEN"] = "Kenya"
    places.to_iso3["kenya"] = "KEN"

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        return (
            "country,market,observed,commodity,price,currency,partner\n"
            "Kenya,Nairobi,2024-05-01,Maize,100,KES,China\n"
        ).encode()

    code = run_manifests(manifests, tmp_path / "out", get=get, env={}, countries=places, clock=_clock)
    assert code == 0
    landing = [json.loads(line) for line in next((tmp_path / "out" / "landing").glob("*/*.jsonl")).read_text().splitlines()]
    prepared = [json.loads(line) for line in next((tmp_path / "out" / "prepared").glob("*/*.jsonl")).read_text().splitlines()]
    assert landing[0]["country"] == "Kenya"
    assert landing[0]["partner"] == "China"
    assert prepared[0]["country_iso3"] == "KEN"
    assert prepared[0]["partner"] == "China"


def test_country_column_missing_from_column_names_is_rejected(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "sources.csv", [_row(geo_country_columns="partner")])
    sources, problems, _skipped = load_manifests(tmp_path)
    assert sources == []
    assert "partner" in problems[0].error


def test_world_constant_produces_no_prepared_rows(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "world.csv", [_row(
        geo_grain="country",
        geo_country_column="",
        geo_country_constant="World",
        geo_market_column="",
    )])
    code = run_manifests(
        manifests, tmp_path / "out", get=lambda *args: _csv_body(), env={}, countries=load_countries(), clock=_clock,
    )
    assert code == 1
    assert list((tmp_path / "out" / "prepared").glob("*/*.jsonl")) == []
    assert list((tmp_path / "out" / "landing").glob("*/*.jsonl")) == []
    sources = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_sources.csv").open(encoding="utf-8")))
    assert sources[0]["rows_loaded"] == "0"
    assert int(sources[0]["rows_outside_africa"]) > 0


class _FakeSMTP:
    sent: list[object] = []

    def __init__(self, host: str, port: int, timeout: int = 30) -> None:
        self.host = host
        self.port = port

    def __enter__(self) -> "_FakeSMTP":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def ehlo(self) -> None:
        return None

    def starttls(self) -> None:
        return None

    def login(self, user: str, password: str) -> None:
        self.password = password

    def send_message(self, message: object) -> None:
        _FakeSMTP.sent.append(message)


def test_report_email_states_sources_rows_and_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    manifests = tmp_path / "manifests"
    out = tmp_path / "out"
    _write_manifest(manifests, "prices.csv", [_row()])
    run_manifests(manifests, out, get=lambda *args: _csv_body(), env={}, countries=_places(), clock=_clock)
    _FakeSMTP.sent = []
    monkeypatch.setenv("INGEST_REPORT_EMAILS", "a@example.test, b@example.test")
    monkeypatch.setenv("INGEST_SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("INGEST_SMTP_PORT", "587")
    monkeypatch.setenv("INGEST_SMTP_USER", "ingest")
    monkeypatch.setenv("INGEST_SMTP_PASSWORD", "secret")
    monkeypatch.setenv("INGEST_SMTP_FROM", "ingest@example.test")
    send_report(
        out,
        history=[{"run_id": "prev-run", "started_at": "2026-10-26T02:00:00+00:00", "rows_loaded": 40}],
        smtp=_FakeSMTP,
    )
    captured = capsys.readouterr()
    assert "secret" not in captured.out
    message = _FakeSMTP.sent[0]
    body = message.get_body(preferencelist=("plain",)).get_content()
    assert "Weekly Data Ingestion Report" in body
    assert "Rows added: 1" in body
    assert "Success rate: 50.0%" in body
    assert "ng_prices" in body
    assert "data-eng" in body
    page = (out / "reports" / "ingest_report.html").read_text(encoding="utf-8")
    summary = json.loads((out / "reports" / "run.json").read_text(encoding="utf-8"))
    assert "Rows added" in page
    assert "NGA" in page
    assert "prev-run" in page
    assert summary["run_id"] in page
    assert (out / "reports" / "ingest_report.pdf").read_bytes().startswith(b"%PDF")
    html_body = message.get_body(preferencelist=("html",)).get_content()
    assert "Weekly Data Ingestion Report" in html_body
    assert "Rows added" in html_body
    assert "ng_prices" in html_body
    jobs = (out / "reports" / "ingest_jobs.html").read_text(encoding="utf-8")
    assert "ng_prices" in jobs
    assert "50.0%" in jobs
    assert 'data-filter="status"' in jobs
    assert 'data-coverage="class"' in jobs
    assert "NGA" in jobs
    assert "Not measured" in jobs
    attachments = list(message.iter_attachments())
    assert any(part.get_filename() == "ingest_jobs.html" for part in attachments)
    pdf = next(part.get_payload(decode=True) for part in attachments if part.get_filename() == "ingest_report.pdf")
    assert pdf.startswith(b"%PDF")
    coverage = next(part.get_content() for part in attachments if part.get_filename() == "ingest_coverage.csv")
    assert "indicator_class" in coverage
    assert "NGA" in coverage
    assert "2024" in coverage


def test_report_email_is_skipped_without_recipients(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_REPORT_EMAILS", raising=False)

    def connect(*args: object, **kwargs: object) -> object:
        raise AssertionError("smtp")

    send_report(tmp_path, smtp=connect)


def test_execute_records_started_before_the_next_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifests = tmp_path / "manifests"
    out = tmp_path / "out"
    _write_manifest(manifests, "a.csv", [_row()])
    _write_manifest(manifests, "b.csv", [_row(
        source_id="ke_prices",
        source_name="Kenya prices",
        endpoint="https://example.test/ke.csv",
        table_name="ke_market_prices",
    )])
    _write_manifest(manifests, "c.csv", [_row(
        source_id="gh_prices",
        source_name="Ghana prices",
        endpoint="https://example.test/gh.csv",
        table_name="gh_market_prices",
    )])
    events: list[str] = []

    class Job:
        output_rows = 1

        def result(self) -> None:
            return None

    class Client:
        def __init__(self, project: str) -> None:
            self.project = project

        def query(self, sql: str) -> object:
            if "ingest_runs" in sql:
                raise NotFound("ingest_runs")

            class Answer:
                def result(self) -> list[object]:
                    return [
                        type("Row", (), {"source_id": "ng_prices", "status": "LOADED"})(),
                        type("Row", (), {"source_id": "ke_prices", "status": "STARTED"})(),
                    ]

            return Answer()

        def get_table(self, table_id: str) -> None:
            raise NotFound(table_id)

        def create_table(self, table: object) -> object:
            return table

        def load_table_from_file(self, handle: object, destination: str, job_config: object) -> Job:
            body = handle.read()
            assert isinstance(body, bytes)
            events.append(body.decode("utf-8"))
            return Job()

    def get(url: str, params: dict[str, str], headers: dict[str, str]) -> bytes:
        events.append(f"fetch {url}")
        return (
            "country,market,observed,commodity,price,currency\n"
            "Nigeria,Lagos,2024-05-01,Maize,100,NGN\n"
        ).encode()

    monkeypatch.setattr("ingestion_engine.cli.load_env_file", lambda path: None)
    monkeypatch.setattr("ingestion_engine.cli.materialize_service_account", lambda: None)
    monkeypatch.setattr("ingestion_engine.cli.send_report", lambda *args, **kwargs: None)
    monkeypatch.setattr("ingestion_engine.cli.bigquery.Client", Client)
    monkeypatch.setattr("ingestion_engine.run.default_get", get)
    monkeypatch.setenv("BQ_PROJECT", "opentrace-test")
    monkeypatch.setenv("INGEST_TARGET_DATASET", "landing_dev")
    monkeypatch.delenv("INGEST_REPORT_DATASET", raising=False)
    monkeypatch.delenv("INGEST_REPORT_EMAILS", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("BQ_SERVICE_ACCOUNT_JSON", raising=False)

    code = main(["--execute", "--manifests", str(manifests), "--out", str(out)])
    assert code == 1
    fetch_at = next(index for index, item in enumerate(events) if item.startswith("fetch "))
    before = "\n".join(events[:fetch_at])
    assert "ke_prices" in before
    assert "stopped before finish" in before
    assert "gh_prices" in before
    assert "STARTED" in before
    assert events[fetch_at] == "fetch https://example.test/gh.csv"
    assert [item for item in events if item.startswith("fetch ")] == ["fetch https://example.test/gh.csv"]
    after = "\n".join(events[fetch_at + 1:])
    assert "LOADED" in after


def test_railway_start_clears_staging_and_does_not_restart() -> None:
    config = json.loads((Path(__file__).parents[1] / "railway.json").read_text(encoding="utf-8"))
    command = config["deploy"]["startCommand"]
    assert "rm -rf /storage/landing /storage/prepared /storage/quarantine /storage/raw" in command
    assert "ingestion-engine --execute" in command
    assert config["deploy"]["restartPolicyType"] == "NEVER"
    assert config["deploy"]["cronSchedule"] == "0 2 * * 0"


def test_clear_staged_keeps_the_reports(tmp_path: Path) -> None:
    for name in ("landing", "prepared", "quarantine", "raw", "reports"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "keep.txt").write_text("x", encoding="utf-8")
    clear_staged(tmp_path)
    for name in ("landing", "prepared", "quarantine", "raw"):
        assert not (tmp_path / name).exists()
    assert (tmp_path / "reports" / "keep.txt").is_file()


def test_indicator_without_a_measure_column_is_rejected(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "sources.csv", [_row(indicator_class="HDI", class_columns="")])
    sources, problems, _skipped = load_manifests(tmp_path)
    assert sources == []
    assert "reported measure" in problems[0].error


def test_xls_inside_a_zip_uses_the_declared_columns(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample.xls"
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "files.csv", [
        _row(
            source_id="ng_xls",
            endpoint="https://example.test/book.zip",
            contract_type="xls",
            table_name="ng_xls_prices",
            file_name="sample.xls",
        ),
    ])
    payload = _zip_bytes({"folder/sample.xls": fixture.read_bytes()})
    code = run_manifests(
        manifests, tmp_path / "out", get=lambda *_args: payload, env={}, countries=_places(), clock=_clock,
    )
    assert code == 0
    coverage = list(csv.DictReader((tmp_path / "out" / "reports" / "ingest_coverage.csv").open(encoding="utf-8")))
    assert sum(int(row["rows_loaded"]) for row in coverage) == 1


def _wrap_csv_line(line: str) -> str:
    return '"' + line.replace('"', '""') + '"'


def test_doubled_quote_csv_parses_as_columns() -> None:
    header = '"country","market","observed","commodity","price","currency"'
    row = '"Nigeria","Lagos","2024-05-01","Maize","100","NGN"'
    body = (_wrap_csv_line(header) + "\n" + _wrap_csv_line(row) + "\n").encode()
    records = parse_csv(body, tuple(COLUMNS))
    assert records[0]["commodity"] == "Maize"
    assert records[0]["price"] == "100"


def test_doubled_quote_csv_drops_the_blank_row_name_and_trailing_semicolons() -> None:
    header = _wrap_csv_line(',"country","price";;;')
    row = _wrap_csv_line('1,"Nigeria","100";;;')
    records = parse_csv((header + "\n" + row + "\n").encode(), ("country", "price"))
    assert records == [{"country": "Nigeria", "price": "100"}]


def test_one_column_csv_with_a_comma_stays_one_column() -> None:
    records = parse_csv(b'note\n"hello, world"\n', ("note",))
    assert records == [{"note": "hello, world"}]


def test_windows_csv_reads_the_micro_sign() -> None:
    records = parse_csv(b"unit\n\xb5g\n", ("unit",))
    assert records == [{"unit": "\u00b5g"}]


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str], body: bytes = b"") -> None:
        self.status_code = status_code
        self.headers = headers
        self._body = body
        self.closed = False

    def iter_content(self, _size: int):
        if self._body:
            yield self._body

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def get(self, *_args, **_kwargs) -> _FakeResponse:
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


def test_waf_challenge_is_not_a_file() -> None:
    session = _FakeSession([
        _FakeResponse(202, {"x-amzn-waf-action": "challenge"}),
    ])
    with pytest.raises(HttpError, match="WAF challenge"):
        request_bytes("https://data.humdata.org/file.csv", session=session, sleep_fn=lambda _seconds: None)
    assert session.calls == 1


def test_http_202_without_a_file_fails_after_retries() -> None:
    session = _FakeSession([
        _FakeResponse(202, {}),
        _FakeResponse(202, {}),
        _FakeResponse(202, {}),
        _FakeResponse(202, {}),
    ])
    with pytest.raises(HttpError, match="HTTP 202"):
        request_bytes(
            "https://data.humdata.org/file.csv",
            session=session,
            max_retries=3,
            sleep_fn=lambda _seconds: None,
        )
    assert session.calls == 4


def test_row_filter_must_name_a_declared_column(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "sources.csv", [_row(row_filter="missing=a")])
    sources, problems, _skipped = load_manifests(tmp_path)
    assert sources == []
    assert problems[0].error == "row_filter columns not in column_names: missing"


def test_row_filter_drops_other_values_without_quarantine(tmp_path: Path) -> None:
    body = (
        "country,market,observed,commodity,price,currency\n"
        "Nigeria,Lagos,2024-05-01,Maize,100,NGN\n"
        "Nigeria,Lagos,2024-05-02,Rice,80,NGN\n"
    ).encode()
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "prices.csv", [_row(row_filter="commodity=Maize")])
    code = run_manifests(
        manifests, tmp_path / "out", get=lambda *_args: body, env={}, countries=_places(), clock=_clock,
    )
    assert code == 0
    row = _loaded_rows(tmp_path / "out")[0]
    assert row["rows_loaded"] == "1"
    assert row["rows_quarantined"] == "0"
