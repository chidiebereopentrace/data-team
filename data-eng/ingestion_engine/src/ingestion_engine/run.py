"""Run every manifest row, keep prepared rows, and write the weekly report."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from ingestion_engine.countries import Countries, load_africa, load_countries
from ingestion_engine.fetch import AuthError, FetchError, default_get, iter_pages
from ingestion_engine.http import HttpError
from ingestion_engine.manifest import ManifestProblem, Source, load_manifests
from ingestion_engine.shape import prepare_row, subject_iso3

Get = Callable[[str, dict[str, str], dict[str, str]], bytes]
Clock = Callable[[], datetime]
RUN_ID_TIME = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True)
class Cursor:
    """What earlier uploaded runs recorded in ingest_sources.

    `loaded` holds the start of the latest run that loaded or partly loaded each source.
    `stopped` holds sources whose latest attempt was left STARTED by a run that died.
    """

    loaded: dict[str, datetime] = field(default_factory=dict)
    stopped: frozenset[str] = frozenset()


def run_time(run_id: str) -> datetime | None:
    """Return the UTC start encoded at the front of a run id."""
    try:
        return datetime.strptime(run_id[:16], RUN_ID_TIME).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def read_cursor(attempts: Iterable[tuple[str, str, str]]) -> Cursor:
    """Build the cursor from (source_id, status, run_id) rows."""
    loaded: dict[str, datetime] = {}
    latest: dict[str, tuple[str, bool]] = {}
    for source_id, status, run_id in attempts:
        if not source_id or not run_id:
            continue
        finished = status != "STARTED"
        if status in {"LOADED", "PARTIAL"}:
            when = run_time(run_id)
            if when is not None and (source_id not in loaded or when > loaded[source_id]):
                loaded[source_id] = when
        prior = latest.get(source_id)
        if prior is None or run_id > prior[0] or (run_id == prior[0] and finished):
            latest[source_id] = (run_id, finished)
    stopped = frozenset(source_id for source_id, (_run, finished) in latest.items() if not finished)
    return Cursor(loaded=loaded, stopped=stopped)


def is_due(source: Source, cursor: Cursor, today: date) -> bool:
    """Fetch a source that never loaded, whose last attempt failed, or whose refresh has come round."""
    last = cursor.loaded.get(source.source_id)
    if last is None:
        return True
    if source.refresh_days is None:
        return False
    return (today - last.date()).days >= source.refresh_days


@dataclass
class SourceResult:
    manifest_file: str
    source_id: str
    source_name: str
    source_url: str
    endpoint: str
    table_name: str
    indicator_class: str
    contract_type: str
    auth_required: bool
    auth_type: str
    owner: str
    status: str
    failure_stage: str
    error: str
    rows_loaded: int = 0
    rows_quarantined: int = 0
    rows_outside_africa: int = 0
    quarantine_reasons: dict[str, int] = field(default_factory=dict)
    load_mode: str = ""
    coverage: dict[tuple[str, str, str, int], int] = field(default_factory=dict)
    last_loaded_at: str = ""


Begin = Callable[[str, Source], None]
Finish = Callable[[str, SourceResult], None]


def _from_source(source: Source, status: str, failure_stage: str, error: str) -> SourceResult:
    return SourceResult(
        manifest_file=source.manifest_file,
        source_id=source.source_id,
        source_name=source.source_name,
        source_url=source.source_url,
        endpoint=source.endpoint,
        table_name=source.table_name,
        indicator_class=source.indicator_class,
        contract_type=source.contract_type,
        auth_required=source.auth_required,
        auth_type=source.auth_type,
        owner=source.owner,
        status=status,
        failure_stage=failure_stage,
        error=error,
        load_mode=source.load_mode,
    )


def _halted(source: Source) -> SourceResult:
    """A source left STARTED by a dead run. It is skipped once, then retried on the next run."""
    return _from_source(source, "FAILED", "fetch", "stopped before finish")


def _from_problem(problem: ManifestProblem, status: str) -> SourceResult:
    return SourceResult(
        manifest_file=problem.manifest_file,
        source_id=problem.source_id,
        source_name=problem.source_name,
        source_url=problem.source_url,
        endpoint=problem.endpoint,
        table_name=problem.table_name,
        indicator_class=problem.indicator_class,
        contract_type=problem.contract_type,
        auth_required=problem.auth_required.lower() == "true",
        auth_type=problem.auth_type,
        owner=problem.owner,
        status=status,
        failure_stage="" if status == "SKIPPED" else problem.stage,
        error="" if status == "SKIPPED" else problem.error,
    )


def _hash(row: dict[str, object], keys: tuple[str, ...]) -> str:
    parts = [f"{key}={row.get(key, '')}" for key in keys]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _lineage(
    row: dict[str, object],
    source: Source,
    *,
    run_id: str,
    ingested_at: str,
    today: date,
    manifest_sha: str,
    response_sha: str,
) -> dict[str, object]:
    row.update(
        _source=source.source_id,
        _run_id=run_id,
        _ingested_at=ingested_at,
        _ingest_date=today.isoformat(),
        _source_url=source.endpoint,
        _record_hash=_hash(row, source.natural_key),
        _manifest_sha256=manifest_sha,
        _response_sha256=response_sha,
        _manifest_file=source.manifest_file,
        _load_mode=source.load_mode,
    )
    return row


def _process(
    source: Source,
    *,
    get: Get,
    env: dict[str, str],
    countries: Countries,
    africa: frozenset[str],
    today: date,
    run_id: str,
    ingested_at: str,
    manifest_sha: str,
    out_dir: Path,
) -> SourceResult:
    result = _from_source(source, "FAILED", "fetch", "")
    print(f"fetching\t{source.source_id}", flush=True)
    seen: set[tuple[str, ...]] = set()
    raw_digest = hashlib.sha256()
    landing_handle = prepared_handle = quarantine_handle = raw_handle = None
    landing_path = out_dir / "landing" / source.table_name / f"{source.source_id}.jsonl"
    prepared_path = out_dir / "prepared" / source.table_name / f"{source.source_id}.jsonl"
    quarantine_path = out_dir / "quarantine" / source.source_id / f"{run_id}.jsonl"
    raw_path = out_dir / "raw" / source.source_id / f"{run_id}.bin"
    try:
        for records, body in iter_pages(source, get=get, env=env):
            if raw_handle is None:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_handle = raw_path.open("wb")
            elif raw_path.stat().st_size:
                raw_handle.write(b"\n")
                raw_digest.update(b"\n")
            raw_handle.write(body)
            raw_digest.update(body)
            response_sha = raw_digest.hexdigest()
            for record in records:
                identity = tuple(record.get(key, "") for key in source.natural_key)
                if identity in seen:
                    continue
                seen.add(identity)
                iso3 = subject_iso3(source, record, countries)
                if iso3 is not None and iso3 not in africa:
                    result.rows_outside_africa += 1
                    continue
                landing_row = _lineage(
                    {column: record.get(column, "") for column in source.column_names},
                    source,
                    run_id=run_id,
                    ingested_at=ingested_at,
                    today=today,
                    manifest_sha=manifest_sha,
                    response_sha=response_sha,
                )
                landing_handle = _jsonl_handle(landing_handle, landing_path)
                landing_handle.write(json.dumps(landing_row, ensure_ascii=False) + "\n")
                prepared, reason = prepare_row(source, record, countries=countries, today=today)
                if prepared is None:
                    reason = reason or ""
                    if not result.error:
                        result.error = reason
                    result.rows_quarantined += 1
                    result.quarantine_reasons[reason] = result.quarantine_reasons.get(reason, 0) + 1
                    quarantine_row = {column: record.get(column, "") for column in source.column_names}
                    quarantine_row["_quarantine_reason"] = reason
                    quarantine_handle = _jsonl_handle(quarantine_handle, quarantine_path)
                    quarantine_handle.write(json.dumps(quarantine_row, ensure_ascii=False) + "\n")
                    continue
                prepared_row = _lineage(
                    prepared,
                    source,
                    run_id=run_id,
                    ingested_at=ingested_at,
                    today=today,
                    manifest_sha=manifest_sha,
                    response_sha=response_sha,
                )
                prepared_handle = _jsonl_handle(prepared_handle, prepared_path)
                prepared_handle.write(json.dumps(prepared_row, ensure_ascii=False) + "\n")
                result.rows_loaded += 1
                bucket = (
                    str(prepared_row["indicator_class"]),
                    str(prepared_row["country_iso3"]),
                    str(prepared_row.get("country_name", "")),
                    int(prepared_row["year"]),
                )
                result.coverage[bucket] = result.coverage.get(bucket, 0) + 1
    except AuthError as exc:
        result.failure_stage = "auth"
        result.error = str(exc)
        return result
    except FetchError as exc:
        result.failure_stage = exc.stage
        result.error = str(exc)
        return result
    except HttpError as exc:
        result.failure_stage = "fetch"
        result.error = str(exc)
        return result
    except Exception as exc:
        result.failure_stage = "fetch"
        result.error = str(exc)
        return result
    finally:
        for handle in (landing_handle, prepared_handle, quarantine_handle, raw_handle):
            if handle is not None:
                handle.close()
        if raw_handle is not None:
            raw_path.with_suffix(".sha256").write_text(raw_digest.hexdigest() + "\n", encoding="utf-8")
    if result.rows_loaded == 0 and result.rows_quarantined == 0 and result.rows_outside_africa == 0 and result.error == "":
        result.failure_stage = "shape"
        result.error = "no records"
        return result
    if result.rows_loaded and result.rows_quarantined:
        result.status = "PARTIAL"
        result.failure_stage = "shape"
    elif result.rows_loaded:
        result.status = "LOADED"
        result.failure_stage = ""
        result.error = ""
    else:
        result.status = "FAILED"
        result.failure_stage = "shape"
        if not result.error:
            result.error = "no prepared rows"
    return result


def _jsonl_handle(handle, path: Path):
    if handle is not None:
        return handle
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _manifest_sha(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.csv")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _reason_text(reasons: dict[str, int]) -> str:
    return "; ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))


COVERAGE_FIELDS = [
    "run_id", "indicator_class", "country_iso3", "country_name", "year", "rows_loaded", "source_ids",
]
FAILURE_FIELDS = [
    "run_id", "manifest_file", "source_id", "source_name", "source_url", "endpoint", "table_name",
    "indicator_class", "contract_type", "auth_required", "auth_type", "owner", "status", "failure_stage",
    "error", "rows_loaded", "rows_quarantined", "rows_outside_africa", "quarantine_reasons",
]
SOURCE_FIELDS = [
    "run_id", "manifest_file", "source_id", "source_name", "table_name", "indicator_class", "contract_type",
    "owner", "status", "failure_stage", "rows_loaded", "rows_quarantined", "rows_outside_africa",
    "quarantine_reasons", "error", "last_loaded_at",
]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or [])
    if not names:
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    names.append(key)
    if not names:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _coverage(results: list[SourceResult], run_id: str) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, int], dict[str, object]] = {}
    for result in results:
        for (indicator, iso3, name, year), count in result.coverage.items():
            key = (indicator, iso3, year)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = {
                    "run_id": run_id,
                    "indicator_class": indicator,
                    "country_iso3": iso3,
                    "country_name": name,
                    "year": year,
                    "rows_loaded": 0,
                    "source_ids": set(),
                }
                buckets[key] = bucket
            bucket["rows_loaded"] = int(bucket["rows_loaded"]) + count
            sources = bucket["source_ids"]
            assert isinstance(sources, set)
            sources.add(result.source_id)
    coverage = []
    for bucket in buckets.values():
        sources = bucket["source_ids"]
        assert isinstance(sources, set)
        coverage.append({**bucket, "source_ids": ",".join(sorted(sources))})
    return coverage


def _failures(results: list[SourceResult], run_id: str) -> list[dict[str, object]]:
    rows = []
    for result in results:
        if result.status not in {"FAILED", "PARTIAL", "SKIPPED"}:
            continue
        rows.append({
            "run_id": run_id,
            "manifest_file": result.manifest_file,
            "source_id": result.source_id,
            "source_name": result.source_name,
            "source_url": result.source_url,
            "endpoint": result.endpoint,
            "table_name": result.table_name,
            "indicator_class": result.indicator_class,
            "contract_type": result.contract_type,
            "auth_required": result.auth_required,
            "auth_type": result.auth_type,
            "owner": result.owner,
            "status": result.status,
            "failure_stage": result.failure_stage,
            "error": result.error,
            "rows_loaded": result.rows_loaded,
            "rows_quarantined": result.rows_quarantined,
            "rows_outside_africa": result.rows_outside_africa,
            "quarantine_reasons": _reason_text(result.quarantine_reasons),
        })
    return rows


def source_row(result: SourceResult, run_id: str) -> dict[str, object]:
    """One ingest_sources row for a finished source."""
    return {
        "run_id": run_id,
        "manifest_file": result.manifest_file,
        "source_id": result.source_id,
        "source_name": result.source_name,
        "table_name": result.table_name,
        "indicator_class": result.indicator_class,
        "contract_type": result.contract_type,
        "owner": result.owner,
        "status": result.status,
        "failure_stage": result.failure_stage,
        "rows_loaded": result.rows_loaded,
        "rows_quarantined": result.rows_quarantined,
        "rows_outside_africa": result.rows_outside_africa,
        "quarantine_reasons": _reason_text(result.quarantine_reasons),
        "error": result.error,
    }


def started_row(source: Source, run_id: str) -> dict[str, object]:
    """One ingest_sources row written before a source is fetched."""
    return {
        "run_id": run_id,
        "manifest_file": source.manifest_file,
        "source_id": source.source_id,
        "source_name": source.source_name,
        "table_name": source.table_name,
        "indicator_class": source.indicator_class,
        "contract_type": source.contract_type,
        "owner": source.owner,
        "status": "STARTED",
        "failure_stage": "",
        "rows_loaded": 0,
        "rows_quarantined": 0,
        "rows_outside_africa": 0,
        "quarantine_reasons": "",
        "error": "",
    }


def _sources(results: list[SourceResult], run_id: str) -> list[dict[str, object]]:
    """The local report rows. They add when each source last loaded."""
    return [source_row(result, run_id) | {"last_loaded_at": result.last_loaded_at} for result in results]


def _note(result: SourceResult) -> None:
    detail = result.error
    if result.status == "CURRENT":
        detail = f"last loaded {result.last_loaded_at[:10]}"
    print(
        f"{result.source_id or result.manifest_file}\t{result.status}\tloaded={result.rows_loaded}\t"
        f"quarantined={result.rows_quarantined}\t{detail}",
        flush=True,
    )


def _close(run_id: str, result: SourceResult, on_done: Finish | None) -> None:
    if on_done is None:
        return
    try:
        on_done(run_id, result)
    except Exception as exc:
        result.status = "FAILED"
        result.failure_stage = result.failure_stage or "publish"
        result.error = str(exc)


def run_manifests(
    directory: Path,
    out_dir: Path,
    *,
    get: Get | None = None,
    env: dict[str, str] | None = None,
    countries: Countries | None = None,
    clock: Clock | None = None,
    africa: frozenset[str] | None = None,
    cursor: Cursor | None = None,
    on_begin: Begin | None = None,
    on_done: Finish | None = None,
) -> int:
    """Fetch and prepare every enabled manifest row that is due. Exit 1 when any source failed."""
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    today = now.date()
    run_id = now.strftime(RUN_ID_TIME) + "-" + uuid.uuid4().hex[:8]
    fetcher = get or default_get
    secrets = env if env is not None else dict(os.environ)
    place = countries or load_countries()
    african = africa if africa is not None else load_africa()
    state = cursor or Cursor()
    sources, problems, skipped = load_manifests(directory)

    def last_loaded(source_id: str) -> str:
        when = state.loaded.get(source_id)
        return when.isoformat() if when else ""

    halted = [source for source in sources if source.source_id in state.stopped]
    current = [
        source for source in sources
        if source.source_id not in state.stopped and not is_due(source, state, today)
    ]
    due = [
        source for source in sources
        if source.source_id not in state.stopped and is_due(source, state, today)
    ]
    results = [_from_problem(item, "FAILED") for item in problems]
    results.extend(_from_problem(item, "SKIPPED") for item in skipped)
    for source in current:
        result = _from_source(source, "CURRENT", "", "")
        result.last_loaded_at = last_loaded(source.source_id)
        results.append(result)
        _note(result)
    for source in halted:
        result = _halted(source)
        result.last_loaded_at = last_loaded(source.source_id)
        _close(run_id, result, on_done)
        results.append(result)
        _note(result)
    manifest_sha = _manifest_sha(directory) if directory.is_dir() else ""
    for source in due:
        if on_begin is not None:
            try:
                on_begin(run_id, source)
            except Exception as exc:
                result = _halted(source)
                result.error = str(exc)
                result.last_loaded_at = last_loaded(source.source_id)
                results.append(result)
                _note(result)
                continue
        result = _process(
            source,
            get=fetcher,
            env=secrets,
            countries=place,
            africa=african,
            today=today,
            run_id=run_id,
            ingested_at=now.isoformat(),
            manifest_sha=manifest_sha,
            out_dir=out_dir,
        )
        _close(run_id, result, on_done)
        if result.status in {"LOADED", "PARTIAL"}:
            result.last_loaded_at = now.isoformat()
        else:
            result.last_loaded_at = last_loaded(source.source_id)
        results.append(result)
        _note(result)

    coverage = _coverage(results, run_id)
    failures = _failures(results, run_id)
    source_rows = _sources(results, run_id)
    loaded = sum(item.rows_loaded for item in results)
    quarantined = sum(item.rows_quarantined for item in results)
    summary = {
        "run_id": run_id,
        "started_at": now.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "manifest_files": len(list(directory.glob("*.csv"))) if directory.is_dir() else 0,
        "sources_enabled": len(sources) + len(problems),
        "sources_current": len(current),
        "sources_fetched": len(due),
        "sources_ingested": sum(item.rows_loaded > 0 for item in results),
        "sources_loaded": sum(item.status == "LOADED" for item in results),
        "sources_partial": sum(item.status == "PARTIAL" for item in results),
        "sources_failed": sum(item.status == "FAILED" for item in results),
        "sources_skipped": sum(item.status == "SKIPPED" for item in results),
        "rows_loaded": loaded,
        "rows_quarantined": quarantined,
        "rows_outside_africa": sum(item.rows_outside_africa for item in results),
        "status": "FAILED" if any(item.status == "FAILED" for item in results) else "FINISHED",
    }
    reports = out_dir / "reports"
    _write_csv(reports / "ingest_runs.csv", [summary])
    _write_csv(reports / "ingest_coverage.csv", coverage, COVERAGE_FIELDS)
    _write_csv(reports / "ingest_failures.csv", failures, FAILURE_FIELDS)
    _write_csv(reports / "ingest_sources.csv", source_rows, SOURCE_FIELDS)
    (reports / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for result in results:
        if result.status == "SKIPPED" or result.failure_stage == "manifest":
            _note(result)
    print(str(reports), flush=True)
    return 0 if summary["status"] == "FINISHED" else 1

