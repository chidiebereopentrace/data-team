"""Command line for a weekly manifest run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from google.cloud import bigquery

from ingestion_engine.bq import append_report, load_jsonl, run_history, source_cursor
from ingestion_engine.mail import send_report
from ingestion_engine.manifest import Source
from ingestion_engine.run import Begin, Finish, SourceResult, run_manifests, source_row, started_row

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEV_DATASETS = ("landing_dev", "raw_dev")


def load_env_file(path: Path) -> None:
    """Fill unset process variables from a local env file. Existing values win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def materialize_service_account() -> Path | None:
    """Write BQ_SERVICE_ACCOUNT_JSON to a private file when no key path is set."""
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
        return None
    raw = os.environ.get("BQ_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        return None
    handle, name = tempfile.mkstemp(prefix="ingestion-engine-", suffix=".json")
    os.close(handle)
    path = Path(name)
    path.write_text(raw, encoding="utf-8")
    os.chmod(path, 0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    return path


def paired_raw_dataset(landing: str) -> str:
    if landing == "landing_dev":
        return "raw_dev"
    if landing.startswith("landing_dev_"):
        return "raw_dev_" + landing[len("landing_dev_"):]
    if landing == "landing":
        return "raw"
    raise SystemExit(f"No raw dataset paired with {landing}.")


def _target_allowed(dataset: str) -> None:
    if dataset in DEV_DATASETS or dataset.startswith("landing_dev_") or dataset.startswith("raw_dev_"):
        return
    allow = os.environ.get("INGEST_ALLOW_PROD", "").strip().lower() in {"1", "true", "yes"}
    if not allow:
        raise SystemExit(f"Refusing dataset {dataset}. Use landing_dev, or set INGEST_ALLOW_PROD=1.")


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _datasets() -> tuple[str, str, str, str]:
    """Return the project, landing dataset, raw dataset, and report dataset."""
    dataset = os.environ.get("INGEST_TARGET_DATASET", "landing_dev").strip() or "landing_dev"
    raw_dataset = paired_raw_dataset(dataset)
    report_dataset = os.environ.get("INGEST_REPORT_DATASET", dataset).strip() or dataset
    _target_allowed(dataset)
    _target_allowed(raw_dataset)
    _target_allowed(report_dataset)
    project = os.environ.get("BQ_PROJECT", "").strip()
    if not project:
        raise SystemExit("BQ_PROJECT is required with --execute.")
    return project, dataset, raw_dataset, report_dataset


def _load_file(client: bigquery.Client, project: str, dataset: str, path: Path) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        first_line = next((line for line in handle if line.strip()), "")
    if not first_line:
        return
    first = json.loads(first_line)
    load_jsonl(
        client,
        project=project,
        dataset=dataset,
        table=path.parent.name,
        path=path,
        mode=str(first.get("_load_mode") or "append_snapshot"),
        ingest_date=str(first.get("_ingest_date") or ""),
    )


def _load_source(
    client: bigquery.Client,
    project: str,
    dataset: str,
    raw_dataset: str,
    out_dir: Path,
    result: SourceResult,
) -> None:
    landing = out_dir / "landing" / result.table_name / f"{result.source_id}.jsonl"
    prepared = out_dir / "prepared" / result.table_name / f"{result.source_id}.jsonl"
    _load_file(client, project, dataset, landing)
    _load_file(client, project, raw_dataset, prepared)


def _append_row(client: bigquery.Client, table_id: str, row: dict[str, object]) -> None:
    handle, name = tempfile.mkstemp(prefix="ingestion-engine-row-", suffix=".csv")
    os.close(handle)
    path = Path(name)
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        append_report(client, table_id, path)
    finally:
        path.unlink(missing_ok=True)


def _delete_source(out_dir: Path, result: SourceResult) -> None:
    for path in (
        out_dir / "landing" / result.table_name / f"{result.source_id}.jsonl",
        out_dir / "prepared" / result.table_name / f"{result.source_id}.jsonl",
    ):
        path.unlink(missing_ok=True)
    for folder in (
        out_dir / "quarantine" / result.source_id,
        out_dir / "raw" / result.source_id,
    ):
        if folder.is_dir():
            shutil.rmtree(folder)


def _bind_cursor(
    client: bigquery.Client,
    project: str,
    dataset: str,
    raw_dataset: str,
    table_id: str,
    out_dir: Path,
) -> tuple[Begin, Finish]:
    """Record STARTED before a fetch, then the finished row and its tables."""

    def on_begin(run_id: str, source: Source) -> None:
        _append_row(client, table_id, started_row(source, run_id))

    def on_done(run_id: str, result: SourceResult) -> None:
        if result.status in {"LOADED", "PARTIAL"}:
            try:
                _load_source(client, project, dataset, raw_dataset, out_dir, result)
            except Exception as exc:
                result.status = "FAILED"
                result.failure_stage = "publish"
                result.error = str(exc)
        _append_row(client, table_id, source_row(result, run_id))
        _delete_source(out_dir, result)

    return on_begin, on_done


def _publish_reports(client: bigquery.Client, project: str, report_dataset: str, out_dir: Path) -> None:
    for name in ("ingest_runs", "ingest_coverage", "ingest_failures"):
        append_report(client, f"{project}.{report_dataset}.{name}", out_dir / "reports" / f"{name}.csv")


def clear_staged(out_dir: Path) -> None:
    """Remove the files BigQuery has already loaded. Keep the reports."""
    for name in ("landing", "prepared", "quarantine", "raw"):
        folder = out_dir / name
        if folder.is_dir():
            shutil.rmtree(folder)


def _report_table(name: str) -> str:
    project = os.environ.get("BQ_PROJECT", "").strip()
    dataset = os.environ.get("INGEST_TARGET_DATASET", "landing_dev").strip() or "landing_dev"
    report = os.environ.get("INGEST_REPORT_DATASET", dataset).strip() or dataset
    return f"{project}.{report}.{name}"


def _past_runs() -> list[dict[str, object]]:
    project = os.environ.get("BQ_PROJECT", "").strip()
    if not project:
        return []
    try:
        client = bigquery.Client(project=project)
        return run_history(client, _report_table("ingest_runs"))
    except Exception as exc:
        print(f"report history failed\t{exc}", flush=True)
        return []


def main(argv: list[str] | None = None) -> int:
    load_env_file(PACKAGE_ROOT / ".env")
    materialize_service_account()
    parser = argparse.ArgumentParser(description="Load manifest CSVs into landing and raw, and write the weekly report.")
    parser.add_argument("--manifests", type=Path, default=None, help="Directory of manifest CSV files")
    parser.add_argument("--out", type=Path, default=None, help="Local directory for raw files, landing JSONL, and reports")
    parser.add_argument("--execute", action="store_true", help="Also load landing_dev, raw_dev, and the report tables in BigQuery")
    args = parser.parse_args(argv)
    manifests = args.manifests or _resolve(os.environ.get("INGEST_MANIFESTS", "manifests"), PACKAGE_ROOT)
    out = args.out or _resolve(os.environ.get("INGEST_OUT", "out"), PACKAGE_ROOT)
    done: set[str] = set()
    stopped: set[str] = set()
    on_begin: Begin | None = None
    on_done: Finish | None = None
    client: bigquery.Client | None = None
    report_dataset = ""
    project = ""
    if args.execute:
        project, dataset, raw_dataset, report_dataset = _datasets()
        client = bigquery.Client(project=project)
        done, stopped = source_cursor(client, _report_table("ingest_sources"))
        on_begin, on_done = _bind_cursor(client, project, dataset, raw_dataset, _report_table("ingest_sources"), out)
    code = run_manifests(
        manifests,
        out,
        done=done,
        stopped=stopped,
        on_begin=on_begin,
        on_done=on_done,
    )
    publish_error = ""
    uploaded = False
    if args.execute and client is not None:
        try:
            _publish_reports(client, project, report_dataset, out)
            uploaded = True
        except Exception as exc:
            publish_error = str(exc)
            print(f"publish failed\t{exc}", flush=True)
            code = 1
    send_report(out, publish_error=publish_error, uploaded=uploaded, history=_past_runs())
    if uploaded:
        clear_staged(out)
    return code
