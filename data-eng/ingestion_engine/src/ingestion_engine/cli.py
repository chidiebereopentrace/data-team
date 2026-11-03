"""Command line for a weekly manifest run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from google.cloud import bigquery

from ingestion_engine.bq import append_report, attempted_source_ids, load_jsonl, run_history
from ingestion_engine.mail import send_report
from ingestion_engine.run import run_manifests

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


def _load_tree(client: bigquery.Client, project: str, dataset: str, folder: Path) -> None:
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            first_line = next((line for line in handle if line.strip()), "")
        first = json.loads(first_line) if first_line else None
        if not first:
            continue
        load_jsonl(
            client,
            project=project,
            dataset=dataset,
            table=path.parent.name,
            path=path,
            mode=str(first.get("_load_mode") or "append_snapshot"),
            ingest_date=str(first.get("_ingest_date") or ""),
        )


def _publish(out_dir: Path) -> None:
    dataset = os.environ.get("INGEST_TARGET_DATASET", "landing_dev").strip() or "landing_dev"
    raw_dataset = paired_raw_dataset(dataset)
    report_dataset = os.environ.get("INGEST_REPORT_DATASET", dataset).strip() or dataset
    _target_allowed(dataset)
    _target_allowed(raw_dataset)
    _target_allowed(report_dataset)
    project = os.environ["BQ_PROJECT"]
    client = bigquery.Client(project=project)
    _load_tree(client, project, dataset, out_dir / "landing")
    _load_tree(client, project, raw_dataset, out_dir / "prepared")
    for name in ("ingest_runs", "ingest_coverage", "ingest_failures", "ingest_sources"):
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


def _already_done() -> set[str]:
    project = os.environ.get("BQ_PROJECT", "").strip()
    if not project:
        return set()
    client = bigquery.Client(project=project)
    return attempted_source_ids(client, _report_table("ingest_sources"))


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
    code = run_manifests(manifests, out, done=_already_done())
    publish_error = ""
    uploaded = False
    if args.execute:
        try:
            _publish(out)
            uploaded = True
        except Exception as exc:
            publish_error = str(exc)
            print(f"publish failed\t{exc}", flush=True)
            code = 1
    send_report(out, publish_error=publish_error, uploaded=uploaded, history=_past_runs())
    if uploaded:
        clear_staged(out)
    return code
