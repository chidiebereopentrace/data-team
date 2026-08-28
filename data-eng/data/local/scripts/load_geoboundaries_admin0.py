#!/usr/bin/env python3
"""
Download geoBoundaries ADM0 polygons for Africa-scoped countries and load to BigQuery.

Usage (from data-eng root):
  python data/local/scripts/load_geoboundaries_admin0.py

Requires: GOOGLE_APPLICATION_CREDENTIALS or gcloud auth; BQ_PROJECT in data/local/.env.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

SEED_PATH = REPO_ROOT / "dbt" / "seeds" / "ref_m49_country.csv"
TERRITORY_ISO2 = {"YT", "RE", "SH", "TF", "EH"}
API_URL = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ADM0/"


def _load_dotenv() -> None:
    env_file = REPO_ROOT / "data" / "local" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().replace("export ", "", 1).strip()
        value = value.strip().strip("'\"")
        if key and not (os.environ.get(key) or "").strip():
            os.environ[key] = value


def _target_iso3() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with SEED_PATH.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            iso2 = (row.get("country_iso2") or "").strip().upper()
            iso3 = (row.get("country_iso3") or "").strip().upper()
            name = (row.get("country_name") or "").strip()
            in_africa = (row.get("in_africa_scope") or "").strip().lower() == "true"
            if not iso3 or not iso2:
                continue
            if in_africa or iso2 in TERRITORY_ISO2:
                rows.append((iso2, iso3, name))
    return rows


def _fetch_geojson(iso3: str) -> dict | None:
    url = API_URL.format(iso3=iso3)
    req = urllib.request.Request(url, headers={"User-Agent": "openTrace-data-team/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"  skip {iso3}: HTTP {exc.code}", file=sys.stderr, flush=True)
        return None
    except urllib.error.URLError as exc:
        print(f"  skip {iso3}: {exc.reason}", file=sys.stderr, flush=True)
        return None

    if not payload:
        return None

    meta = payload[0] if isinstance(payload, list) else payload
    download_url = meta.get("gjDownloadURL")
    if not download_url:
        return None

    with urllib.request.urlopen(download_url, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _geography_geojson(geojson: dict) -> str:
    if geojson.get("type") == "FeatureCollection":
        geoms = [
            feature["geometry"]
            for feature in geojson.get("features", [])
            if feature.get("geometry")
        ]
        if not geoms:
            raise ValueError("feature collection has no geometries")
        if len(geoms) == 1:
            return json.dumps(geoms[0], ensure_ascii=False)
        return json.dumps({"type": "GeometryCollection", "geometries": geoms}, ensure_ascii=False)
    if geojson.get("type") == "Feature":
        return json.dumps(geojson.get("geometry"), ensure_ascii=False)
    return json.dumps(geojson, ensure_ascii=False)


def _bbox_from_geojson(geojson: dict) -> tuple[float, float, float, float]:
    coords: list[list[float]] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("type") == "FeatureCollection":
                for feature in node.get("features", []):
                    walk(feature)
                return
            if node.get("type") == "Feature":
                walk(node.get("geometry"))
                return
            if node.get("type") in {"Polygon", "MultiPolygon", "LineString", "MultiLineString"}:
                walk(node.get("coordinates"))
                return
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[0], (int, float)) and isinstance(node[1], (int, float)):
                coords.append([float(node[0]), float(node[1])])
            else:
                for child in node:
                    walk(child)

    walk(geojson)
    if not coords:
        raise ValueError("no coordinates in geojson")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lats), max(lats), min(lons), max(lons)


def _load_to_bq(records: list[dict]) -> None:
    from google.cloud import bigquery

    project = os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4")
    dataset = os.environ.get("BQ_DATASET_BRONZE", "raw_dev")
    staging_id = f"{project}.{dataset}._geoboundaries_admin0_staging"
    table_id = f"{project}.{dataset}.geoboundaries_admin0_africa"

    client = bigquery.Client(project=project)
    client.query(f"drop table if exists `{table_id}`").result()
    client.query(
        f"""
        create table if not exists `{table_id}` (
          country_iso2 string,
          country_iso3 string,
          country_name string,
          geog geography,
          min_lat float64,
          max_lat float64,
          min_lng float64,
          max_lng float64,
          source string,
          loaded_at timestamp
        )
        """
    ).result()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ndjson", encoding="utf-8", delete=False, newline="\n"
    ) as tmp:
        for row in records:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        ndjson_path = tmp.name

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField("country_iso2", "STRING"),
            bigquery.SchemaField("country_iso3", "STRING"),
            bigquery.SchemaField("country_name", "STRING"),
            bigquery.SchemaField("geojson", "STRING"),
            bigquery.SchemaField("min_lat", "FLOAT64"),
            bigquery.SchemaField("max_lat", "FLOAT64"),
            bigquery.SchemaField("min_lng", "FLOAT64"),
            bigquery.SchemaField("max_lng", "FLOAT64"),
            bigquery.SchemaField("loaded_at", "TIMESTAMP"),
        ],
    )
    with open(ndjson_path, "rb") as fh:
        client.load_table_from_file(fh, staging_id, job_config=job_config).result()

    Path(ndjson_path).unlink(missing_ok=True)

    client.query(f"truncate table `{table_id}`").result()
    client.query(
        f"""
        insert into `{table_id}` (
          country_iso2, country_iso3, country_name, geog, min_lat, max_lat, min_lng, max_lng, source, loaded_at
        )
        select
          country_iso2,
          country_iso3,
          country_name,
          safe.st_geogfromgeojson(geojson),
          min_lat,
          max_lat,
          min_lng,
          max_lng,
          'geoboundaries',
          loaded_at
        from `{staging_id}`
        where safe.st_geogfromgeojson(geojson) is not null
        """
    ).result()
    client.delete_table(staging_id, not_found_ok=True)
    print(f"Loaded {len(records)} rows to {table_id}", flush=True)


def main() -> int:
    _load_dotenv()
    loaded_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []

    for iso2, iso3, name in _target_iso3():
        print(f"Fetching {iso3} ({name})...", flush=True)
        geojson = _fetch_geojson(iso3)
        if geojson is None:
            continue
        min_lat, max_lat, min_lng, max_lng = _bbox_from_geojson(geojson)
        records.append(
            {
                "country_iso2": iso2,
                "country_iso3": iso3,
                "country_name": name,
                "geojson": _geography_geojson(geojson),
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lng": min_lng,
                "max_lng": max_lng,
                "loaded_at": loaded_at,
            }
        )

    if not records:
        print("No boundaries fetched.", file=sys.stderr)
        return 1

    _load_to_bq(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
