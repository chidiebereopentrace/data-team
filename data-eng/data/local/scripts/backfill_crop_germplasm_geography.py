#!/usr/bin/env python3
"""
Backfill raw_dev.crop_germplasm_africa.geography from arcgis_layer_rice_germplasm WKT.

Same objectid on both tables; ArcGIS branch holds geometry while crop table is null.

Usage (from repo root):
  python data-eng/data/local/scripts/backfill_crop_germplasm_geography.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _load_dotenv() -> None:
    env_file = REPO_ROOT / "data-eng" / "data" / "local" / ".env"
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


def main() -> int:
    _load_dotenv()
    from google.cloud import bigquery

    project = os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4")
    dataset = os.environ.get("BQ_DATASET_BRONZE", "raw_dev")
    crop = f"`{project}.{dataset}.crop_germplasm_africa`"
    arcgis = f"`{project}.{dataset}.arcgis_layer_rice_germplasm_in_africa_3d2a9`"

    client = bigquery.Client(project=project)
    sql = f"""
    merge {crop} c
    using {arcgis} a
      on c.objectid = a.objectid
    when matched and c.geography is null and a.geometry_wkt is not null then
      update set geography = safe.st_geogfromtext(a.geometry_wkt)
    """
    job = client.query(sql)
    job.result()
    stats = job.num_dml_affected_rows
    print(f"Updated {stats} crop_germplasm_africa rows from ArcGIS WKT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
