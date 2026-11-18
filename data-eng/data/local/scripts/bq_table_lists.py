#!/usr/bin/env python3
"""Live discovery and shared naming for local copies of transform datasets."""
from __future__ import annotations

import hashlib
import os
import re

LAYERS: tuple[str, ...] = (
    "landing",
    "raw_dev",
    "staging_dev",
    "intermediate_dev",
    "mart_dev",
)

# Env vars: prefer new names; fall back to legacy BQ_DATASET_BRONZE/SILVER/GOLD.
DATASET_ENV = {
    "landing": "BQ_DATASET_LANDING",
    "raw_dev": "BQ_DATASET_RAW_DEV",
    "staging_dev": "BQ_DATASET_STAGING_DEV",
    "intermediate_dev": "BQ_DATASET_INTERMEDIATE",
    "mart_dev": "BQ_DATASET_MART_DEV",
}
DATASET_ENV_LEGACY = {
    "raw_dev": "BQ_DATASET_BRONZE",
    "staging_dev": "BQ_DATASET_SILVER",
    "mart_dev": "BQ_DATASET_GOLD",
}
DEFAULT_DATASET = {
    "landing": "landing",
    "raw_dev": "raw_dev",
    "staging_dev": "staging_dev",
    "intermediate_dev": "intermediate_dev",
    "mart_dev": "mart_dev",
}

# Postgres identifiers are capped at 63 bytes. Longer BigQuery names are stored
# under a stable shortened name.
PG_IDENT_MAX = 63


def dataset_id_for_layer(layer: str) -> str:
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer: {layer}. Expected one of {LAYERS}")
    primary = (os.environ.get(DATASET_ENV[layer]) or "").strip()
    if primary:
        return primary
    legacy_key = DATASET_ENV_LEGACY.get(layer)
    if legacy_key:
        legacy = (os.environ.get(legacy_key) or "").strip()
        if legacy:
            return legacy
    return DEFAULT_DATASET[layer]


def local_schema_for_layer(layer: str) -> str:
    """Local Postgres schema matches BigQuery dataset id."""
    return dataset_id_for_layer(layer)


def local_table_name(table_id: str) -> str:
    """Postgres table name. Names over 63 bytes keep a 40-character prefix plus a hash."""
    if len(table_id.encode("utf-8")) <= PG_IDENT_MAX:
        return table_id
    digest = hashlib.sha1(table_id.encode("utf-8")).hexdigest()[:12]
    prefix = re.sub(r"[^0-9A-Za-z_]", "_", table_id)[:40].strip("_") or "t"
    return f"{prefix}_{digest}"[:PG_IDENT_MAX]


def load_layer_tables(layer: str, *, client=None) -> list[str]:
    """Discover current tables and views; explicit per-layer subsets are optional."""
    dataset_id = dataset_id_for_layer(layer)
    override = os.environ.get(f"BQ_{layer.upper()}_TABLES", "").strip()
    if override:
        return sorted({name.strip() for name in override.split(",") if name.strip()})
    project = os.environ.get("BQ_PROJECT", "").strip()
    if not project:
        raise ValueError("Set BQ_PROJECT before discovering warehouse tables")
    if client is None:
        from google.cloud import bigquery
        client = bigquery.Client(project=project)
    timeout = int(os.environ.get("BQ_METADATA_TIMEOUT_SECONDS", "15"))
    return sorted(table.table_id for table in client.list_tables(
        f"{project}.{dataset_id}", timeout=timeout))
