#!/usr/bin/env bash
# Load .env and run schema creation + data sync to populate local Postgres.
# Run from repo root: bash data/local/scripts/populate_local_db.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$REPO_ROOT/data/local/.env"
FAILED_STEPS=()
PYTHON_BIN="${PYTHON_BIN:-python3}"
LAYERS=(landing raw_dev staging_dev intermediate_dev mart_dev)
SCHEMA_ARGS=()
if [ "$#" -gt 0 ]; then
  if [ "$#" -ne 2 ] || [ "$1" != "--dataset" ]; then
    echo "Usage: $0 [--dataset landing|raw_dev|staging_dev|intermediate_dev|mart_dev]" >&2
    exit 2
  fi
  case "$2" in
    landing|raw_dev|staging_dev|intermediate_dev|mart_dev) LAYERS=("$2"); SCHEMA_ARGS=(--dataset "$2") ;;
    *) echo "Unknown dataset layer: $2" >&2; exit 2 ;;
  esac
fi

run_step() {
  local label="$1"
  shift
  echo "=== $label ==="
  if "$@"; then
    echo "OK: $label"
  else
    local status=$?
    echo "ERROR: $label failed with exit code $status; continuing." >&2
    FAILED_STEPS+=("$label (exit $status)")
  fi
}

cd "$REPO_ROOT"
# When running in Docker, compose sets LOCAL_DB_URL to postgres:5432; don't overwrite with .env's localhost
SAVE_LOCAL_DB_URL="${LOCAL_DB_URL:-}"
SAVE_GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-}"
if [ -f "$ENV_FILE" ]; then
  set -a
  if ! source "$ENV_FILE"; then
    echo "ERROR: Could not fully load $ENV_FILE; continuing with available settings." >&2
    FAILED_STEPS+=("load .env")
  fi
  set +a
fi
[ -n "$SAVE_LOCAL_DB_URL" ] && export LOCAL_DB_URL="$SAVE_LOCAL_DB_URL"
[ -n "$SAVE_GOOGLE_APPLICATION_CREDENTIALS" ] && export GOOGLE_APPLICATION_CREDENTIALS="$SAVE_GOOGLE_APPLICATION_CREDENTIALS"

run_step "1. Generate and apply local DDL from BigQuery schemas" \
  "$PYTHON_BIN" data/local/scripts/bq_schema_to_local_pg.py "${SCHEMA_ARGS[@]}"

for layer in "${LAYERS[@]}"; do
  run_step "2. Refresh $layer tables from live BigQuery discovery" \
    "$PYTHON_BIN" data/local/scripts/sync_all_tables.py --dataset "$layer"
done

GIS_SCRIPT="data/ingestion/satellite/gis_data_ingestion.py"
if [ "${#LAYERS[@]}" -eq 5 ] && [ -f "$GIS_SCRIPT" ]; then
  run_step "3. Load GIS CSV into local DB" "$PYTHON_BIN" "$GIS_SCRIPT"
else
  echo "SKIP: optional GIS load (not selected or script is absent)."
fi

echo ""
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
  echo "Populate run completed with ${#FAILED_STEPS[@]} failed step(s):" >&2
  printf ' - %s\n' "${FAILED_STEPS[@]}" >&2
  echo "Successful steps were kept; rerun after fixing the listed failures." >&2
  exit 1
fi

echo "Populate run completed for layers: ${LAYERS[*]}. Local schemas match the configured BigQuery datasets."
