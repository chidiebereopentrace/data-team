"""Coverage ledger gates: the committed artifact matches the contracts and claims nothing unearned."""
from __future__ import annotations

import json

import pytest
import yaml

from ml.rag.chatbot.coverage_ledger import (
    LEDGER_PATH,
    OPERATIONS,
    READINESS_ORDER,
    lineage_parent,
    load_ledger,
    operation_supported,
    serving_tables,
)
from ml.rag.chatbot.mart_indicator_classes import all_class_codes
from ml.rag.helpers.build_coverage_ledger import LIVE_PATH, REGISTRY_PATH, snapshot_hash

REBUILD = "cd ml-eng; python -m ml.rag.helpers.build_coverage_ledger"


@pytest.fixture(scope="module")
def ledger() -> dict:
    assert LEDGER_PATH.is_file(), f"ledger missing; run: {REBUILD}"
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_ledger_is_current_with_its_inputs(ledger: dict) -> None:
    assert ledger["snapshot_hash"] == snapshot_hash(), f"contracts changed; rebuild: {REBUILD}"


def test_every_class_has_a_row_with_valid_readiness(ledger: dict) -> None:
    assert set(ledger["classes"]) == set(all_class_codes())
    for code, row in ledger["classes"].items():
        assert row["readiness"] in READINESS_ORDER, code
        assert row["serving_tables"], f"{code} has no serving table"


def test_every_mart_object_has_a_disposition(ledger: dict) -> None:
    for table, entry in ledger["objects"].items():
        assert entry["disposition"] in ("serving", "dimension", "bridge", "unsupported"), table
        if entry["disposition"] == "unsupported":
            assert entry.get("reason"), f"{table} unsupported without a reason"


def test_every_serving_table_declares_operations() -> None:
    declared = (yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}).get("table_operations") or {}
    missing = [t for t in serving_tables() if t not in declared]
    assert not missing, f"serving tables without table_operations: {missing}"
    for table in serving_tables():
        for op in OPERATIONS:
            supported, reason = operation_supported(table, op)
            assert supported or reason, f"{table}.{op} refused without a reason"


def test_readiness_is_earned(ledger: dict) -> None:
    live = json.loads(LIVE_PATH.read_text(encoding="utf-8")) if LIVE_PATH.is_file() else {}
    for table, entry in ledger["objects"].items():
        state = entry["readiness"]
        if state in ("fixture_verified", "integration_verified"):
            assert entry["probe"]["ok"], f"{table} verified without a passing probe"
        if state == "integration_verified":
            assert (live.get("tables") or {}).get(table, {}).get("ok"), f"{table} claims live verification"
        if state == "blocked":
            assert entry.get("blockers"), f"{table} blocked without blockers"


def test_aggregates_record_their_dbt_parent() -> None:
    load_ledger.cache_clear()
    assert lineage_parent("agg_production_country_year") == "fct_production"
    assert lineage_parent("agg_prices_country_month") == "fct_prices"
    assert lineage_parent("fct_production") is None
