"""Metadata, dry-run, replace_partition, merge tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from google.cloud import bigquery

from ingestion.common.config import get_settings
from ingestion.common.load import (
    FakeBQClient,
    add_metadata,
    load_dataframe,
    record_hash,
)
from ingestion.runner import build_context, run_source


def test_metadata_and_stable_hash() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "market_id": 1,
                "commodity_id": 2,
                "pricetype": "Retail",
                "unit": "KG",
                "price": 1.0,
            }
        ]
    )
    key = ["date", "market_id", "commodity_id", "pricetype", "unit"]
    a = add_metadata(df, source="hdx", run_id="r1", source_url="u", key_cols=key)
    b = add_metadata(df, source="hdx", run_id="r2", source_url="u", key_cols=key)
    assert set(
        ["_source", "_run_id", "_ingested_at", "_ingest_date", "_source_url", "_record_hash"]
    ).issubset(a.columns)
    assert a["_record_hash"].iloc[0] == b["_record_hash"].iloc[0]
    assert a["_record_hash"].iloc[0] == record_hash(df.iloc[0], key)


def test_dry_run_zero_bq_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_OFFLINE_FIXTURES", "1")
    client = FakeBQClient()
    ctx = build_context("hdx_wfp_prices", dry_run=True, limit=10, client=client)
    # Force out dir under tmp if needed
    result = run_source(ctx)
    assert result.status == "ok"
    assert result.out_path is not None
    assert result.out_path.exists()
    assert client.load_calls == []
    assert client.query_calls == []
    assert client.insert_rows_json_calls == []


def test_replace_partition_uses_truncate_decorator() -> None:
    settings = get_settings("landing_dev", load_dotenv_file=False)
    client = FakeBQClient()
    df = pd.DataFrame(
        {
            "x": [1, 2],
            "_source": ["s", "s"],
            "_run_id": ["r", "r"],
            "_ingested_at": [pd.Timestamp.now("UTC"), pd.Timestamp.now("UTC")],
            "_ingest_date": [date(2026, 10, 17), date(2026, 10, 17)],
            "_source_url": ["u", "u"],
            "_record_hash": ["a", "b"],
        }
    )
    result = load_dataframe(
        client, settings, df, table="hdx_wfp_food_prices", mode="replace_partition", run_id="r"
    )
    assert "$20261017" in result.destination
    assert client.load_calls[0]["write_disposition"] == bigquery.WriteDisposition.WRITE_TRUNCATE


def test_merge_issues_merge_and_drops_staging() -> None:
    settings = get_settings("landing_dev", load_dotenv_file=False)
    client = FakeBQClient()
    df = pd.DataFrame(
        {
            "country": ["Nigeria"],
            "market": ["Lagos"],
            "date": ["2024-01-01"],
            "commodity": ["Rice"],
            "_source": ["wb_rtfp"],
            "_run_id": ["run-12345678-abcd"],
            "_ingested_at": [pd.Timestamp.now("UTC")],
            "_ingest_date": [date(2026, 10, 17)],
            "_source_url": ["u"],
            "_record_hash": ["hash1"],
        }
    )
    result = load_dataframe(
        client, settings, df, table="wb_rtfp", mode="merge", run_id="run-12345678-abcd"
    )
    assert any("MERGE" in q.upper() for q in client.query_calls)
    assert result.dropped_staging is True
    assert client.deleted
    # Second run with same hash — still one MERGE, no duplicate insert_rows
    load_dataframe(
        client, settings, df, table="wb_rtfp", mode="merge", run_id="run-12345678-abcd"
    )
    assert len([q for q in client.query_calls if "MERGE" in q.upper()]) == 2
    assert client.insert_rows_json_calls == []


def test_dry_run_target_landing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_ALLOW_PROD", raising=False)
    monkeypatch.setenv("CI", "")
    from ingestion.common.config import TargetGuardError

    with pytest.raises(TargetGuardError):
        build_context("wb_rtfp", target="landing", dry_run=True)
