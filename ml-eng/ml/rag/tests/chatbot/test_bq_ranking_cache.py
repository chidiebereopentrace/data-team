"""Tests for session-scoped BQ ranking cache."""
from __future__ import annotations

from ml.rag.chatbot.bq_ranking_cache import (
    bq_results_from_cache,
    cache_entry_from_bq_results,
    fingerprint_ranking,
    is_ranking_follow_up,
)


def _ranked_item(*, sql: str, template: str = "faostat_country_rank") -> dict:
    return {
        "content": "ranked prose",
        "source": "bigquery",
        "metadata": {
            "bq_enrichment": "ranked_table",
            "sql": sql,
            "template": template,
            "ranked_rows": [
                {"rank": 1, "label": "Nigeria", "value": 100, "unit": "tonnes"},
                {"rank": 2, "label": "Ethiopia", "value": 80, "unit": "tonnes"},
            ],
        },
    }


def test_fingerprint_ranking_stable() -> None:
    dec = {"time_start": "2020-01-01", "time_end": "2020-12-31"}
    sql = "SELECT country_name FROM t WHERE year = 2020"
    a = fingerprint_ranking(dec, sql, "faostat_country_rank")
    b = fingerprint_ranking(dec, sql, "faostat_country_rank")
    assert a == b
    assert fingerprint_ranking(dec, sql.replace("2020", "2021"), "faostat_country_rank") != a


def test_cache_round_trip() -> None:
    sql = (
        "SELECT country_name, SUM(value) AS total "
        "FROM `proj.staging_dev.stg_faostat_production` WHERE year = 2020"
    )
    dec = {"time_start": "2020-01-01", "time_end": "2020-12-31"}
    item = _ranked_item(sql=sql)
    entry = cache_entry_from_bq_results([item], query="best activity 2020", decomposition=dec)
    assert entry is not None
    assert entry["fingerprint"]
    restored = bq_results_from_cache(entry)
    assert len(restored) == 1
    assert restored[0]["metadata"]["bq_enrichment"] == "ranked_table"


def test_is_ranking_follow_up_list_query() -> None:
    sql = "SELECT country_name FROM t WHERE year = 2020"
    dec = {"time_start": "2020-01-01", "time_end": "2020-12-31"}
    entry = cache_entry_from_bq_results([_ranked_item(sql=sql)], query="best 2020", decomposition=dec)
    assert entry is not None
    assert is_ranking_follow_up("list the top 10 countries", dec, entry)
    assert not is_ranking_follow_up("best agricultural activity in 2021", dec, entry)


def test_is_ranking_follow_up_export() -> None:
    sql = "SELECT country_name FROM t WHERE year = 2020"
    dec = {"time_start": "2020-01-01", "time_end": "2020-12-31"}
    entry = cache_entry_from_bq_results([_ranked_item(sql=sql)], query="best 2020", decomposition=dec)
    assert entry is not None
    assert is_ranking_follow_up("export this as csv", dec, entry)
