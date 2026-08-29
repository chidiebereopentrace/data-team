"""
Path B ACF facade tests — cited evidence → adapt → score_evidence.
"""
from __future__ import annotations

from datetime import date

from acf.enums import ClaimLevel, QuestionType

from ml.rag.chatbot.acf_scoring import (
    adapt_cited_claims,
    curated_product_acf,
    no_evidence_acf,
    score_cited_evidence,
)
from ml.rag.chatbot.acf_metadata import (
    context_item_to_acf_record,
    derive_tier_and_data_level,
    enrich_acf_payload_fields,
    project_bq_row_acf,
    stamp_temporal_direction,
    warehouse_row_to_acf_record,
)
from ml.rag.chatbot.bq_context_enrich import _stamp_acf_metadata
from ml.rag.chatbot.acf_question import classify_acf_question


def _news_item(**meta_extra):
    meta = {
        "doc_kind": "news_article",
        "geo_scope": "country",
        "geo_countries": "Kenya",
        "geo_country_primary": "Kenya",
        "published_at": "2025-06-01",
        "document_id": "news-doc-1",
        "domains": "yield",
        **meta_extra,
    }
    return {
        "content": "Maize yields declined in Kenya.",
        "source": "news",
        "_context_kind": "news",
        "metadata": meta,
    }


def test_no_evidence_acf_shape() -> None:
    acf = no_evidence_acf()
    assert acf.band == "no_evidence"
    assert acf.score == 0
    assert acf.explanation


def test_curated_product_acf_shape() -> None:
    acf = curated_product_acf()
    assert acf.band == "strong"
    assert acf.score == 90


def test_derive_tier_national() -> None:
    tier, dl = derive_tier_and_data_level(
        {"geo_scope": "country", "geo_countries": "Kenya"}
    )
    assert tier == 2
    assert dl == "national"


def test_derive_tier_sub_national() -> None:
    tier, dl = derive_tier_and_data_level(
        {"geo_scope": "country", "geo_countries": "Kenya", "region": "Nakuru"}
    )
    assert tier == 3
    assert dl == "sub_national"


def test_enrich_acf_payload_fields() -> None:
    out = enrich_acf_payload_fields(
        {
            "geo_scope": "country",
            "geo_countries": "Senegal",
            "published_at": "2024-03-15",
            "document_id": "abc",
        }
    )
    assert out["tier"] == 2
    assert out["data_level"] == "national"
    assert out["as_of_date"] == "2024-03-15"
    assert out["source_id"] == "abc"


def test_context_item_remaps_geo_scope_to_places() -> None:
    item = _news_item()
    record = context_item_to_acf_record(item)
    assert record is not None
    assert record["geo_scope"] == ["Kenya"]  # places, not coverage keyword
    assert record["data_level"] == "national"
    assert record["direction"] == "unknown"


def test_adapt_and_score_cited_news() -> None:
    items = [
        _news_item(),
        _news_item(
            document_id="news-doc-2",
            published_at="2025-05-01",
            geo_countries="Kenya",
        ),
    ]
    claims = adapt_cited_claims(items)
    assert len(claims) >= 1
    acf = score_cited_evidence(
        items,
        query="How is Kenya's maize production trending this season?",
        decomposition={"geography": ["Kenya"], "domains": ["yield"]},
        reference_date=date(2025, 7, 1),
    )
    assert 0 <= acf.score <= 100
    assert acf.band in {
        "very_strong",
        "strong",
        "moderate",
        "limited",
        "low",
        "no_evidence",
    }
    assert acf.band_label
    assert acf.explanation
    assert acf.claim_level == "national"


def test_score_empty_citations_is_no_evidence() -> None:
    acf = score_cited_evidence([], query="Anything?")
    assert acf.band == "no_evidence"
    assert acf.score == 0


def test_classify_sub_national_and_time_sensitive() -> None:
    c = classify_acf_question(
        "Which counties in Kenya have declining maize prices this month?",
        {"geography": ["Kenya"], "domains": ["price"]},
    )
    assert c.claim_level == ClaimLevel.SUB_NATIONAL
    assert c.question_type == QuestionType.TIME_SENSITIVE


def test_classify_structural() -> None:
    c = classify_acf_question(
        "How have yields changed over the past decade in Ghana?",
        {"geography": ["Ghana"], "domains": ["yield"]},
    )
    assert c.question_type == QuestionType.STRUCTURAL


def test_project_bq_row_acf() -> None:
    meta = project_bq_row_acf(
        {
            "country": "Kenya",
            "admin_1": "Nakuru",
            "year": 2024,
            "product": "maize",
            "yield": 2.1,
            "sql": "SELECT * FROM `proj.bronze.yield_raw_data`",
        }
    )
    assert meta["tier"] == 3
    assert meta["data_level"] == "sub_national"
    assert meta["as_of_date"] == "2024-01-01"
    assert meta["metric"] == "maize"
    assert "yield_raw_data" in meta["source_id"]
    assert meta["direction"] == "unknown"


def test_project_bq_row_acf_preserves_warehouse_contract() -> None:
    meta = project_bq_row_acf(
        {
            "tier": 1,
            "data_level": "sub_national",
            "place_scope": ["ETH", "Oromia"],
            "country_iso3": "ETH",
            "metric": "price_retail_maize",
            "source_key": "faostat_prices_eth",
            "as_of_date": "2024-06-30",
            "as_of_date_basis": "observation",
            "value": 42.5,
        }
    )
    assert meta["tier"] == 1
    assert meta["data_level"] == "sub_national"
    assert meta["place_scope"] == ["ETH", "Oromia"]
    assert meta["metric"] == "price_retail_maize"
    assert meta["source_id"] == "faostat_prices_eth"
    assert meta["as_of_date_basis"] == "observation"


def test_warehouse_row_to_acf_record_prices() -> None:
    row = {
        "tier": 1,
        "data_level": "sub_national",
        "place_scope": ["ETH", "ETR103", "Oromia"],
        "metric": "price_retail_maize",
        "source_id": "faostat_prices_eth",
        "as_of_date": "2024-06-30",
        "value": 42.5,
        "unit": "ETB/kg",
    }
    record = warehouse_row_to_acf_record(row)
    assert record is not None
    assert record["tier"] == 1
    assert record["data_level"] == "sub_national"
    assert record["geo_scope"] == ["ETH", "ETR103", "Oromia"]
    assert record["place_scope"] == ["ETH", "ETR103", "Oromia"]
    assert record["metric"] == "price_retail_maize"
    assert record["source_id"] == "faostat_prices_eth"
    assert record["direction"] == "unknown"


def test_warehouse_row_skips_null_as_of_date() -> None:
    row = {
        "tier": 3,
        "data_level": "community",
        "place_scope": ["KEN"],
        "metric": "household_snapshot",
        "source_id": "ilri_hh",
        "as_of_date": None,
        "value": 1.0,
    }
    assert warehouse_row_to_acf_record(row) is None


def test_warehouse_cited_item_adapts_and_scores() -> None:
    item = {
        "content": "{...}",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "tier": 1,
            "data_level": "national",
            "place_scope": ["KEN", "Kenya"],
            "metric": "ipc_phase_3",
            "source_id": "fews_ipc_ken",
            "as_of_date": "2025-05-01",
            "value": 2.1,
            "unit": "million",
        },
    }
    record = context_item_to_acf_record(item)
    assert record is not None
    assert record["tier"] == 1
    assert record["geo_scope"] == ["KEN", "Kenya"]
    claims = adapt_cited_claims([item])
    assert len(claims) == 1
    acf = score_cited_evidence(
        [item],
        query="How is Kenya food security?",
        reference_date=date(2025, 7, 1),
    )
    assert acf.band != "no_evidence"


def test_context_prefers_ingested_direction_and_magnitude() -> None:
    item = {
        "content": "prose",
        "source": "news",
        "_context_kind": "news",
        "metadata": {
            "geo_countries": "Kenya",
            "published_at": "2025-06-01",
            "document_id": "n1",
            "metric": "maize_yield",
            "direction": "decreasing",
            "magnitude": -12.0,
            "unit": "pct_yoy",
            "finding": "Maize yields declined 12% YoY.",
        },
    }
    record = context_item_to_acf_record(item)
    assert record is not None
    assert record["metric"] == "maize_yield"
    assert record["direction"] == "decreasing"
    assert record["magnitude"] == -12.0
    assert record["unit"] == "pct_yoy"
    assert "finding" not in record  # audit-only; not passed to from_payload


def test_bq_row_with_prior_value_scores() -> None:
    item = {
        "content": "{...}",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": project_bq_row_acf(
            {
                "country": "Kenya",
                "year": 2024,
                "product": "maize",
                "value": 88.0,
                "prior_value": 100.0,
                "sql": "SELECT * FROM bronze.yield_raw_data",
            }
        ),
    }
    acf = score_cited_evidence(
        [item],
        query="How is Kenya maize yield trending?",
        reference_date=date(2025, 1, 1),
    )
    assert acf.score >= 0
    assert acf.band != "no_evidence" or acf.score == 0  # may still score low
    # With one claim we expect a real score path
    claims = adapt_cited_claims([item])
    assert len(claims) == 1


def test_context_ranked_table_bq_with_stamped_metadata() -> None:
    item = {
        "content": "ranked table prose",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "bq_enrichment": "ranked_table",
            "as_of_date": "2020-01-01",
            "year": 2020,
            "metric": "Physical crop/livestock production output",
            "unit": "tonnes",
            "direction": "increasing",
            "value": 120.0,
            "prior_value": 100.0,
            "magnitude": 20.0,
            "coverage_strength": 1.0,
            "geo_country_primary": "Nigeria",
            "geo_countries": "Nigeria",
            "ranked_rows": [
                {
                    "rank": 1,
                    "label": "Nigeria",
                    "value": 120.0,
                    "unit": "tonnes",
                    "measure_label": "Production",
                }
            ],
            "sql": (
                "SELECT country_name, SUM(value) AS total "
                "FROM `proj.staging_dev.stg_faostat_production` "
                "WHERE year = 2020 AND element = 'Production'"
            ),
        },
    }
    record = context_item_to_acf_record(item)
    assert record is not None
    assert record["as_of_date"] == "2020-01-01"
    assert record["geo_scope"] == ["Nigeria"]
    assert record["direction"] == "increasing"
    assert record["metric"] != "general"
    claims = adapt_cited_claims([item])
    assert len(claims) == 1
    acf = score_cited_evidence(
        [item],
        query="Which country had the best agricultural activity in 2020?",
        reference_date=date(2021, 1, 1),
    )
    assert acf.band != "no_evidence"


def test_adapt_country_level_bq_without_region() -> None:
    item = {
        "content": "Ghana rice production 973000 t in 2020",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "country_name": "Ghana",
            "product_name": "Rice",
            "element": "Production",
            "year": 2020,
            "value": 973000,
            "unit": "t",
            "direction": "unknown",
            "geo_country_primary": "Ghana",
            "geo_countries": "Ghana",
            "as_of_date": "2020-01-01",
            "tier": 2,
            "sql": "SELECT * FROM `proj.staging_dev.stg_faostat_production`",
        },
    }
    claims = adapt_cited_claims([item])
    assert len(claims) == 1


def test_adapt_subnational_bq_with_region_uses_from_row() -> None:
    item = {
        "content": "Northern Ghana IPC phase 3",
        "source": "bigquery",
        "_context_kind": "bigquery",
        "metadata": {
            "region": "Northern",
            "country_name": "Ghana",
            "year": 2024,
            "value": 3,
            "as_of_date": "2024-06-01",
            "geo_country_primary": "Ghana",
            "sql": "SELECT * FROM `proj.staging_dev.stg_fews_ipc`",
        },
    }
    claims = adapt_cited_claims([item])
    assert len(claims) == 1


def test_stamp_temporal_direction_yoy_pair() -> None:
    meta = stamp_temporal_direction({"total_curr": 120.0, "total_prev": 100.0})
    assert meta["value"] == 120.0
    assert meta["prior_value"] == 100.0
    assert meta["direction"] == "increasing"
    assert meta["magnitude"] == 20.0


def test_project_bq_row_acf_maps_total_prev() -> None:
    meta = project_bq_row_acf(
        {
            "country_iso3": "KEN",
            "year": 2024,
            "total_curr": 110.0,
            "total_prev": 100.0,
        },
        table_hint="fct_production",
    )
    assert meta["direction"] == "increasing"
    assert meta["value"] == 110.0
    assert meta["prior_value"] == 100.0
    assert meta["magnitude"] == 10.0


def test_stamp_acf_metadata_preserves_place_scope() -> None:
    raw = {
        "country_iso3": "ETH",
        "tier": 2,
        "place_scope": ["ETH", "Oromia"],
        "metric": "production_maize_physical",
        "source_key": "faostat",
        "as_of_date_basis": "observation",
        "value": 1000,
        "year": 2020,
    }
    meta = _stamp_acf_metadata(
        {},
        None,
        sql="SELECT * FROM `p.mart_dev.fct_production` WHERE year = 2020",
        decomposition=None,
        row=raw,
    )
    assert meta["place_scope"] == ["ETH", "Oromia"]
    assert meta["source_key"] == "faostat"
    assert meta["as_of_date_basis"] == "observation"
    assert meta["metric"] == "production_maize_physical"
