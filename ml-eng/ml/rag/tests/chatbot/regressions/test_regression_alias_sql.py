"""Short table aliases (cou, pro, wfp, ye) resolve by scope; invented columns are still caught."""
from __future__ import annotations

from ml.rag.chatbot.bq_sql_validate import validate_sql_column_allowlist

MART = "opentrace-prod-5ga4.mart_dev"

ALIASED = f"""
SELECT cou.country_name, pro.product_name, wfp.value, ye.year
FROM (SELECT country_iso3, product_key, value, year FROM `{MART}.fct_prices`) AS wfp
JOIN `{MART}.dim_product` AS pro ON pro.product_key = wfp.product_key
JOIN `{MART}.dim_geography` AS cou ON cou.country_iso3 = wfp.country_iso3
JOIN (SELECT DISTINCT year FROM `{MART}.fct_prices`) AS ye ON ye.year = wfp.year
"""


def test_short_aliases_are_not_read_as_columns() -> None:
    assert validate_sql_column_allowlist(ALIASED) is None


def test_invented_column_behind_an_alias_is_rejected() -> None:
    bad = ALIASED.replace("pro.product_name", "pro.crop_label")
    err = validate_sql_column_allowlist(bad) or ""
    assert "crop_label" in err


def test_derived_table_projection_is_the_only_visible_shape() -> None:
    bad = ALIASED.replace("wfp.value", "wfp.market_name")
    err = validate_sql_column_allowlist(bad) or ""
    assert "market_name" in err
