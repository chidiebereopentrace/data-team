"""Capped YAML samples are not the label domain: unseen labels are rejected only against a complete domain."""

from __future__ import annotations

from ml.rag.chatbot.bq_sql_validate import label_domain, validate_sql_value_samples


def _sql(where: str, table: str = "fct_production") -> str:
    return f"SELECT year, value FROM `p.mart_dev.{table}` WHERE {where}"


def test_product_outside_capped_samples_is_not_rejected() -> None:
    assert label_domain({"fct_production"}, "product_name") is None
    sql = _sql("product_name = 'Bambara beans' AND element = 'Production'")
    assert validate_sql_value_samples(sql, {"fct_production"}) is None


def test_label_outside_a_complete_domain_is_rejected() -> None:
    domain = label_domain({"fct_production"}, "element")
    assert domain is not None and "production" in domain
    err = validate_sql_value_samples(_sql("element = 'prod'"), {"fct_production"})
    assert err is not None and "prod" in err


def test_price_type_domain_is_complete() -> None:
    assert validate_sql_value_samples(_sql("price_type = 'Retail'", "fct_prices"), {"fct_prices"}) is None
    assert validate_sql_value_samples(_sql("price_type = 'Retail price'", "fct_prices"), {"fct_prices"})
