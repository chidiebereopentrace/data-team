"""Deterministic SQL templates as NL2SQL fallback (metrics-layer lite)."""
from __future__ import annotations

import re
from typing import Any

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_RANK_RE = re.compile(
    r"\b(highest|lowest|top|rank|ranking|most|least|which country)\b",
    re.IGNORECASE,
)
_PRODUCTION_RE = re.compile(
    r"\b(production|agricultural production|ag(?:ricultural)? output|crop production)\b",
    re.IGNORECASE,
)
_PRICE_RE = re.compile(
    r"\b(price|prices|producer price|cost of|how much)\b",
    re.IGNORECASE,
)
_FOOD_SEC_RE = re.compile(
    r"\b(food security|ipc|phase\s*[3-5]|insecure|crisis|emergency)\b",
    re.IGNORECASE,
)
_CONTINENT_RE = re.compile(
    r"\b(africa|african|sub[- ]?saharan|west africa|east africa|southern africa|"
    r"north africa|central africa)\b",
    re.IGNORECASE,
)

# Common crop aliases → FAOSTAT-ish product_name filter fragment.
_CROP_ALIASES: tuple[tuple[str, str], ...] = (
    ("maize", "Maize"),
    ("corn", "Maize"),
    ("rice", "Rice"),
    ("wheat", "Wheat"),
    ("millet", "Millet"),
    ("sorghum", "Sorghum"),
    ("cassava", "Cassava"),
    ("yam", "Yams"),
    ("soy", "Soybeans"),
    ("groundnut", "Groundnuts"),
    ("peanut", "Groundnuts"),
)


def _year_from_context(
    *,
    time_start: str | None,
    time_end: str | None,
    query: str,
) -> int | None:
    for raw in (time_start, time_end):
        if not raw:
            continue
        m = _YEAR_RE.search(str(raw))
        if m:
            return int(m.group(0))
    m = _YEAR_RE.search(query or "")
    if m:
        return int(m.group(0))
    return None


def _tables_set(selected_tables: set[str] | list[str] | None) -> set[str]:
    return {str(t).strip().split(".")[-1].lower() for t in (selected_tables or []) if str(t).strip()}


def _blob(query: str, entities: list[str] | None) -> str:
    q = query or ""
    if entities:
        return f"{q} {' '.join(str(e) for e in entities)}"
    return q


def _extract_crop(blob: str) -> str | None:
    low = (blob or "").lower()
    for alias, product in _CROP_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return product
    return None


def match_faostat_crop_rank(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if "stg_faostat_production" not in tables:
        return False
    blob = _blob(query, entities)
    if _extract_crop(blob) is None:
        return False
    if not (_PRODUCTION_RE.search(blob) or _RANK_RE.search(blob)):
        return False
    if not _CONTINENT_RE.search(blob) and not _RANK_RE.search(blob):
        return False
    return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None


def match_faostat_country_rank(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> bool:
    """True when the Africa production country-ranking template should apply."""
    tables = _tables_set(selected_tables)
    if "stg_faostat_production" not in tables:
        return False
    blob = _blob(query, entities)
    # Prefer crop-specific template when a crop is named.
    if _extract_crop(blob) is not None:
        return False
    if not _PRODUCTION_RE.search(blob):
        return False
    if not _RANK_RE.search(blob):
        return False
    if not _CONTINENT_RE.search(blob):
        return False
    return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None


def match_faostat_price_rank(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if "stg_faostat_prices" not in tables:
        return False
    blob = _blob(query, entities)
    if not _PRICE_RE.search(blob):
        return False
    if not (_RANK_RE.search(blob) or _CONTINENT_RE.search(blob)):
        return False
    return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None


def match_fews_food_security(
    *,
    query: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
) -> bool:
    tables = _tables_set(selected_tables)
    if "stg_fews_food_security" not in tables:
        return False
    blob = _blob(query, entities)
    if not _FOOD_SEC_RE.search(blob):
        return False
    return _year_from_context(time_start=time_start, time_end=time_end, query=query or "") is not None


def build_faostat_country_rank_sql(
    *,
    project_id: str,
    dataset: str,
    year: int,
    limit: int = 20,
    element: str = "Production",
    product_name: str | None = None,
) -> str:
    """Single-table FAOSTAT country production ranking SQL."""
    lim = max(1, min(int(limit or 20), 100))
    fqn = f"`{project_id}.{dataset}.stg_faostat_production`"
    product_clause = ""
    if product_name:
        safe = product_name.replace("'", "")
        product_clause = f"AND LOWER(product_name) LIKE '%{safe.lower()}%' "
    return (
        f"SELECT country_name, SUM(value) AS total "
        f"FROM {fqn} "
        f"WHERE year = {int(year)} "
        f"AND LOWER(element) LIKE '%{element.lower()}%' "
        f"{product_clause}"
        f"GROUP BY country_name "
        f"ORDER BY total DESC "
        f"LIMIT {lim}"
    )


def build_faostat_price_rank_sql(
    *,
    project_id: str,
    dataset: str,
    year: int,
    limit: int = 20,
    product_name: str | None = None,
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    fqn = f"`{project_id}.{dataset}.stg_faostat_prices`"
    product_clause = ""
    if product_name:
        safe = product_name.replace("'", "")
        product_clause = f"AND LOWER(product_name) LIKE '%{safe.lower()}%' "
    return (
        f"SELECT country_name, AVG(value) AS avg_price "
        f"FROM {fqn} "
        f"WHERE year = {int(year)} "
        f"AND LOWER(element) LIKE '%producer price%' "
        f"{product_clause}"
        f"GROUP BY country_name "
        f"ORDER BY avg_price DESC "
        f"LIMIT {lim}"
    )


def build_fews_food_security_sql(
    *,
    project_id: str,
    dataset: str,
    year: int,
    limit: int = 20,
) -> str:
    lim = max(1, min(int(limit or 20), 100))
    fqn = f"`{project_id}.{dataset}.stg_fews_food_security`"
    return (
        f"SELECT country, admin_1, AVG(phase_code) AS avg_phase "
        f"FROM {fqn} "
        f"WHERE year = {int(year)} "
        f"AND phase_code IS NOT NULL "
        f"GROUP BY country, admin_1 "
        f"ORDER BY avg_phase DESC "
        f"LIMIT {lim}"
    )


def try_sql_template(
    *,
    query: str,
    project_id: str,
    dataset: str,
    selected_tables: set[str] | list[str] | None,
    entities: list[str] | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    limit: int = 20,
) -> dict[str, Any] | None:
    """
    Return ``{\"sql\": ..., \"template\": ...}`` when a template matches.

    NL2SQL remains primary; call this only after NL2SQL yields nothing or all prepares fail.
    Match order: crop production rank → country production rank → prices → FEWS food security.
    """
    if not project_id or not dataset:
        return None
    year = _year_from_context(time_start=time_start, time_end=time_end, query=query or "")
    blob = _blob(query, entities)
    crop = _extract_crop(blob)

    if match_faostat_crop_rank(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
    ):
        assert year is not None and crop is not None
        return {
            "sql": build_faostat_country_rank_sql(
                project_id=project_id,
                dataset=dataset,
                year=year,
                limit=limit,
                product_name=crop,
            ),
            "template": "faostat_crop_rank",
            "year": year,
            "product_name": crop,
        }

    if match_faostat_country_rank(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
    ):
        assert year is not None
        return {
            "sql": build_faostat_country_rank_sql(
                project_id=project_id,
                dataset=dataset,
                year=year,
                limit=limit,
            ),
            "template": "faostat_country_rank",
            "year": year,
        }

    if match_faostat_price_rank(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
    ):
        assert year is not None
        return {
            "sql": build_faostat_price_rank_sql(
                project_id=project_id,
                dataset=dataset,
                year=year,
                limit=limit,
                product_name=crop,
            ),
            "template": "faostat_price_rank",
            "year": year,
        }

    if match_fews_food_security(
        query=query,
        selected_tables=selected_tables,
        entities=entities,
        time_start=time_start,
        time_end=time_end,
    ):
        assert year is not None
        return {
            "sql": build_fews_food_security_sql(
                project_id=project_id,
                dataset=dataset,
                year=year,
                limit=limit,
            ),
            "template": "fews_food_security",
            "year": year,
        }

    return None
