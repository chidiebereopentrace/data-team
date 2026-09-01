"""Facet geography → ISO3 list for warehouse engines (closed-world geo)."""
from __future__ import annotations

import re
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import _AFRICA_COUNTRY_ISO3
from ml.rag.chatbot.geo_regions import countries_for_regions, detect_regions_in_text

_ISO3_RE = re.compile(r"^[A-Z]{3}$")
_NAME_TO_ISO3: dict[str, str] = {k.casefold(): v for k, v in _AFRICA_COUNTRY_ISO3.items()}
_ISO3_IN_SQL_RE = re.compile(
    r"country_iso3\s*(?:=\s*'([A-Z]{3})'|IN\s*\(([^)]+)\))",
    re.I,
)
_VALID_ISO3 = frozenset(_AFRICA_COUNTRY_ISO3.values())


def name_to_iso3(label: str) -> str | None:
    """Map a country name or ISO3 token to ISO3."""
    raw = (label or "").strip()
    if not raw:
        return None
    if _ISO3_RE.match(raw.upper()):
        iso = raw.upper()
        return iso if iso in _VALID_ISO3 else None
    return _NAME_TO_ISO3.get(raw.casefold())


def resolve_geography_iso3(
    query: str,
    *,
    geography: list[str] | None = None,
    expanded_regions: list[str] | None = None,
) -> list[str]:
    """All ISO3 codes from facet geography / expanded regions / zone detection."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(label: str) -> None:
        iso = name_to_iso3(label)
        if iso and iso not in seen:
            seen.add(iso)
            out.append(iso)

    for g in geography or []:
        _add(str(g))

    if not out and expanded_regions:
        for region in expanded_regions:
            for country in countries_for_regions([str(region)]):
                _add(country)

    if not out:
        regions = detect_regions_in_text(query or "")
        if regions:
            for country in countries_for_regions(regions):
                _add(country)

    return out


def infer_country_iso3_from_query(query: str) -> str | None:
    """Single ISO3 from query text — word-boundary only (no COM substring bugs)."""
    q = (query or "").strip()
    if not q:
        return None
    q_cf = q.casefold()
    for name, iso in _AFRICA_COUNTRY_ISO3.items():
        if re.search(rf"\b{re.escape(name.casefold())}\b", q_cf):
            return iso
    for m in re.finditer(r"\b([A-Za-z]{3})\b", q):
        cand = m.group(1).upper()
        if cand in _VALID_ISO3:
            return cand
    return None


def extract_sql_country_iso3_literals(sql: str) -> list[str]:
    """Parse country_iso3 = 'XXX' or IN ('A','B') from SQL."""
    found: list[str] = []
    for m in _ISO3_IN_SQL_RE.finditer(sql or ""):
        single, in_list = m.group(1), m.group(2)
        if single:
            found.append(single.upper())
        elif in_list:
            for part in re.findall(r"'([A-Z]{3})'", in_list, re.I):
                found.append(part.upper())
    return found


def validate_sql_country_iso3_subset(
    sql: str,
    allowed_iso3: list[str],
) -> str | None:
    """Return error message if SQL uses ISO3 outside allowed facet list."""
    if len(allowed_iso3) < 2:
        return None
    allowed = set(allowed_iso3)
    literals = extract_sql_country_iso3_literals(sql)
    if not literals:
        return "multi-country panel requires country_iso3 IN facet list"
    bad = [iso for iso in literals if iso not in allowed]
    if bad:
        return f"country_iso3 {bad[0]} not in facet geography"
    if len(literals) == 1 and len(allowed) >= 2:
        return "multi-country panel must use country_iso3 IN (...), not single ="
    return None


__all__ = [
    "name_to_iso3",
    "resolve_geography_iso3",
    "infer_country_iso3_from_query",
    "extract_sql_country_iso3_literals",
    "validate_sql_country_iso3_subset",
]
