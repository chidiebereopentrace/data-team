"""Facet geography → ISO3 list for warehouse engines (closed-world geo)."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from ml.rag.chatbot.bq_table_schema_yaml import _AFRICA_COUNTRY_ISO3
from ml.rag.chatbot.geo_regions import countries_for_regions, detect_regions_in_text

_ISO3_RE = re.compile(r"^[A-Z]{3}$")
_ISO2_RE = re.compile(r"^[A-Z]{2}$")


def _fold_country_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.replace("\u2019", "'"))
    return "".join(char for char in normalized if not unicodedata.combining(char)).casefold()


_NAME_TO_ISO3: dict[str, str] = {
    _fold_country_name(name): code for name, code in _AFRICA_COUNTRY_ISO3.items()
}
_COUNTRY_NAMES_LONGEST_FIRST = tuple(sorted(_AFRICA_COUNTRY_ISO3, key=len, reverse=True))
_ISO3_IN_SQL_RE = re.compile(
    r"country_iso3\s*(?:=\s*'([A-Z]{3})'|IN\s*\(([^)]+)\))",
    re.I,
)
_VALID_ISO3 = frozenset(_AFRICA_COUNTRY_ISO3.values())

# African ISO2 → ISO3 (fallback when opentrace_ontology is unavailable).
_AFRICA_ISO2_TO_ISO3: dict[str, str] = {
    "DZ": "DZA",
    "AO": "AGO",
    "BJ": "BEN",
    "BW": "BWA",
    "BF": "BFA",
    "BI": "BDI",
    "CV": "CPV",
    "CM": "CMR",
    "CF": "CAF",
    "TD": "TCD",
    "KM": "COM",
    "CG": "COG",
    "CD": "COD",
    "DJ": "DJI",
    "EG": "EGY",
    "GQ": "GNQ",
    "ER": "ERI",
    "SZ": "SWZ",
    "ET": "ETH",
    "GA": "GAB",
    "GM": "GMB",
    "GH": "GHA",
    "GN": "GIN",
    "GW": "GNB",
    "CI": "CIV",
    "KE": "KEN",
    "LS": "LSO",
    "LR": "LBR",
    "LY": "LBY",
    "MG": "MDG",
    "MW": "MWI",
    "ML": "MLI",
    "MR": "MRT",
    "MU": "MUS",
    "MA": "MAR",
    "MZ": "MOZ",
    "NA": "NAM",
    "NE": "NER",
    "NG": "NGA",
    "RW": "RWA",
    "ST": "STP",
    "SN": "SEN",
    "SC": "SYC",
    "SL": "SLE",
    "SO": "SOM",
    "ZA": "ZAF",
    "SS": "SSD",
    "SD": "SDN",
    "TZ": "TZA",
    "TG": "TGO",
    "TN": "TUN",
    "UG": "UGA",
    "ZM": "ZMB",
    "ZW": "ZWE",
}


def iso2_to_iso3(label: str) -> str | None:
    """Map an African ISO2 code to ISO3, or None when unknown / non-African."""
    raw = (label or "").strip().upper()
    if not _ISO2_RE.match(raw):
        return None
    return _AFRICA_ISO2_TO_ISO3.get(raw)


def name_to_iso3(label: str) -> str | None:
    """Map a country name, ISO3, or African ISO2 token to ISO3."""
    raw = (label or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if _ISO3_RE.match(upper):
        return upper if upper in _VALID_ISO3 else None
    if _ISO2_RE.match(upper):
        return _AFRICA_ISO2_TO_ISO3.get(upper)
    return _NAME_TO_ISO3.get(_fold_country_name(raw))


def countries_named_in_text(text: str) -> set[str]:
    """Canonical country names that appear as words in ``text``.

    Longer names are taken first so Guinea-Bissau is not also Guinea, and
    South Sudan is not also Sudan.
    """
    masked = _fold_country_name(text or "")
    if not masked.strip():
        return set()
    found: set[str] = set()
    for name in _COUNTRY_NAMES_LONGEST_FIRST:
        needle = _fold_country_name(name)
        pattern = re.compile(rf"\b{re.escape(needle)}\b")
        if not pattern.search(masked):
            continue
        found.add(name.casefold())
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
    return found


def iso3_to_canonical_name(iso3: str) -> str | None:
    """Canonical Africa gazetteer name for an ISO3 code."""
    iso = (iso3 or "").strip().upper()
    if not iso:
        return None
    for name, code in _AFRICA_COUNTRY_ISO3.items():
        if code == iso:
            return name
    return None


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
    "iso3_to_canonical_name",
    "name_to_iso3",
    "resolve_geography_iso3",
    "infer_country_iso3_from_query",
    "extract_sql_country_iso3_literals",
    "validate_sql_country_iso3_subset",
]
