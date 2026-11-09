"""Fill coverage letters, datapoint counts, passage counts, and ACF scores.

A datapoint is one non-null value cell. A cell in the mart is Federated.
The same cell in landing and not in the mart is Partial. Neither is a Gap.
"""
from __future__ import annotations

import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import yaml
from acf import from_row, score_evidence
from acf.enums import ClaimLevel, QuestionType
from google.cloud import bigquery
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
CROSSWALK = Path(__file__).resolve().parent / "coverage_crosswalk.yaml"
WIDE_YEAR = re.compile(r"^Y((?:19|20)\d{2})$", re.IGNORECASE)


def _read_env(path: Path, prefix: str | None = None) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().replace("export ", "", 1).strip()
        value = value.strip().strip("'\"")
        if prefix and not key.startswith(prefix):
            continue
        if key and key not in os.environ:
            os.environ[key] = value


def _load_dotenv() -> None:
    _read_env(ROOT.parent / "data" / "local" / ".env")
    _read_env(ROOT.parent.parent / "ml-eng" / "config" / ".env", prefix="QDRANT_")


def _key_file() -> None:
    if (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip():
        return
    key = ROOT.parent / "data" / "local" / "keys" / "opentrace-bq-key.json"
    if key.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key)


def quote_bq(name: str) -> str:
    return "`" + str(name).replace("`", "") + "`"


def _client() -> tuple[bigquery.Client, str]:
    _load_dotenv()
    _key_file()
    project = (os.environ.get("BQ_PROJECT") or "").strip()
    if not project:
        raise SystemExit("Set BQ_PROJECT (for example in data-eng/data/local/.env)")
    return bigquery.Client(project=project), project


def _columns(client: bigquery.Client, project: str, dataset: str, table: str) -> list[str]:
    sql = f"""
    SELECT column_name
    FROM {quote_bq(project)}.{quote_bq(dataset)}.INFORMATION_SCHEMA.COLUMNS
    WHERE table_name = @table
    ORDER BY ordinal_position
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("table", "STRING", table)]
        ),
    )
    return [row.column_name for row in job.result()]


def _find(columns: list[str], *candidates: str) -> str | None:
    by_lower = {name.lower(): name for name in columns}
    for candidate in candidates:
        hit = by_lower.get(candidate.lower())
        if hit:
            return hit
    return None


def _wide_years(columns: list[str]) -> list[str]:
    return [name for name in columns if WIDE_YEAR.match(name)]


def _iso_maps(client: bigquery.Client, project: str) -> tuple[dict[str, str], dict[str, str]]:
    sql = f"""
    SELECT area_code, country_iso2, country_iso3
    FROM {quote_bq(project)}.intermediate_dev.int_faostat_area_iso
    WHERE country_iso3 IS NOT NULL
    """
    by_area: dict[str, str] = {}
    by_iso2: dict[str, str] = {}
    for row in client.query(sql).result():
        iso3 = str(row.country_iso3).upper()
        if row.area_code:
            by_area[str(row.area_code)] = iso3
        if row.country_iso2:
            by_iso2[str(row.country_iso2).upper()] = iso3
    return by_area, by_iso2


def _text_expr(column: str | None) -> str:
    if not column:
        return "CAST(NULL AS STRING)"
    return f"CAST({quote_bq(column)} AS STRING)"


def _matches(spec: dict, product, element, metric, indicator, property, measure) -> bool:
    def exact(expected, value) -> bool:
        return (value or "").strip().lower() == expected.lower()

    def pattern(expr, value) -> bool:
        return re.search(expr, (value or "").strip(), re.IGNORECASE) is not None

    if spec.get("element"):
        if not exact(spec["element"], element):
            return False
    elif spec.get("element_regex") and not pattern(spec["element_regex"], element):
        return False
    if spec.get("item_regex") and not pattern(spec["item_regex"], product):
        return False
    if spec.get("metric_regex") and not pattern(spec["metric_regex"], metric):
        return False
    if spec.get("indicator_regex") and not pattern(spec["indicator_regex"], indicator):
        return False
    if spec.get("property_regex") and not pattern(spec["property_regex"], property):
        return False
    if spec.get("measure_type") and not exact(spec["measure_type"], measure):
        return False
    return True


_MART_ROWS: dict[tuple, list] = {}
_LANDING_ROWS: dict[tuple, list] = {}


def _opt_expr(columns: list[str], name: str, fallback: str) -> str:
    found = _find(columns, name)
    if not found:
        return fallback
    return f"t.{quote_bq(found)}"


def _as_date(value, year: int) -> str:
    if hasattr(value, "date"):
        value = value.date()
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    if value:
        return str(value)[:10]
    return f"{int(year):04d}-01-01"


def _data_level(value) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"global", "national", "sub_national", "community"}:
        return text
    return None


def _mart_rows(client, project, source: dict, spec: dict) -> list[dict]:
    mart = source.get("mart")
    if not mart:
        return []
    table = mart["table"]
    columns = _columns(client, project, "mart_dev", table)
    value = spec.get("value") or mart["value"]
    if value not in columns:
        raise SystemExit(f"{table} has no value column {value}")
    year_sql = mart.get("year_sql") or quote_bq(mart.get("year") or "year")
    iso_sql = mart.get("iso_sql") or quote_bq(mart.get("iso") or "country_iso3")
    source_col = mart.get("source") or "source_key"
    source_sql = quote_bq(source_col) if source_col in columns else "CAST(NULL AS STRING)"
    product_sql = spec.get("product_sql") or mart.get("product_sql") or "CAST(NULL AS STRING)"
    join_sql = (mart.get("join") or "").replace("{project}", project)
    where = mart.get("where") or "TRUE"
    tier_sql = _opt_expr(columns, "tier", "CAST(NULL AS INT64)")
    level_sql = _opt_expr(columns, "data_level", "CAST(NULL AS STRING)")
    asof_sql = _opt_expr(columns, "as_of_date", "CAST(NULL AS DATE)")
    cache_key = (table, value, year_sql, iso_sql, product_sql, join_sql, where)
    if cache_key not in _MART_ROWS:
        print(f"  pull mart {table}.{value}", flush=True)
        sql = f"""
        SELECT iso, product, year, element, metric, indicator, property, measure,
               source_id, COUNT(*) AS n,
               ANY_VALUE(tier) AS tier,
               ANY_VALUE(data_level) AS data_level,
               MAX(as_of_date) AS as_of_date
        FROM (
          SELECT UPPER(CAST({iso_sql} AS STRING)) AS iso,
                 {product_sql} AS product,
                 {year_sql} AS year,
                 CAST({source_sql} AS STRING) AS source_id,
                 {_text_expr(_find(columns, "element"))} AS element,
                 {_text_expr(_find(columns, "metric"))} AS metric,
                 {_text_expr(_find(columns, "indicator"))} AS indicator,
                 {_text_expr(_find(columns, "soil_property"))} AS property,
                 {_text_expr(_find(columns, "measure_type"))} AS measure,
                 {tier_sql} AS tier,
                 {level_sql} AS data_level,
                 {asof_sql} AS as_of_date
          FROM {quote_bq(project)}.mart_dev.{quote_bq(table)} t
          {join_sql}
          WHERE {quote_bq(value)} IS NOT NULL
            AND {where}
        )
        WHERE iso IS NOT NULL AND year IS NOT NULL
        GROUP BY iso, product, year, element, metric, indicator, property, measure, source_id
        """
        pulled = []
        for row in client.query(sql).result():
            pulled.append(
                {
                    "iso": row.iso,
                    "product": (row.product or "").strip(),
                    "year": int(row.year),
                    "element": row.element,
                    "metric": row.metric,
                    "indicator": row.indicator,
                    "property": row.property,
                    "measure": row.measure,
                    "source_id": None if row.source_id is None else str(row.source_id),
                    "n": int(row.n),
                    "tier": None if row.tier is None else int(row.tier),
                    "data_level": _data_level(row.data_level),
                    "as_of": _as_date(row.as_of_date, int(row.year)),
                }
            )
        _MART_ROWS[cache_key] = pulled
    kept = []
    for row in _MART_ROWS[cache_key]:
        if _matches(spec, row["product"], row["element"], row["metric"], row["indicator"], row["property"], row["measure"]):
            kept.append(row)
    return kept


def _landing_faostat(client, project, landing: dict, spec: dict, area_iso: dict[str, str]) -> list[dict]:
    dataset = landing["dataset"]
    table = landing["table"]
    cache_key = ("faostat", dataset, table)
    if cache_key not in _LANDING_ROWS:
        columns = _columns(client, project, dataset, table)
        area = _find(columns, "area_code", "recipient_country_code", "country_code")
        years = _wide_years(columns)
        if not years or not area:
            raise SystemExit(f"{dataset}.{table} has no year columns or area code")
        print(f"  pull landing {dataset}.{table}", flush=True)
        product = _find(columns, "item")
        points = ", ".join(
            "STRUCT('"
            + name
            + "' AS year_col, SAFE_CAST("
            + quote_bq(name)
            + " AS FLOAT64) AS value)"
            for name in years
        )
        sql = f"""
        SELECT CAST({quote_bq(area)} AS STRING) AS area_code,
               {_text_expr(product)} AS product,
               year_point.year_col AS year_col,
               {_text_expr(_find(columns, "element"))} AS element,
               {_text_expr(_find(columns, "indicator"))} AS indicator,
               COUNT(*) AS n
        FROM {quote_bq(project)}.{quote_bq(dataset)}.{quote_bq(table)},
        UNNEST([{points}]) AS year_point
        WHERE year_point.value IS NOT NULL
        GROUP BY area_code, product, year_col, element, indicator
        """
        pulled = []
        for row in client.query(sql).result():
            iso = area_iso.get(str(row.area_code))
            year_match = WIDE_YEAR.match(row.year_col or "")
            if not iso or not year_match:
                continue
            pulled.append(
                {
                    "iso": iso,
                    "product": (row.product or "").strip(),
                    "year": int(year_match.group(1)),
                    "element": row.element,
                    "metric": None,
                    "indicator": row.indicator,
                    "property": None,
                    "measure": None,
                    "n": int(row.n),
                }
            )
        _LANDING_ROWS[cache_key] = pulled
    kept = []
    for row in _LANDING_ROWS[cache_key]:
        if _matches(spec, row["product"], row["element"], None, row["indicator"], None, None):
            kept.append(row)
    return kept


def _landing_hdx(client, project, iso2_to_iso3: dict[str, str], spec: dict) -> list[dict]:
    cache_key = ("hdx",)
    if cache_key not in _LANDING_ROWS:
        sql = f"""
        SELECT table_name
        FROM {quote_bq(project)}.landing_dev.INFORMATION_SCHEMA.TABLES
        WHERE table_type = 'BASE TABLE'
          AND STARTS_WITH(table_name, 'hdx_wfp_food_prices_')
        """
        tables = [row.table_name for row in client.query(sql).result()]
        parts = []
        for table in tables:
            code = table.rsplit("_", 1)[-1].upper()
            iso = code if len(code) == 3 else iso2_to_iso3.get(code)
            if not iso:
                continue
            parts.append(
                f"""
                SELECT '{iso}' AS iso, commodity AS product,
                       SAFE_CAST(REGEXP_EXTRACT(date, r'(?:19|20)\\d{{2}}') AS INT64) AS year,
                       COUNT(*) AS n
                FROM {quote_bq(project)}.landing_dev.{quote_bq(table)}
                WHERE price IS NOT NULL
                GROUP BY product, year
                """
            )
        print(f"  pull landing hdx ({len(parts)} countries)", flush=True)
        pulled = []
        if parts:
            for row in client.query(" UNION ALL ".join(parts)).result():
                if row.iso and row.year is not None:
                    pulled.append(
                        {
                            "iso": row.iso,
                            "product": (row.product or "").strip(),
                            "year": int(row.year),
                            "element": None,
                            "metric": None,
                            "indicator": None,
                            "property": None,
                            "measure": None,
                            "n": int(row.n),
                        }
                    )
        _LANDING_ROWS[cache_key] = pulled
    kept = []
    for row in _LANDING_ROWS[cache_key]:
        if _matches(spec, row["product"], None, None, None, None, None):
            kept.append(row)
    return kept


def _landing_harveststat(client, project, iso2_to_iso3: dict[str, str], known_iso: set[str], spec: dict) -> list[dict]:
    cache_key = ("harveststat",)
    if cache_key not in _LANDING_ROWS:
        print("  pull landing harveststat_africa_yield", flush=True)
        table = f"{quote_bq(project)}.landing_dev.harveststat_africa_yield"
        sql = f"""
        SELECT country_code, product, harvest_year, COUNT(*) AS n
        FROM {table}
        WHERE SAFE_CAST(yield AS FLOAT64) IS NOT NULL
        GROUP BY country_code, product, harvest_year
        """
        pulled = []
        for row in client.query(sql).result():
            code = (row.country_code or "").upper()
            iso = code if code in known_iso else iso2_to_iso3.get(code)
            year = None
            year_match = re.search(r"(?:19|20)\d{2}", str(row.harvest_year or ""))
            if year_match:
                year = int(year_match.group(0))
            if iso and year is not None:
                pulled.append(
                    {
                        "iso": iso,
                        "product": (row.product or "").strip(),
                        "year": year,
                        "element": None,
                        "metric": None,
                        "indicator": None,
                        "property": None,
                        "measure": None,
                        "n": int(row.n),
                    }
                )
        _LANDING_ROWS[cache_key] = pulled
    kept = []
    for row in _LANDING_ROWS[cache_key]:
        if _matches(spec, row["product"], None, None, None, None, None):
            kept.append(row)
    return kept


def _landing_rows(client, project, source: dict, spec: dict, area_iso, iso2_to_iso3, known_iso) -> list[dict]:
    landing = source.get("landing")
    if not landing:
        return []
    kind = landing["kind"]
    if kind == "faostat_wide":
        return _landing_faostat(client, project, landing, spec, area_iso)
    if kind == "hdx_prices":
        return _landing_hdx(client, project, iso2_to_iso3, spec)
    if kind == "harveststat_yield":
        return _landing_harveststat(client, project, iso2_to_iso3, known_iso, spec)
    raise SystemExit(f"Unknown landing kind {kind}")


def _resolve_product(name: str, match, livestock_names: list[tuple[str, str]]) -> str | None:
    if not name:
        return None
    key = match(name)
    if key:
        return key
    low = name.lower()

    def has_word(*words: str) -> bool:
        return any(re.search(r"(^|[^a-z])" + re.escape(word) + r"([^a-z]|$)", low) for word in words)

    if has_word("egg", "eggs"):
        return "poultry_layer"
    if has_word("milk"):
        return "dairy"
    if has_word("chicken", "chickens", "hen", "hens", "poultry"):
        return "poultry_broiler"
    if has_word("cattle", "beef"):
        return "cattle_beef"
    if has_word("goat", "goats"):
        return "goats"
    if has_word("sheep", "mutton", "lamb"):
        return "sheep"
    if has_word("pig", "pigs", "swine"):
        return "pigs"
    if has_word("camel", "camels"):
        return "camels"
    if has_word("tilapia"):
        return "tilapia"
    if has_word("catfish"):
        return "catfish"
    if has_word("shrimp", "prawn", "prawns"):
        return "shrimp"
    if has_word("honey", "beeswax"):
        return "honey"
    if has_word("rabbit", "rabbits"):
        return "rabbits"
    if has_word("donkey", "donkeys"):
        return "donkeys"
    if has_word("fish", "crustacea", "crustacean", "mollusc", "molluscs"):
        if has_word("marine", "ocean", "pelagic", "demersal"):
            return "fish_capture_marine"
        return "fish_capture_inland"
    for token, crop_key in livestock_names:
        if token and token in low:
            return crop_key
    return None


def _letter_for(mart_keys: set, landing_keys: set, n_sources: int) -> tuple[str, int, int]:
    partial = landing_keys - mart_keys
    if not mart_keys and not partial:
        return "G", 0, 0
    years = [item[1] if isinstance(item, tuple) else item for item in (mart_keys | partial)]
    return ("P" if partial else "F"), min(years), n_sources


def _grain(row: dict) -> tuple:
    return (
        row["iso"],
        row["product"],
        row["year"],
        row.get("element") or "",
        row.get("metric") or "",
        row.get("indicator") or "",
        row.get("property") or "",
        row.get("measure") or "",
    )


def _note(headline: dict, side: str, source_name: str, value: str, rows: list[dict]) -> None:
    grouped: dict[tuple, int] = {}
    for row in rows:
        grain = _grain(row)
        grouped[grain] = grouped.get(grain, 0) + int(row["n"])
    for grain, n in grouped.items():
        headline[(side, source_name, value, grain)] = n


def _headline_total(headline: dict) -> int:
    mart_cells = {
        (source, grain[0], grain[1], grain[2])
        for (side, source, _value, grain) in headline
        if side == "mart"
    }
    total = 0
    for (side, source, _value, grain), n in headline.items():
        if side == "landing" and (source, grain[0], grain[1], grain[2]) in mart_cells:
            continue
        total += n
    return total


def _acf_score(records: list[dict], place: str) -> int:
    claims = []
    for record in records:
        if record["tier"] not in (1, 2, 3) or not record["data_level"]:
            continue
        metric = record["metric"] or record["element"] or record["indicator"] or "value"
        try:
            claims.append(
                from_row(
                    {
                        "tier": record["tier"],
                        "data_level": record["data_level"],
                        "as_of_date": record["as_of"],
                        "metric": metric,
                        "direction": "unknown",
                        "source_id": record["source_id"],
                        "place_scope": place,
                    },
                    geo_field="place_scope",
                )
            )
        except ValueError:
            continue
    if not claims:
        return 0
    return score_evidence(
        claims,
        claim_level=ClaimLevel.NATIONAL,
        question_type=QuestionType.STRUCTURAL,
    ).score


def _pack(mart: set, landing: set, srcs: dict):
    by_iso_mart, by_iso_landing = {}, {}
    by_crop_mart, by_crop_landing = {}, {}
    sources_by_iso = {}
    for iso, crop, year in mart:
        by_iso_mart.setdefault(iso, set()).add((crop, year))
        by_crop_mart.setdefault((iso, crop), set()).add(year)
        sources_by_iso.setdefault(iso, set()).update(srcs.get((iso, crop, year), ()))
    for iso, crop, year in landing:
        by_iso_landing.setdefault(iso, set()).add((crop, year))
        by_crop_landing.setdefault((iso, crop), set()).add(year)
    return by_iso_mart, by_iso_landing, by_crop_mart, by_crop_landing, sources_by_iso


_ONTOLOGY = ROOT.parent.parent / "ml-eng" / "opentrace_ontology" / "data" / "ontology.sqlite"
_COLLECTIONS = (
    ("QDRANT_COLLECTION_NEWS", "news_data"),
    ("QDRANT_COLLECTION_ACADEMIC_PAPERS", "academic_papers"),
    ("QDRANT_COLLECTION_POLICIES", "policies"),
    ("QDRANT_COLLECTION_PUBLIC_REPORTS", "public_reports"),
    ("QDRANT_COLLECTION_FORMATION", "formation"),
    ("QDRANT_COLLECTION_OTA_INSIGHTS", "OTA_insights"),
)
_COLLECTION_CATEGORIES = {
    "news_data": ("MKT", "HUM"),
    "academic_papers": ("TECH",),
    "policies": ("POL", "EVT"),
    "public_reports": ("NUT", "CLIM", "LAND"),
    "formation": ("TECH", "FSVC"),
    "OTA_insights": ("PROD", "ECON", "INV"),
}
_COLUMN_FIELD = {
    "element": "element",
    "metric": "metric",
    "indicator": "indicator",
    "indicator_name": "indicator",
    "product_name": "product",
    "measure_type": "measure",
    "soil_property": "property",
}


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _spec_fields_match(spec: dict, fields: dict) -> bool:
    def any_exact(key: str, expected: str) -> bool:
        return any((item or "").strip().lower() == expected.lower() for item in fields.get(key, ()))

    def any_pattern(key: str, expr: str) -> bool:
        return any(re.search(expr, (item or "").strip(), re.IGNORECASE) for item in fields.get(key, ()))

    if spec.get("element"):
        if not any_exact("element", spec["element"]):
            return False
    elif spec.get("element_regex") and not any_pattern("element", spec["element_regex"]):
        return False
    if spec.get("item_regex") and not any_pattern("product", spec["item_regex"]):
        return False
    if spec.get("metric_regex") and not any_pattern("metric", spec["metric_regex"]):
        return False
    if spec.get("indicator_regex") and not any_pattern("indicator", spec["indicator_regex"]):
        return False
    if spec.get("property_regex") and not any_pattern("property", spec["property_regex"]):
        return False
    if spec.get("measure_type") and not any_exact("measure", spec["measure_type"]):
        return False
    return True


def _text_parts(value) -> list[str]:
    parts = []
    for item in _as_list(value):
        if not isinstance(item, str):
            continue
        for part in re.split(r"[;|,]", item):
            part = part.strip()
            if part:
                parts.append(part)
    return parts


def _count_passages(countries, inds, ccrops, levels, by_name, sources, match, livestock_names, iso2_to_iso3, cats):
    _load_dotenv()
    url = (os.environ.get("QDRANT_URL") or "").strip()
    if not url:
        raise SystemExit("Set QDRANT_URL and QDRANT_API_KEY")
    if not _ONTOLOGY.exists():
        raise SystemExit(f"Ontology database not found: {_ONTOLOGY}")
    db = sqlite3.connect(f"file:{_ONTOLOGY.as_posix()}?mode=ro", uri=True)
    targets: dict[str, list] = defaultdict(list)
    for entity_id, table, column, stored in db.execute(
        'SELECT entity_id, "table", "column", stored_value FROM value_map'
    ):
        targets[entity_id].append((table, column, stored or ""))

    def gathered(entity_id: str, fact_table: str) -> tuple[bool, dict]:
        fields: dict[str, list] = defaultdict(list)
        hit = False
        for table, column, stored in targets.get(entity_id, ()):
            field = _COLUMN_FIELD.get(column)
            if not field:
                continue
            if table == fact_table:
                hit = True
            if field == "product" or table == fact_table or table.startswith("dim_"):
                fields[field].append(stored)
        return hit, fields

    entity_indicators: dict[str, set] = defaultdict(set)
    entity_crops: dict[str, set] = {}
    for entity_id, rows in targets.items():
        crop_keys: set[str] = set()
        if entity_id.startswith(("OT:CROP:", "OT:ANIMAL:", "OT:PRODUCT:")):
            resolved = _resolve_product(entity_id.split(":")[-1].replace("_", " "), match, livestock_names)
            if resolved:
                crop_keys.add(resolved)
            for _table, column, stored in rows:
                if column == "product_name":
                    resolved = _resolve_product(stored, match, livestock_names)
                    if resolved:
                        crop_keys.add(resolved)
        entity_crops[entity_id] = crop_keys

    for ind in inds:
        rule = by_name[ind["name"]]
        if rule.get("empty"):
            continue
        for part in rule.get("parts") or [rule]:
            fact = (sources[part["source"]].get("mart") or {}).get("table")
            if not fact:
                continue
            for entity_id in targets:
                hit, fields = gathered(entity_id, fact)
                if hit and _spec_fields_match(part, fields):
                    entity_indicators[entity_id].add(ind["id"])

    cat_names = {cat["code"]: cat["name"].lower() for cat in cats}
    by_cat: dict[str, list[str]] = defaultdict(list)
    all_by_cat: dict[str, list[str]] = defaultdict(list)
    for ind in inds:
        all_by_cat[ind["cat"]].append(ind["id"])
        if not by_name[ind["name"]].get("empty"):
            by_cat[ind["cat"]].append(ind["id"])

    def indicators_for_collection(collection_name: str) -> set[str]:
        found: set[str] = set()
        for code in _COLLECTION_CATEGORIES.get(collection_name, ()):
            found.update(all_by_cat.get(code, ()))
        return found

    def category_ids(text: str) -> set[str]:
        found: set[str] = set()
        for code, name in cat_names.items():
            if name and name in text:
                found.update(by_cat.get(code, ()))
        if "agricultural economics" in text:
            found.update(by_cat.get("ECON", ()))
        return found

    order = {country["iso"]: index for index, country in enumerate(countries)}
    names = {country["name"].lower(): country["iso"] for country in countries}
    crop_lists = {iso: {row["k"] for row in ccrops[iso]} for iso in order}
    passagecount = {ind["id"]: [0] * len(countries) for ind in inds}

    def country_token(raw) -> str | None:
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if text.startswith("OT:GEO:"):
            text = text.split(":")[-1]
        upper = text.upper()
        if upper in order:
            return upper
        if upper in iso2_to_iso3 and iso2_to_iso3[upper] in order:
            return iso2_to_iso3[upper]
        return names.get(text.lower())

    client = QdrantClient(
        url=url,
        api_key=(os.environ.get("QDRANT_API_KEY") or "").strip() or None,
        timeout=120,
        check_compatibility=False,
    )
    payload_keys = [
        "ontology_ids",
        "ontology_countries",
        "geo_country_primary",
        "geo_countries",
        "geo_scope",
        "country",
        "commodity",
        "commodity_group",
        "domains",
        "metric",
    ]
    total = 0
    for env_name, default in _COLLECTIONS:
        name = (os.environ.get(env_name) or default).strip() or default
        try:
            expected = client.get_collection(name).points_count
        except Exception as exc:
            raise SystemExit(f"Qdrant collection {name} could not be read: {exc}") from exc
        if expected is None:
            raise SystemExit(f"Qdrant collection {name} did not report a point count")
        collection_inds = indicators_for_collection(name)
        seen = 0
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=payload_keys,
                with_vectors=False,
            )
            if not points:
                break
            seen += len(points)
            for point in points:
                payload = point.payload or {}
                ids = [item for item in _as_list(payload.get("ontology_ids")) if isinstance(item, str)]
                found = set()
                for entity_id in ids:
                    iso = country_token(entity_id)
                    if iso:
                        found.add(iso)
                for raw in (
                    _text_parts(payload.get("ontology_countries"))
                    + _text_parts(payload.get("geo_country_primary"))
                    + _text_parts(payload.get("geo_countries"))
                    + _text_parts(payload.get("geo_scope"))
                    + _text_parts(payload.get("country"))
                ):
                    iso = country_token(raw)
                    if iso:
                        found.add(iso)
                if not found:
                    continue
                ind_ids: set[str] = set()
                point_crops: set[str] = set()
                for entity_id in ids:
                    ind_ids |= entity_indicators.get(entity_id, set())
                    point_crops |= entity_crops.get(entity_id, set())
                label_text = " ; ".join(
                    _text_parts(payload.get("domains")) + _text_parts(payload.get("metric"))
                ).lower()
                ind_ids |= category_ids(label_text)
                for raw in _as_list(payload.get("commodity")) + _as_list(payload.get("commodity_group")):
                    if isinstance(raw, str):
                        resolved = _resolve_product(raw, match, livestock_names)
                        if resolved:
                            point_crops.add(resolved)
                ind_ids -= collection_inds
                for ind_id in ind_ids | collection_inds:
                    for iso in found:
                        if ind_id not in collection_inds and levels[ind_id] != "Country" and not (point_crops & crop_lists[iso]):
                            continue
                        passagecount[ind_id][order[iso]] += 1
            if offset is None:
                break
        if seen != expected:
            raise SystemExit(f"{name}: scrolled {seen} points, collection reports {expected}")
        total += expected
        print(f"  passages {name} {expected}", flush=True)
    return total, passagecount


def fill_coverage(countries, inds, ccrops, crop_inds, levels, match, livestock, cats):
    rules = yaml.safe_load(CROSSWALK.read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in rules["indicators"]}
    missing = sorted({d["name"] for d in inds if d["name"] not in by_name})
    if missing:
        raise SystemExit("Crosswalk is missing indicators:\n" + "\n".join(missing))
    sources = rules["sources"]
    client, project = _client()
    area_iso, iso2_to_iso3 = _iso_maps(client, project)
    known_iso = {c["iso"] for c in countries}
    livestock_names = []
    for key, label, _tier in livestock:
        livestock_names.append((label.split("(")[0].strip().lower(), key))
        livestock_names.append((key.replace("_", " "), key))
    livestock_names.sort(key=lambda item: -len(item[0]))

    print("passages", flush=True)
    passages, passagecount = _count_passages(
        countries, inds, ccrops, levels, by_name, sources, match, livestock_names, iso2_to_iso3, cats
    )

    cache: dict[tuple, tuple] = {}
    headline: dict[tuple, int] = {}

    def load_part(part: dict):
        source = sources[part["source"]]
        token = (
            part["source"],
            part.get("element"),
            part.get("element_regex"),
            part.get("item_regex"),
            part.get("metric_regex"),
            part.get("indicator_regex"),
            part.get("property_regex"),
            part.get("measure_type"),
            part.get("value"),
        )
        if token in cache:
            return cache[token]
        print(f"  query {part['source']} {part.get('element') or part.get('item_regex') or part.get('metric_regex') or ''}", flush=True)
        mart_rows = _mart_rows(client, project, source, part)
        landing_rows = _landing_rows(client, project, source, part, area_iso, iso2_to_iso3, known_iso)
        value = part.get("value") or (source.get("mart") or {}).get("value") or ""
        cache[token] = (mart_rows, landing_rows, part["source"], value)
        return cache[token]

    def resolve_rows(raw_mart, raw_landing, level):
        mart, landing = set(), set()
        sources_out: dict[tuple, set] = {}
        counts: dict[tuple, int] = defaultdict(int)
        evidence: dict[str, dict] = {}
        accepted_mart: list[dict] = []
        accepted_landing: list[dict] = []
        resolved: dict[str, str | None] = {}

        def crop_of(product: str):
            if product not in resolved:
                resolved[product] = _resolve_product(product, match, livestock_names)
            return resolved[product]

        def place(row: dict):
            iso = row["iso"]
            if iso not in known_iso:
                return None
            if level == "Country":
                key = (iso, "", row["year"])
            else:
                crop = crop_of(row["product"])
                if not crop:
                    return None
                key = (iso, crop, row["year"])
            return key

        mart_cells = set()
        for row in raw_mart:
            key = place(row)
            if key is None:
                continue
            mart.add(key)
            counts[key] += int(row["n"])
            accepted_mart.append(row)
            mart_cells.add((row["iso"], row["product"], row["year"]))
            if row["source_id"]:
                sources_out.setdefault(key, set()).add(row["source_id"])
            if row["tier"] in (1, 2, 3) and row["data_level"]:
                slot = evidence.setdefault(row["iso"], {})
                ident = (row["source_id"], row["tier"], row["data_level"])
                previous = slot.get(ident)
                if previous is None or row["as_of"] > previous["as_of"]:
                    slot[ident] = row
        for row in raw_landing:
            key = place(row)
            if key is None:
                continue
            landing.add(key)
            if (row["iso"], row["product"], row["year"]) in mart_cells:
                continue
            counts[key] += int(row["n"])
            accepted_landing.append(row)
        return mart, landing, sources_out, counts, evidence, accepted_mart, accepted_landing

    per_indicator = {}
    empty_names = []
    for ind in inds:
        rule = by_name[ind["name"]]
        if rule.get("empty"):
            empty_names.append(ind["name"])
            per_indicator[ind["id"]] = (*_pack(set(), set(), {}), {}, {}, {})
            continue
        parts = rule.get("parts") or [rule]
        mart, landing = set(), set()
        srcs: dict[tuple, set] = {}
        counts: dict[tuple, int] = defaultdict(int)
        evidence: dict[str, dict] = {}
        for part in parts:
            raw_mart, raw_landing, source_name, value = load_part(part)
            (
                part_mart,
                part_landing,
                part_sources,
                part_counts,
                part_evidence,
                accepted_mart,
                accepted_landing,
            ) = resolve_rows(raw_mart, raw_landing, ind["level"])
            _note(headline, "mart", source_name, value, accepted_mart)
            _note(headline, "landing", source_name, value, accepted_landing)
            mart |= part_mart
            landing |= part_landing
            for key, n in part_counts.items():
                counts[key] += n
            for key, names in part_sources.items():
                srcs.setdefault(key, set()).update(names)
            for iso, slot in part_evidence.items():
                dest = evidence.setdefault(iso, {})
                for ident, record in slot.items():
                    previous = dest.get(ident)
                    if previous is None or record["as_of"] > previous["as_of"]:
                        dest[ident] = record
        by_crop_n: dict[tuple, int] = defaultdict(int)
        for (iso, crop, _year), n in counts.items():
            by_crop_n[(iso, crop)] += n
        per_indicator[ind["id"]] = (*_pack(mart, landing, srcs), counts, evidence, by_crop_n)

    cov, covmeta, covcount = {}, {}, {}
    letters = {"F": 0, "P": 0, "G": 0}
    for ind in inds:
        by_iso_mart, by_iso_landing, _by_crop_mart, _by_crop_landing, sources_by_iso, counts, evidence, _by_crop_n = per_indicator[ind["id"]]
        by_iso_n: dict[str, int] = defaultdict(int)
        for (iso, _crop, _year), n in counts.items():
            by_iso_n[iso] += n
        status, meta, count_row = "", [], []
        for country in countries:
            iso = country["iso"]
            letter, year, n_sources = _letter_for(
                by_iso_mart.get(iso, set()),
                by_iso_landing.get(iso, set()),
                len(sources_by_iso.get(iso, ())),
            )
            letters[letter] += 1
            status += letter
            if letter == "G":
                meta.append([0, 0, 0])
                count_row.append(0)
            else:
                score = _acf_score(list(evidence.get(iso, {}).values()), country["name"])
                meta.append([year, n_sources, score])
                count_row.append(by_iso_n.get(iso, 0))
        cov[ind["id"]] = status
        covmeta[ind["id"]] = meta
        covcount[ind["id"]] = count_row

    cropcov, cropn = {}, {}
    for country in countries:
        iso = country["iso"]
        rows, numbers = [], []
        for crop in ccrops[iso]:
            is_live = crop["cat"] == "Livestock & fisheries"
            chars, nums = [], []
            for iid in crop_inds:
                if (levels[iid] == "Value chain") != is_live:
                    chars.append("-")
                    nums.append(0)
                    continue
                _iso_mart, _iso_landing, by_crop_mart, by_crop_landing, _sources, _counts, _evidence, by_crop_n = per_indicator[iid]
                letter, _year, _n = _letter_for(
                    by_crop_mart.get((iso, crop["k"]), set()),
                    by_crop_landing.get((iso, crop["k"]), set()),
                    0,
                )
                chars.append(letter)
                nums.append(0 if letter == "G" else by_crop_n.get((iso, crop["k"]), 0))
            rows.append("".join(chars))
            numbers.append(nums)
        cropcov[iso] = rows
        cropn[iso] = numbers

    datapoints = _headline_total(headline)
    print(
        f"coverage letters F={letters['F']} P={letters['P']} G={letters['G']}"
        f" empty_indicators={len(set(empty_names))}",
        flush=True,
    )
    print(f"datapoints {datapoints} passages {passages}", flush=True)
    return cov, covmeta, cropcov, covcount, cropn, passages, passagecount, datapoints
