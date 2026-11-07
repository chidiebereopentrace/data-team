"""Price vocabulary: every product the price mart holds, per country, from live BigQuery.

The table YAML value samples are truncated (~100 of ~1,000 product names), so
price SQL cannot be grounded on them. This artifact lists every product_key with
its dim_product label, and per country the price types, sources, units,
currencies and date span that actually exist. It also lists every market
(dim_market name and admin-1 region) with the product keys it holds, so a
question naming a market or region filters on market keys. Price templates
filter on these keys and never on free text.

Build:
  cd ml-eng && python -m ml.rag.chatbot.price_vocab --build
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from google.cloud import bigquery

from ml.rag.chatbot.agri_entities import CROP_ENTITY_RE
from ml.rag.chatbot.bq_mart_sql import mart_dataset
from ml.rag.chatbot.bq_table_schema_yaml import value_samples_for_table
from ml.rag.chatbot.geo_iso3 import name_to_iso3
from ml.rag.local_env import load_rag_dotenv

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "price_vocab.json"

# Words that describe a price, not a product. They never select a product.
_GENERIC = frozenset(
    {
        "price", "cost", "market", "retail", "wholesale", "producer", "average", "mean",
        "national", "food", "local", "imported", "import", "grain", "kg", "tonne", "the", "of", "and", "in", "for", "per",
        "quality", "mixed", "other", "value", "index", "general", "consumer", "unit",
        "currency", "usd", "lcu", "slc", "standard", "rate", "exchange",
    }
)
_ALIASES = {"corn": "maize", "peanut": "groundnut", "irish": "potato", "palmoil": "palm"}
_SOURCE_WORDS = {"wfp": "wfp", "fews": "fews", "faostat": "faostat", "fao": "faostat"}
_TYPE_WORDS = (
    (re.compile(r"\bwholesale\b", re.IGNORECASE), "Wholesale"),
    (re.compile(r"\bretail\b", re.IGNORECASE), "Retail"),
    (re.compile(r"\b(producer|farm[- ]?gate)\b", re.IGNORECASE), "Producer"),
)


@dataclass(frozen=True)
class PriceSeries:
    product_key: str
    product_name: str
    country_iso3: str
    price_types: list[str]
    price_sources: list[str]
    metrics: list[str]
    units: list[str]
    currencies: list[str]
    first_date: str
    last_date: str
    market_count: int


@dataclass(frozen=True)
class PriceMarket:
    country_iso3: str
    admin_1: str
    market_name: str
    market_keys: list[str]
    product_keys: list[str]
    last_date: str


@dataclass
class PriceVocab:
    series: list[PriceSeries] = field(default_factory=list)
    markets: list[PriceMarket] = field(default_factory=list)
    built_at: str = ""

    def for_country(self, iso3: str | None) -> list[PriceSeries]:
        if not iso3:
            return list(self.series)
        code = iso3.upper()
        return [s for s in self.series if s.country_iso3 == code]

    def markets_in(self, iso3: str) -> list[PriceMarket]:
        code = iso3.upper()
        return [m for m in self.markets if m.country_iso3 == code]


@dataclass(frozen=True)
class PricePlace:
    """A sub-national place named in the question: one market or one admin-1 region."""

    level: str
    name: str
    admin_1: str
    market_keys: tuple[str, ...]
    product_keys: frozenset[str]


@dataclass(frozen=True)
class ProductResolution:
    """Outcome of matching question text to price products."""

    named: bool
    series: list[PriceSeries]
    closest: list[str]

    @property
    def product_keys(self) -> list[str]:
        return sorted({s.product_key for s in self.series})

    @property
    def product_names(self) -> list[str]:
        return sorted({s.product_name for s in self.series})


def _fold(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in norm if not unicodedata.combining(ch)).lower()


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("oes"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokens(text: str) -> set[str]:
    out = set()
    for raw in re.findall(r"[a-z0-9]+", _fold(text)):
        tok = _stem(raw)
        out.add(_ALIASES.get(tok, tok))
    return out


def _content(toks: set[str]) -> set[str]:
    return {t for t in toks if t not in _GENERIC and not t.isdigit() and len(t) > 2}


@lru_cache(maxsize=1)
def _product_tokens() -> dict[str, frozenset[str]]:
    return {s.product_name: frozenset(tokens(s.product_name)) for s in load_price_vocab().series}


@lru_cache(maxsize=1)
def _product_vocabulary_words() -> frozenset[str]:
    words: set[str] = set()
    for toks in _product_tokens().values():
        words |= _content(set(toks))
    return frozenset(words)


def names_price_product(text: str) -> bool:
    """True when the text names a product the price marts hold."""
    return bool(_content(tokens(text)) & _product_vocabulary_words())


def resolve_price_products(text: str, iso3: str | None = None) -> ProductResolution:
    """Match question text to price series: every product sharing the best set of content words.

    "maize" selects the whole maize family; "maize flour" narrows to flour; "palm oil"
    picks Oil (palm) over Oil (vegetable). Only series held for ``iso3`` are kept.
    """
    q_content = _content(tokens(text)) & _product_vocabulary_words()
    if not q_content:
        return ProductResolution(named=bool(CROP_ENTITY_RE.search(text or "")), series=[], closest=[])
    product_tokens = _product_tokens()

    def score(name: str) -> int:
        return len(q_content & product_tokens.get(name, frozenset()))

    global_best = max((score(n) for n in product_tokens), default=0)
    pool = load_price_vocab().for_country(iso3)
    local = {s.product_name: score(s.product_name) for s in pool}
    local_best = max(local.values(), default=0)
    if local_best and local_best == global_best:
        names = {n for n, sc in local.items() if sc == local_best}
        return ProductResolution(named=True, series=[s for s in pool if s.product_name in names], closest=[])
    closest = sorted((n for n, sc in local.items() if sc), key=lambda n: (-local[n], n))
    return ProductResolution(named=True, series=[], closest=closest[:3])


_PLACE_STOPWORDS = frozenset({"national", "average", "market", "markets", "town", "city", "central", "capital"})


def _place_key(name: str) -> str:
    base = re.sub(r"\(.*?\)", " ", _fold(name))
    base = re.sub(r"[^a-z0-9]+", " ", base).strip()
    return re.sub(r"\s+market$", "", base)


def _place_named(key: str, folded_query: str) -> bool:
    if len(key) < 4 or key in _PLACE_STOPWORDS or key in _GENERIC or key in _product_vocabulary_words():
        return False
    if name_to_iso3(key):
        return False
    return f" {key} " in folded_query


def resolve_price_place(text: str, iso3: str) -> PricePlace | None:
    """The market or admin-1 region the question names inside ``iso3``, if any.

    A name shared by a region and a market (Kano state, Kano market) resolves to
    the region, which covers the city's markets.
    """
    markets = load_price_vocab().markets_in(iso3)
    if not markets:
        return None
    folded = " " + re.sub(r"[^a-z0-9]+", " ", _fold(text)) + " "
    regions: dict[str, list[PriceMarket]] = {}
    for m in markets:
        if m.admin_1:
            regions.setdefault(m.admin_1, []).append(m)
    region_hits = sorted((a for a in regions if _place_named(_place_key(a), folded)), key=len, reverse=True)
    if region_hits:
        name = region_hits[0]
        group = regions[name]
        return PricePlace(
            level="admin1",
            name=name,
            admin_1=name,
            market_keys=tuple(sorted({k for m in group for k in m.market_keys})),
            product_keys=frozenset(k for m in group for k in m.product_keys),
        )
    market_hits = sorted(
        (m for m in markets if _place_named(_place_key(m.market_name), folded)),
        key=lambda m: len(m.market_name),
        reverse=True,
    )
    if not market_hits:
        return None
    hit = market_hits[0]
    same = [m for m in markets if _place_key(m.market_name) == _place_key(hit.market_name)]
    return PricePlace(
        level="market",
        name=hit.market_name,
        admin_1=hit.admin_1,
        market_keys=tuple(sorted({k for m in same for k in m.market_keys})),
        product_keys=frozenset(k for m in same for k in m.product_keys),
    )


def region_place(iso3: str, admin_1: str) -> PricePlace | None:
    """The admin-1 region as a place, for widening a market that lacks the product."""
    group = [m for m in load_price_vocab().markets_in(iso3) if m.admin_1 == admin_1]
    if not admin_1 or not group:
        return None
    return PricePlace(
        level="admin1",
        name=admin_1,
        admin_1=admin_1,
        market_keys=tuple(sorted({k for m in group for k in m.market_keys})),
        product_keys=frozenset(k for m in group for k in m.product_keys),
    )


def choose_price_source(text: str, series: list[PriceSeries]) -> str | None:
    """One price_source per answer, from the sources the matched series actually hold."""
    held = {src for s in series for src in s.price_sources}
    if not held:
        return None
    for word, src in _SOURCE_WORDS.items():
        if re.search(rf"\b{word}\b", text or "", re.IGNORECASE) and src in held:
            return src
    if re.search(r"\b(producer|farm[- ]?gate)\b", text or "", re.IGNORECASE) and "faostat" in held:
        return "faostat"
    for src in ("wfp", "fews", "faostat"):
        if src in held:
            return src
    return sorted(held)[0]


_DEFAULT_TYPES = ("Retail", "Wholesale", "Producer Price (USD/tonne)", "Producer Price (LCU/tonne)", "Producer")


def choose_price_type(text: str, series: list[PriceSeries], source: str | None) -> str | None:
    """Exact price_type literal the series hold. Price SQL always pins one.

    An explicit retail/wholesale/producer mention wins (None when not held);
    otherwise Retail, then Wholesale, then producer prices in USD.
    """
    held = {t for s in series if not source or source in s.price_sources for t in s.price_types}
    for pattern, wanted in _TYPE_WORDS:
        if pattern.search(text or ""):
            if wanted in held:
                return wanted
            starts = sorted(t for t in held if t.startswith(wanted))
            return starts[0] if starts else None
    for wanted in _DEFAULT_TYPES:
        if wanted in held:
            return wanted
    return sorted(held)[0] if held else None


_VOCAB_COLUMNS = {"price_type": "price_types", "price_source": "price_sources", "metric": "metrics"}


@lru_cache(maxsize=8)
def price_vocab_values(column: str) -> frozenset[str]:
    """Every value the price mart holds for a discriminator column (empty when not built)."""
    attr = _VOCAB_COLUMNS.get(column.lower())
    if not attr:
        return frozenset()
    return frozenset(v for s in load_price_vocab().series for v in getattr(s, attr))


def price_sql_vocab_errors(sql: str) -> list[str]:
    """Literals in price SQL that the mart does not hold, and missing unit/currency labels."""
    errors: list[str] = []
    table = "fct_prices" if "fct_prices" in sql else "agg_prices_country_month"
    samples = value_samples_for_table(table)
    for col in ("price_type", "price_source", "metric"):
        allowed = set(price_vocab_values(col)) or set(samples.get(col) or [])
        if not allowed:
            continue
        for lit in re.findall(rf"\b{col}\s*=\s*'([^']*)'", sql):
            if lit not in allowed:
                errors.append(f"{col}='{lit}' not in {table} values")
        for group in re.findall(rf"\b{col}\s+IN\s*\(([^)]*)\)", sql, re.IGNORECASE):
            for lit in re.findall(r"'([^']*)'", group):
                if lit not in allowed:
                    errors.append(f"{col} IN '{lit}' not in {table} values")
    keys = {s.product_key for s in load_price_vocab().series}
    if keys:
        for group in re.findall(r"\bproduct_key\s+IN\s*\(([^)]*)\)", sql, re.IGNORECASE):
            for lit in re.findall(r"'([^']*)'", group):
                if lit not in keys:
                    errors.append(f"product_key '{lit}' not in price vocabulary")
    if table == "fct_prices":
        head = sql.split(" FROM ", 1)[0]
        for col in ("unit", "currency"):
            if not re.search(rf"\b{col}\b", head):
                errors.append(f"{col} not selected from fct_prices")
    return errors


_BUILD_SQL = """
SELECT
  p.product_key,
  d.product_name,
  p.country_iso3,
  ARRAY_AGG(DISTINCT p.price_type IGNORE NULLS) AS price_types,
  ARRAY_AGG(DISTINCT p.price_source IGNORE NULLS) AS price_sources,
  ARRAY_AGG(DISTINCT p.metric IGNORE NULLS) AS metrics,
  ARRAY_AGG(DISTINCT p.unit IGNORE NULLS) AS units,
  ARRAY_AGG(DISTINCT p.currency IGNORE NULLS) AS currencies,
  MIN(p.as_of_date) AS first_date,
  MAX(p.as_of_date) AS last_date,
  COUNT(DISTINCT p.market_key) AS market_count
FROM `{project}.{dataset}.fct_prices` p
JOIN `{project}.{dataset}.dim_product` d ON d.product_key = p.product_key
WHERE p.product_key IS NOT NULL AND p.country_iso3 IS NOT NULL AND d.product_name IS NOT NULL
  AND {price_level}
GROUP BY 1, 2, 3
"""

_BUILD_MARKETS_SQL = """
SELECT
  p.country_iso3,
  IFNULL(TRIM(m.admin_1), '') AS admin_1,
  TRIM(m.market_name) AS market_name,
  ARRAY_AGG(DISTINCT p.market_key) AS market_keys,
  ARRAY_AGG(DISTINCT p.product_key IGNORE NULLS) AS product_keys,
  MAX(p.as_of_date) AS last_date
FROM `{project}.{dataset}.fct_prices` p
JOIN `{project}.{dataset}.dim_market` m ON m.market_key = p.market_key
WHERE p.country_iso3 IS NOT NULL AND m.market_name IS NOT NULL AND TRIM(m.market_name) != ''
  AND {price_level}
GROUP BY 1, 2, 3
"""

# Index series (e.g. FAOSTAT "Producer Price Index (2014-2016 = 100)") are not price
# levels; letting them in would stretch series dates past the last real price.
_PRICE_LEVEL = "NOT REGEXP_CONTAINS(LOWER(IFNULL(p.price_type, '')), r'\\bindex\\b')"


def build_price_vocab(client: bigquery.Client) -> PriceVocab:
    project = os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4").strip()
    fmt = {"project": project, "dataset": mart_dataset(), "price_level": _PRICE_LEVEL}
    sql = _BUILD_SQL.format(**fmt)
    market_rows = client.query(_BUILD_MARKETS_SQL.format(**fmt)).result(timeout=300)
    markets = [
        PriceMarket(
            country_iso3=str(r["country_iso3"]).upper(),
            admin_1=str(r["admin_1"] or ""),
            market_name=str(r["market_name"]),
            market_keys=sorted(str(x) for x in r["market_keys"] or []),
            product_keys=sorted(str(x) for x in r["product_keys"] or []),
            last_date=str(r["last_date"] or ""),
        )
        for r in market_rows
    ]
    rows = client.query(sql).result(timeout=300)
    series = [
        PriceSeries(
            product_key=str(r["product_key"]),
            product_name=str(r["product_name"]).strip(),
            country_iso3=str(r["country_iso3"]).upper(),
            price_types=sorted(str(x) for x in r["price_types"] or []),
            price_sources=sorted(str(x) for x in r["price_sources"] or []),
            metrics=sorted(str(x) for x in r["metrics"] or []),
            units=sorted(str(x) for x in r["units"] or [] if str(x).strip()),
            currencies=sorted(str(x) for x in r["currencies"] or [] if str(x).strip()),
            first_date=str(r["first_date"] or ""),
            last_date=str(r["last_date"] or ""),
            market_count=int(r["market_count"] or 0),
        )
        for r in rows
    ]
    return PriceVocab(series=series, markets=markets, built_at=datetime.now(UTC).isoformat())


def write_price_vocab(vocab: PriceVocab, path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "built_at": vocab.built_at,
        "series": [asdict(s) for s in vocab.series],
        "markets": [asdict(m) for m in vocab.markets],
    }
    path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    load_price_vocab.cache_clear()
    price_vocab_values.cache_clear()
    _product_tokens.cache_clear()
    _product_vocabulary_words.cache_clear()
    return path


@lru_cache(maxsize=1)
def load_price_vocab(path: Path = ARTIFACT) -> PriceVocab:
    if not path.is_file():
        return PriceVocab()
    raw = json.loads(path.read_text(encoding="utf-8"))
    series = [PriceSeries(**item) for item in raw.get("series") or []]
    markets = [PriceMarket(**item) for item in raw.get("markets") or []]
    return PriceVocab(series=series, markets=markets, built_at=str(raw.get("built_at") or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the price vocabulary artifact from live BigQuery")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return 1
    load_rag_dotenv(Path(__file__).resolve().parents[3])
    client = bigquery.Client(project=os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4"))
    vocab = build_price_vocab(client)
    path = write_price_vocab(vocab)
    products = len({s.product_key for s in vocab.series})
    countries = len({s.country_iso3 for s in vocab.series})
    regions = len({(m.country_iso3, m.admin_1) for m in vocab.markets if m.admin_1})
    print(
        f"{len(vocab.series)} series, {products} products, {countries} countries, "
        f"{len(vocab.markets)} markets, {regions} admin-1 regions -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "PriceMarket",
    "PricePlace",
    "PriceSeries",
    "PriceVocab",
    "ProductResolution",
    "build_price_vocab",
    "choose_price_source",
    "choose_price_type",
    "load_price_vocab",
    "price_sql_vocab_errors",
    "price_vocab_values",
    "region_place",
    "resolve_price_place",
    "resolve_price_products",
    "tokens",
    "write_price_vocab",
]
