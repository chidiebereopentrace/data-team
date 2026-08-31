"""
BigQuery retriever: natural-language questions → BigQuery SQL over mart_dev only.
Uses an LLM for NL-to-SQL; validates and runs only SELECTs against BQ_DATASET_GOLD.
"""
from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from ml.rag.chatbot.acf_metadata import project_bq_row_acf
from ml.rag.chatbot.bq_byte_budget import hint_max_bytes, truncate_utf8
from ml.rag.chatbot.bq_sql_patterns import try_sql_patterns
from ml.rag.chatbot.bq_sql_templates import try_sql_template
from ml.rag.chatbot.bq_sql_validate import (
    broaden_empty_sql_once,
    dry_run_sql,
    inject_missing_metric_filters,
    inject_time_bounds,
    max_bytes_billed_for_source,
    sql_retry_enabled,
    validate_dry_run_bytes,
    validate_required_metric_filters,
    validate_semantic_coherence,
    validate_sql_column_allowlist,
    validate_sql_table_allowlist,
    validate_sql_value_samples,
)
from ml.rag.chatbot.query_decomposer import _NON_COUNTRY_GEO
from ml.rag.chatbot.bq_table_schema_yaml import join_fragments_for_tables
from ml.rag.llm_chat import llm_chat_complete, llm_default_timeout_s, llm_model_id
from ml.rag.local_env import load_rag_dotenv
from ml.rag.observability import (
    get_observe_decorator,
    observed_span,
    run_with_tracing_context,
    sql_hash,
    trace_elapsed_ms,
    update_current_span_metadata,
)
from ml.rag.retrievers.base import BaseRetriever
from ml.rag.session_store import get_bq_schema_cache, set_bq_schema_cache

logger = logging.getLogger(__name__)

_observe_span = get_observe_decorator()

_REPO_ROOT = Path(__file__).resolve().parents[3]


_MART_TABLE_PREFIXES = ("fct_", "agg_", "dim_", "bridge_")


def _is_mart_table_id(table_id: str) -> bool:
    return (table_id or "").strip().lower().startswith(_MART_TABLE_PREFIXES)


def _load_dotenv() -> None:
    """Delegate to the single RAG env loader. Safe to call multiple times."""
    try:
        load_rag_dotenv(_REPO_ROOT)
    except Exception:
        pass


def _get_datasets_config() -> dict[str, str]:
    """Mart (gold) dataset ID from env — sole NL-to-SQL target."""
    return {
        "mart": os.environ.get("BQ_DATASET_GOLD", "mart_dev").strip(),
    }


def _nl2sql_model_id() -> str:
    """Dedicated NL2SQL model when set; otherwise the global chat model."""
    return (os.environ.get("RAG_BQ_NL2SQL_MODEL_ID") or "").strip() or llm_model_id()


def _nl2sql_call_timeout_s() -> float:
    """
    Soft per-call budget (seconds) for a single NL2SQL generation inside a parallel
    batch. Distinct from RAG_BQ_NL2SQL_TIMEOUT_S (the hard HTTP client timeout / last
    resort safety net, default 300s) — this value bounds how long the *batch* waits
    for any one table-hint call before moving on without it, so one unusually slow
    reasoning-model call cannot drag the whole request past this ceiling.
    """
    try:
        return max(2.0, float(os.environ.get("RAG_BQ_NL2SQL_CALL_TIMEOUT_S", "20") or 20))
    except ValueError:
        return 20.0


def _call_llama_for_sql(messages: list[dict[str, str]], *, max_tokens: int | None = None) -> str:
    """Call LLM for NL-to-SQL; return raw text (expected to be SQL) or empty string."""
    cap = max_tokens or int(os.environ.get("RAG_BQ_NL2SQL_MAX_TOKENS", "1024") or 1024)
    bq_timeout = float(os.environ.get("RAG_BQ_NL2SQL_TIMEOUT_S", "0") or 0) or llm_default_timeout_s()
    return llm_chat_complete(
        messages,
        model=_nl2sql_model_id(),
        max_tokens=cap,
        temperature=0.0,
        timeout_s=bq_timeout,
        purpose="bq.nl2sql",
    )


# Filter guidance: YAML Columns + *_value_samples are authoritative.
_SCHEMA_FILTER_GUIDE = """
Filter columns by question intent — use ONLY exact names from each table's Columns block
in the table hints (never invent bronze/raw names like area or item when Columns lists
country_name / product_name):
- Geography: country_iso3 on facts; join dim_geography for country names
- Time: year, month, as_of_date, date_key when listed
- Product: product_key + join dim_product, or product_name when denormalized
- Metric discriminators: metric, production_grain, price_source, price_type, measure_type, trade_grain, climate_grain, input_grain — filter with exact sample strings
- Prefer SELECT grain dims + value + unit + warehouse ACF contract columns over bare SELECT *
- On fct_* facts include when listed in Columns: tier, data_level, geo_scope, place_scope, metric, source_id (or source_key), as_of_date, as_of_date_basis

Query patterns:
- Time-bounded questions -> WHERE on as_of_date or year from Columns
- Rankings / comparisons -> GROUP BY country_iso3
- JOIN dim_geography / dim_product / dim_indicator when filtering by names

Mart tables (use only tables in the Schema / table hints — examples):
- Production: fct_production, agg_production_annual
- Yield: fct_yield
- Food security / IPC: fct_food_security, agg_food_security_monthly
- Prices: fct_prices, agg_prices_country_month
- Trade: fct_trade
- Climate: fct_climate
- Employment/HDI: fct_employment, fct_hdi, fct_economics
"""

# Bare identifier `country` → `country_iso3` when SQL targets mart production facts.
_BARE_COUNTRY_IDENT_RE = re.compile(r"(?<![A-Za-z0-9_])country(?![A-Za-z0-9_])", re.IGNORECASE)
_MART_PRODUCTION_IN_SQL_RE = re.compile(r"fct_production|agg_production", re.IGNORECASE)

# Forbidden SQL tokens (case-insensitive) for safety
_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|MERGE|TRUNCATE|ALTER|GRANT|REVOKE|EXEC|EXECUTE|CALL)\b",
    re.IGNORECASE,
)

_QUERY_SPLIT_RE = re.compile(r"\n---+\s*(?:QUERY)?\s*---+\n", re.IGNORECASE)


def _year_int_from_bound(value: str | None) -> int | None:
    """Extract a calendar year from an ISO date or bare year string."""
    raw = str(value or "").strip()
    if not raw:
        return None
    match = re.search(r"\b((?:19|20)\d{2})\b", raw)
    if not match:
        return None
    year = int(match.group(1))
    if 1900 <= year <= 2100:
        return year
    return None


def _continental_scope_hint(
    query: str | None,
    entities: list[str] | None,
) -> str | None:
    """Detect Africa/subregion scope for BQ filtering (not a single country)."""
    found: list[str] = []
    q = (query or "").lower()
    for region in _NON_COUNTRY_GEO:
        if region in q and region not in found:
            found.append(region)
    if entities:
        for ent in entities:
            el = str(ent).strip().lower()
            if el in _NON_COUNTRY_GEO and el not in found:
                found.append(el)
    if not found:
        return None
    return (
        "- CONTINENTAL/REGIONAL scope (Africa or subregion): rank or aggregate across "
        "African countries using country_name (or equivalent) on the fact table only; "
        "do NOT filter country_name = 'Africa' or a region label such as 'West Africa' "
        "or 'Sahel'; do NOT invent dim_geography / dim_* "
        "or any country-list subquery; do NOT join GDP/HDI tables for a country list "
        "unless that table is in the selected table set"
    )


def _format_query_constraints(
    *,
    geo_country: str | None,
    geo_countries: list[str] | None = None,
    time_start: str | None,
    time_end: str | None,
    entities: list[str] | None,
    domains: list[str] | None,
    query: str | None = None,
) -> str:
    """Structured filters from query decomposition (must appear in generated SQL)."""
    lines: list[str] = []
    countries = [str(c).strip() for c in (geo_countries or []) if str(c).strip()]
    if not countries and geo_country:
        countries = [geo_country.strip()]
    if len(countries) >= 2:
        lines.append(
            f"- REQUIRED: include rows for ALL of these countries {countries!r} "
            "(filter using the geography column from the Columns block — typically "
            "country_name; use country / geographic_unit_name only if listed there; "
            "GROUP BY that geography column when comparing)"
        )
    elif len(countries) == 1:
        lines.append(
            f"- REQUIRED country/area filter: {countries[0]!r} "
            "(use the geography column from the Columns block — typically country_name)"
        )
    if time_start or time_end:
        y_start = _year_int_from_bound(time_start)
        y_end = _year_int_from_bound(time_end)
        if y_start is not None or y_end is not None:
            start_y = y_start if y_start is not None else y_end
            end_y = y_end if y_end is not None else y_start
            lines.append(
                f"- REQUIRED year filter: year BETWEEN {start_y} AND {end_y} "
                "(use INT64 year / planting_year / harvest_year / observation_year from Columns; "
                "never compare year to ISO date strings — that is integer arithmetic)"
            )
        else:
            lines.append(
                f"- REQUIRED time range: start={time_start or 'any'}, end={time_end or 'any'} "
                "(use the time column from Columns: year, planting_year, harvest_year, "
                "observation_year, etc.)"
            )
    if entities:
        ent = [str(e).strip() for e in entities if str(e).strip()]
        if ent:
            lines.append(f"- Key entities to cover in filters or SELECT: {', '.join(ent)}")
    if domains:
        dom = [str(d).strip() for d in domains if str(d).strip()]
        if dom:
            lines.append(f"- Topic domains: {', '.join(dom)}")
    # Expanded country lists already encode regional scope; the continental hint
    # forbids IN-lists and would push the model toward country_name = 'West Africa'.
    if len(countries) < 2:
        continental = _continental_scope_hint(query, entities)
        if continental:
            lines.append(continental)
    if not lines:
        return ""
    return "Query constraints from decomposition (MUST honor in WHERE / GROUP BY):\n" + "\n".join(lines)


def _extract_single_select(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    # Strip Llama chat template tokens that may appear in NL2SQL output
    text = re.sub(r"\[/?INST\]", "", text, flags=re.IGNORECASE).strip()
    if "```" in text:
        for block in re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", text):
            cleaned = block.strip().rstrip(";")
            upper = cleaned.upper()
            if upper.startswith("SELECT") or upper.startswith("WITH"):
                return cleaned
        text = re.sub(r"```[\s\S]*?```", "", text).strip()
    # Accept a leading SELECT/WITH or the first SELECT/WITH embedded after noise
    upper = text.upper()
    if upper.startswith("SELECT") or upper.startswith("WITH"):
        return text.rstrip(";")
    match = re.search(r"((?:WITH|SELECT)\b[\s\S]*)", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().rstrip(";")
        cupper = candidate.upper()
        if cupper.startswith("SELECT") or cupper.startswith("WITH"):
            return candidate
    return ""


def _parse_sql_queries(raw: str, max_queries: int) -> list[str]:
    """Parse up to max_queries SELECT statements from one LLM response."""
    if not raw or max_queries < 1:
        return []
    chunks = _QUERY_SPLIT_RE.split(raw)
    if len(chunks) <= 1:
        chunks = re.split(r"\n(?=(?:WITH|SELECT)\s)", raw, flags=re.IGNORECASE)
    seen: set[str] = set()
    out: list[str] = []
    for chunk in chunks:
        sql = _extract_single_select(chunk)
        cupper = sql.upper()
        if not (cupper.startswith("SELECT") or cupper.startswith("WITH")):
            continue
        norm = " ".join(sql.split())
        if norm in seen:
            continue
        seen.add(norm)
        out.append(sql)
        if len(out) >= max_queries:
            break
    return out


def _rewrite_faostat_country_ident(sql: str) -> str:
    """Map bare ``country`` → ``country_iso3`` when SQL references mart production tables."""
    if not sql or not _MART_PRODUCTION_IN_SQL_RE.search(sql):
        return sql
    return _BARE_COUNTRY_IDENT_RE.sub("country_iso3", sql)


def _validate_sql(sql: str, allowed_dataset_ids: set[str], default_limit: int) -> str | None:
    """
    Ensure SQL is a safe SELECT-only query over allowed datasets. Returns cleaned SQL or None.
    Accepts WITH … SELECT CTEs (read-only pattern builders).
    """
    rewritten = _rewrite_faostat_country_ident(sql)
    normalized = " ".join(rewritten.split()).strip()
    upper = normalized.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return None
    if upper.startswith("WITH") and " SELECT " not in f" {upper} ":
        return None
    if _FORBIDDEN_SQL.search(normalized):
        return None
    # Ensure referenced datasets are in the allowed set (mart_dev only for RAG).
    # Only inspect backtick-quoted FQNs so aliases like curr.year are ignored.
    if allowed_dataset_ids:
        allowed_lower = {a.lower() for a in allowed_dataset_ids}
        for part in re.findall(r"`([^`]+)`", normalized):
            if "." in part:
                segments = part.split(".")
                ds = segments[-2].lower()
                if ds not in allowed_lower:
                    return None
    if "LIMIT" not in normalized.upper():
        normalized = f"{normalized.rstrip(';')} LIMIT {default_limit}"
    return normalized


def _bq_diagnostic_item(
    *,
    status: str,
    message: str,
    sql: str = "",
    prep_error: str | None = None,
    nl2sql_raw: str | None = None,
) -> dict[str, Any]:
    """Inspector-visible BQ failure row (filtered out of generation context)."""
    meta: dict[str, Any] = {
        "sql": sql,
        "status": status,
        "validation_failed": True,
    }
    if prep_error:
        meta["prep_error"] = prep_error[:500]
    if nl2sql_raw:
        meta["nl2sql_raw"] = nl2sql_raw[:500]
    return {
        "content": f"[BQ {status}: {message[:200]}]",
        "source": "bigquery",
        "metadata": meta,
    }


class BQRetriever(BaseRetriever):
    """
    Retrieve context by querying BigQuery. Uses mart_dev only (BQ_DATASET_GOLD).
    Uses an LLM for NL-to-SQL when no explicit sql is provided.
    """

    def __init__(
        self,
        project_id: str | None = None,
        max_rows: int = 100,
        nl2sql_enabled: bool | None = None,
    ):
        _load_dotenv()
        if project_id is not None:
            self.project_id = project_id.strip()
        else:
            self.project_id = (os.environ.get("BQ_PROJECT", "") or "").strip()
        self.datasets_config = _get_datasets_config()
        self.max_rows = max_rows
        if nl2sql_enabled is not None:
            self.nl2sql_enabled = nl2sql_enabled
        else:
            self.nl2sql_enabled = os.environ.get("RAG_BQ_NL2SQL_ENABLED", "1").strip().lower() in ("1", "true", "on")
        self._client = None
        self._last_nl2sql_raws: list[str] = []
        self.last_sql_source: str | None = None
        self.last_bq_execute_ms: float | None = None
        self.last_bq_nl2sql_ms: float | None = None

    def _record_nl2sql_raw(self, raw: str | None, *, parsed_ok: bool) -> None:
        """Keep short snippets of failed/empty NL2SQL generations for inspector debug."""
        if parsed_ok:
            return
        text = (raw or "").strip()
        note = text[:500] if text else "(empty LLM response)"
        if note not in self._last_nl2sql_raws:
            self._last_nl2sql_raws.append(note)
        # Cap memory for parallel multi-hint runs.
        if len(self._last_nl2sql_raws) > 5:
            self._last_nl2sql_raws = self._last_nl2sql_raws[-5:]

    def _get_client(self):
        if self._client is None:
            from google.cloud import bigquery
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    def _schema_cache_key(self) -> str:
        """Stable key for the shared (Redis or fallback) schema cache."""
        parts = [self.project_id or "no-project"]
        for k, v in sorted(self.datasets_config.items()):
            parts.append(f"{k}={v or ''}")
        return ":".join(parts)

    def _get_schema(self) -> str:
        """Build a compact schema summary for mart_dev; cached via Redis when configured."""
        cache_key = self._schema_cache_key()
        cached = get_bq_schema_cache(cache_key)
        if cached is not None:
            return cached

        schema_text: str
        try:
            from google.cloud.bigquery import DatasetReference, TableReference
            client = self._get_client()
            lines = []
            for layer, dataset_id in self.datasets_config.items():
                if not dataset_id:
                    continue
                try:
                    dataset_ref = DatasetReference(self.project_id, dataset_id)
                    table_list = list(client.list_tables(dataset_ref))
                except Exception:
                    lines.append(f"[{layer}] dataset = {dataset_id}. (tables not listed)")
                    continue
                table_desc = []
                for t in table_list[:50]:  # cap tables per dataset
                    try:
                        table_ref = TableReference(dataset_ref, t.table_id)
                        table = client.get_table(table_ref)
                        cols = ", ".join(f"{f.name} {f.field_type}" for f in (table.schema or [])[:20])
                        table_desc.append(f"{t.table_id} ({cols})")
                    except Exception:
                        table_desc.append(t.table_id)
                lines.append(f"[{layer}] dataset = {dataset_id}. Tables: " + "; ".join(table_desc))
            schema_text = "\n".join(lines) if lines else "[No schema]"
        except Exception:
            schema_text = "[Schema unavailable]"

        set_bq_schema_cache(cache_key, schema_text)
        return schema_text

    def _schema_for_nl2sql(self, table_hints: list[str] | None) -> str:
        """Compact schema text; skip live BQ catalog when rich per-table YAML hints are present."""
        skip_live = os.environ.get("RAG_BQ_SKIP_LIVE_SCHEMA", "on").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        )
        if skip_live and table_hints:
            ds = self.datasets_config.get("mart", "").strip()
            return (
                f"Project: `{self.project_id}`. Mart dataset: `{ds}`. "
                "Use only the fct_* / agg_* / dim_* tables described in the table hints below."
            )
        return self._get_schema()

    def _build_nl2sql_messages(
        self,
        question: str,
        table_hints: list[str] | None,
        *,
        geo_country: str | None,
        geo_countries: list[str] | None = None,
        time_start: str | None,
        time_end: str | None,
        entities: list[str] | None,
        domains: list[str] | None,
        multi_query: bool,
        max_queries: int,
        selected_tables: list[str] | None = None,
        query: str | None = None,
    ) -> list[dict[str, str]]:
        schema_text = self._schema_for_nl2sql(table_hints)
        constraints_block = _format_query_constraints(
            geo_country=geo_country,
            geo_countries=geo_countries,
            time_start=time_start,
            time_end=time_end,
            entities=entities,
            domains=domains,
            query=query or question,
        )
        hints_block = ""
        hints_truncated = False
        if table_hints:
            cleaned = [str(h).strip() for h in table_hints if str(h).strip()]
            if cleaned:
                total_budget = hint_max_bytes()
                per_hint_cap = max(400, total_budget // max(1, len(cleaned)))
                packed: list[str] = []
                used = 0
                for h in cleaned:
                    frag, was_cut = truncate_utf8(h, per_hint_cap)
                    cost = len(frag.encode("utf-8"))
                    if used + cost > total_budget:
                        remain = total_budget - used
                        if remain > 64:
                            frag, _ = truncate_utf8(h, remain)
                            packed.append(f"- {frag}")
                            hints_truncated = True
                        else:
                            hints_truncated = True
                        break
                    packed.append(f"- {frag}")
                    used += cost
                    hints_truncated = hints_truncated or was_cut
                hints_block = (
                    "\n\nSelected mart_dev table schemas from YAML "
                    "(use only these tables; honor sql_generation_hints and filtering_guidance):\n"
                    + "\n".join(packed)
                )
                if hints_truncated:
                    hints_block += "\n[bq_hint_truncated=true]"
        if multi_query:
            output_rule = (
                f"9) Output up to {max_queries} separate BigQuery SELECT queries when needed. "
                f"Put each query on its own block separated by a line containing only ---QUERY---. "
                "No explanation, no markdown fences."
            )
        else:
            output_rule = (
                "9) Output exactly one SELECT. No explanation, no markdown, no code fence."
            )
        ds = self.datasets_config.get("mart", "mart_dev")
        allowed_tables = [
            str(t).strip().split(".")[-1]
            for t in (selected_tables or [])
            if str(t).strip()
        ]
        tables_rule = (
            f"Tables you may use: {', '.join(allowed_tables)}. "
            "JOINs are allowed between these tables and documented dim_* targets using on= keys "
            "from semantic_relationships / JOIN fragments. Never reference tables outside this list."
            if allowed_tables
            else (
                "Use ONLY mart tables and columns from the Schema / table hints. "
                "JOIN dim_geography, dim_product, dim_indicator when filtering by names."
            )
        )
        join_fragments = join_fragments_for_tables(allowed_tables)
        join_rule = (
            "8) When JOIN fragments are listed, use those ON predicates exactly; "
            "if fragments say to run separate SELECTs, do not invent joins. "
            if join_fragments
            else "8) Prefer single-table SELECTs unless documented joins are present. "
        )
        system = (
            f"You are a BigQuery expert for OpenTrace agricultural data in the `{ds}` dataset only. "
            "Rules: "
            f"1) {tables_rule} "
            f"Use full names: `{self.project_id}.{ds}.fct_*` / `agg_*` / `dim_*`. "
            "2) Use exact column names from the Columns blocks only — prefer country_iso3 on facts; "
            "join dim_geography for country names when needed. "
            "3) For every table hint, equality-filter metric-discriminator columns that have "
            "*_value_samples (metric, production_grain, price_source, price_type, measure_type, "
            "trade_grain, climate_grain, etc.) using exact sample strings. "
            "4) Prefer SELECT of grain columns + value + unit + warehouse ACF contract columns "
            "(tier, data_level, geo_scope, place_scope, metric, source_id/source_key, as_of_date, "
            "as_of_date_basis when listed) over bare SELECT *. "
            "Never SUM/AVG across mixed discriminator values. "
            "5) When Query constraints are present, REQUIRED country and time filters MUST appear. "
            "6) For country rankings: filter discriminators + year, then "
            "SUM(value) GROUP BY country_iso3 ORDER BY total DESC. "
            "7) country_iso3 values are ISO3 codes — never filter = 'Africa'. "
            f"{join_rule}"
            f"{output_rule}"
        )
        constraints_section = f"\n\n{constraints_block}\n" if constraints_block else ""
        join_section = f"\n\n{join_fragments}\n" if join_fragments else ""
        user = (
            f"Filter and table hints:\n{_SCHEMA_FILTER_GUIDE}"
            f"{constraints_section}"
            f"{hints_block}"
            f"{join_section}\n"
            f"Schema:\n{schema_text}\n\n"
            f"Question: {question}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _nl_to_sql_one(
        self,
        question: str,
        table_hints: list[str] | None = None,
        *,
        geo_country: str | None = None,
        geo_countries: list[str] | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        entities: list[str] | None = None,
        domains: list[str] | None = None,
        selected_tables: list[str] | None = None,
        query: str | None = None,
    ) -> str:
        """Generate one BigQuery SELECT (focused on a single table hint when provided)."""
        messages = self._build_nl2sql_messages(
            question,
            table_hints,
            geo_country=geo_country,
            geo_countries=geo_countries,
            time_start=time_start,
            time_end=time_end,
            entities=entities,
            domains=domains,
            multi_query=False,
            max_queries=1,
            selected_tables=selected_tables,
            query=query,
        )
        raw = _call_llama_for_sql(messages)
        sql = _extract_single_select(raw)
        self._record_nl2sql_raw(raw, parsed_ok=bool(sql))
        if not sql and raw:
            logger.warning("NL-to-SQL: LLM returned non-SELECT text (first 200 chars): %s", raw[:200])
        return sql

    def _prepare_sql(
        self,
        raw_sql: str,
        *,
        question: str,
        table_hints: list[str] | None,
        selected_tables: set[str],
        allowed_datasets: set[str],
        limit: int,
        client: Any,
        geo_country: str | None = None,
        geo_countries: list[str] | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        entities: list[str] | None = None,
        domains: list[str] | None = None,
        query: str | None = None,
        primary_measures: list[str] | None = None,
        geography: list[str] | None = None,
        crop_required: bool | None = None,
        geography_required: bool | None = None,
        sql_source: str = "",
        decomposition: dict[str, Any] | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Validate SQL, enforce table allowlist, dry-run, and optionally retry once.

        Returns (validated_sql, error_message). error_message is set when SQL cannot run.
        """
        validated = _validate_sql(raw_sql, allowed_datasets, limit)
        if validated is None:
            return None, "validation_failed"

        def _post_checks(sql: str) -> str | None:
            allow_err = validate_sql_table_allowlist(sql, selected_tables)
            if allow_err:
                return allow_err
            col_err = validate_sql_column_allowlist(sql, selected_tables or None)
            if col_err:
                return col_err
            metric_err = validate_required_metric_filters(sql, selected_tables or None)
            if metric_err:
                return metric_err
            sample_err = validate_sql_value_samples(sql, selected_tables or None)
            if sample_err:
                return sample_err
            coherence_err = validate_semantic_coherence(
                sql,
                query=question or query or "",
                primary_measures=primary_measures,
                geography=geography,
                time_start=time_start,
                time_end=time_end,
                crop_required=crop_required,
                geography_required=geography_required,
                table_ids=selected_tables or None,
                decomposition=decomposition,
            )
            if coherence_err:
                return coherence_err
            bytes_cap = max_bytes_billed_for_source(sql_source)
            bytes_err = validate_dry_run_bytes(
                client,
                sql,
                max_bytes=bytes_cap,
                sql_source=sql_source,
            )
            if bytes_err:
                return bytes_err
            dry_err = dry_run_sql(client, sql)
            if dry_err:
                return f"dry_run_failed: {dry_err[:300]}"
            return None

        def _maybe_inject(sql: str) -> str:
            cue = question or query or ""
            metric_err = validate_required_metric_filters(sql, selected_tables or None)
            if metric_err:
                fixed, notes = inject_missing_metric_filters(
                    sql,
                    selected_tables or None,
                    query=cue,
                    primary_measures=primary_measures,
                )
                if notes:
                    revalidated = _validate_sql(fixed, allowed_datasets, limit)
                    if revalidated is not None:
                        logger.info("BQ discriminator auto-inject: %s", "; ".join(notes))
                        sql = revalidated
            if time_start or time_end:
                decomp = {"time_start": time_start or "", "time_end": time_end or ""}
                timed, time_notes = inject_time_bounds(
                    sql,
                    decomp,
                    selected_tables or None,
                )
                if time_notes:
                    revalidated = _validate_sql(timed, allowed_datasets, limit)
                    if revalidated is not None:
                        logger.info("BQ time bounds auto-inject: %s", "; ".join(time_notes))
                        sql = revalidated
            return sql

        validated = _maybe_inject(validated)
        check_err = _post_checks(validated)
        if check_err and sql_retry_enabled():
            allowed_list = ", ".join(sorted(selected_tables)) or "(none)"
            retry_question = (
                f"{question}\n\n"
                f"Previous SQL failed validation:\n{check_err}\n\n"
                "Fix the SQL. Use ONLY columns from the Columns blocks in the table hints. "
                "Equality-filter every metric discriminator that has *_value_samples "
                "(element, price_type, measure_type, indicator, treatment, …) using exact sample strings. "
                "Prefer grain columns + value + unit over SELECT *. "
                f"Use ONLY these tables: {allowed_list}. "
                "Never invent dim_geography, dim_*, bronze/raw column names, or any table outside that list. "
                "JOIN only tables in the selected set using documented on= keys. "
                "For Africa/continental rankings stay on the fact table "
                "(no country-list subquery; never filter country_name = 'Africa')."
            )
            retry_sql = self._nl_to_sql_one(
                retry_question,
                table_hints=table_hints,
                geo_country=geo_country,
                geo_countries=geo_countries,
                time_start=time_start,
                time_end=time_end,
                entities=entities,
                domains=domains,
                selected_tables=sorted(selected_tables),
                query=query,
            )
            if retry_sql:
                retry_validated = _validate_sql(retry_sql, allowed_datasets, limit)
                if retry_validated is None:
                    return None, f"retry_validation_failed: {check_err[:200]}"
                validated = _maybe_inject(retry_validated)
                check_err = _post_checks(validated)
        if check_err:
            return None, check_err
        return validated, None

    def _nl_to_sql_queries(
        self,
        question: str,
        table_hints: list[str] | None = None,
        *,
        geo_country: str | None = None,
        geo_countries: list[str] | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        entities: list[str] | None = None,
        domains: list[str] | None = None,
        selected_tables: list[str] | None = None,
        query: str | None = None,
    ) -> list[str]:
        """
        Generate up to RAG_BQ_MAX_SQL_QUERIES (default 10) SELECT statements via NL-to-SQL.

        Mode RAG_BQ_NL2SQL_MODE:
        - per_hint (default): one LLM call per vector-matched table hint (up to max queries).
        - batch: one LLM call returning multiple queries separated by ---QUERY---.
        """
        max_queries = max(1, int(os.environ.get("RAG_BQ_MAX_SQL_QUERIES", "10") or 10))
        cleaned_hints = [str(h).strip() for h in (table_hints or []) if str(h).strip()]
        default_mode = "batch" if len(cleaned_hints) > 1 else "per_hint"
        mode = os.environ.get("RAG_BQ_NL2SQL_MODE", default_mode).strip().lower()
        nl2sql_t0 = time.perf_counter()
        self._last_nl2sql_raws = []

        with observed_span(
            "retrieval.bq.nl2sql",
            input_data={
                "question": question[:200],
                "table_hints_count": len(cleaned_hints),
                "mode": mode,
            },
        ):
            parallel_used = False
            timed_out_hints = 0
            hints_for_calls: list[str | None] = (
                list(cleaned_hints[:max_queries]) if cleaned_hints else [None]
            )
            queries: list[str] = []

            if mode == "batch":
                messages = self._build_nl2sql_messages(
                    question,
                    cleaned_hints[:max_queries],
                    geo_country=geo_country,
                    geo_countries=geo_countries,
                    time_start=time_start,
                    time_end=time_end,
                    entities=entities,
                    domains=domains,
                    multi_query=True,
                    max_queries=max_queries,
                    selected_tables=selected_tables,
                    query=query,
                )
                raw_batch = _call_llama_for_sql(messages)
                parsed = _parse_sql_queries(raw_batch, max_queries)
                self._record_nl2sql_raw(raw_batch, parsed_ok=bool(parsed))
                if parsed:
                    queries = parsed

            if not queries:
                parallel = os.environ.get("RAG_BQ_NL2SQL_PARALLEL", "off").strip().lower() in (
                    "1",
                    "true",
                    "on",
                    "yes",
                )
                workers = max(1, int(os.environ.get("RAG_BQ_NL2SQL_PARALLEL_WORKERS", "4") or 4))

                def _gen_one(hint: str | None) -> str:
                    return self._nl_to_sql_one(
                        question,
                        table_hints=[hint] if hint else cleaned_hints or None,
                        geo_country=geo_country,
                        geo_countries=geo_countries,
                        time_start=time_start,
                        time_end=time_end,
                        entities=entities,
                        domains=domains,
                        selected_tables=selected_tables,
                        query=query,
                    )

                seen: set[str] = set()
                if parallel and len(hints_for_calls) > 1:
                    parallel_used = True
                    call_budget = _nl2sql_call_timeout_s()
                    # Do not block on the whole ThreadPoolExecutor lifecycle waiting for a
                    # slow reasoning call: give the batch a soft per-call budget and move on
                    # without that hint's SQL if it doesn't finish in time. The abandoned
                    # thread keeps running in the background (Python cannot forcibly cancel
                    # it) and its result is simply discarded when it eventually completes —
                    # shutdown(wait=False) below avoids blocking process exit/GC on it.
                    pool = ThreadPoolExecutor(max_workers=min(workers, len(hints_for_calls)))
                    try:
                        futs = {
                            pool.submit(run_with_tracing_context(_gen_one, h)): h
                            for h in hints_for_calls
                        }
                        # Single fixed budget from the moment the batch was submitted —
                        # whatever hasn't finished by then is abandoned (not re-armed per
                        # completion), so one slow hint can no longer drag the batch past
                        # call_budget regardless of how many other hints finish first.
                        done, not_done = wait(
                            futs, timeout=call_budget, return_when=ALL_COMPLETED
                        )
                        timed_out_hints = len(not_done)
                        for fut in done:
                            try:
                                sql = fut.result()
                            except Exception:
                                sql = ""
                            if not sql:
                                continue
                            norm = " ".join(sql.split())
                            if norm in seen:
                                continue
                            seen.add(norm)
                            queries.append(sql)
                    finally:
                        pool.shutdown(wait=False)
                else:
                    for hint in hints_for_calls:
                        sql = _gen_one(hint)
                        if not sql:
                            continue
                        norm = " ".join(sql.split())
                        if norm in seen:
                            continue
                        seen.add(norm)
                        queries.append(sql)
                        if len(queries) >= max_queries:
                            break

            if not queries:
                logger.warning(
                    "NL-to-SQL: 0 queries from %s hint(s) (mode=%s); check RAG_LLM_BASE_URL, "
                    "RAG_LLM_MODEL_ID (must match LM Studio), and timeout logs",
                    len(cleaned_hints),
                    mode,
                )

            sql_hashes = list(
                dict.fromkeys(h for h in (sql_hash(q) for q in queries) if h)
            )
            update_current_span_metadata(
                {
                    "mode": mode,
                    "table_hints_count": len(cleaned_hints),
                    "sql_query_count": len(queries),
                    "parallel": parallel_used,
                    "sql_hashes": sql_hashes[:10],
                    "latency_ms": trace_elapsed_ms(nl2sql_t0),
                    "timed_out_hints": timed_out_hints,
                }
            )
            if timed_out_hints:
                logger.warning(
                    "NL-to-SQL: %d/%d table-hint call(s) exceeded the %.0fs per-call budget "
                    "and were skipped (RAG_BQ_NL2SQL_CALL_TIMEOUT_S)",
                    timed_out_hints,
                    len(hints_for_calls),
                    _nl2sql_call_timeout_s(),
                )
            self.last_bq_nl2sql_ms = trace_elapsed_ms(nl2sql_t0)
            return queries[:max_queries]

    @_observe_span(as_type="span", name="retrieval.bq", capture_input=False, capture_output=False)
    def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Run one or more BQ queries and return rows as context items.

        NL-to-SQL generates up to RAG_BQ_MAX_SQL_QUERIES (default 10) SELECTs from
        reasoner-selected YAML table hints. kwargs["sql"] may be a single string or list.
        Fail closed: if no validated SQL is produced, return diagnostic items (no canned fallback).

        Optional kwargs: geo_country, time_start, time_end, entities, domains, table_hints,
        selected_tables, query_intents. Graph node aggregates SQL into state
        ``bq_sql_queries`` / ``bq_sql_debug``.
        """
        t0 = time.perf_counter()
        self.last_sql_source = None
        self.last_bq_execute_ms = None
        self.last_bq_nl2sql_ms = None
        if not self.project_id:
            update_current_span_metadata({"status": "no_project", "row_count": 0})
            return [
                _bq_diagnostic_item(
                    status="no_project",
                    message="BQ_PROJECT is not set; NL2SQL/execute skipped",
                    prep_error="BQ_PROJECT is not set",
                )
            ]
        client = self._get_client()
        table_hints = kwargs.get("table_hints")
        hint_list: list[str] | None = None
        if isinstance(table_hints, list) and table_hints:
            hint_list = [str(x) for x in table_hints if str(x).strip()]

        geo_country = kwargs.get("geo_country")
        if isinstance(geo_country, str):
            geo_country = geo_country.strip() or None
        else:
            geo_country = None

        raw_geo_countries = kwargs.get("geo_countries")
        geo_countries: list[str] | None = None
        if isinstance(raw_geo_countries, (list, tuple)):
            geo_countries = [str(c).strip() for c in raw_geo_countries if str(c).strip()] or None
        if geo_countries and len(geo_countries) >= 2:
            geo_country = None

        time_start = kwargs.get("time_start")
        if not isinstance(time_start, str) or not time_start.strip():
            time_start = None
        else:
            time_start = time_start.strip()[:10]

        time_end = kwargs.get("time_end")
        if not isinstance(time_end, str) or not time_end.strip():
            time_end = None
        else:
            time_end = time_end.strip()[:10]

        entities = kwargs.get("entities")
        if not isinstance(entities, list):
            entities = None
        domains = kwargs.get("domains")
        if not isinstance(domains, list):
            domains = None

        raw_selected = kwargs.get("selected_tables")
        selected_tables: set[str] = set()
        if isinstance(raw_selected, (list, tuple)):
            for item in raw_selected:
                tid = str(item).strip().split(".")[-1].lower()
                if _is_mart_table_id(tid):
                    selected_tables.add(tid)

        raw_intents = kwargs.get("query_intents")
        query_intents: list[Any] | None = raw_intents if isinstance(raw_intents, list) else None
        crop_required = bool(kwargs.get("crop_required", True))
        geography_required = bool(kwargs.get("geography_required", True))
        raw_pm = kwargs.get("primary_measures")
        primary_measures: list[str] | None = None
        if isinstance(raw_pm, list):
            primary_measures = [str(m).strip() for m in raw_pm if str(m).strip()] or None
        decomp = kwargs.get("decomposition")
        geography: list[str] | None = None
        if isinstance(decomp, dict):
            geo_raw = decomp.get("geography")
            if isinstance(geo_raw, list):
                geography = [str(g).strip() for g in geo_raw if str(g).strip()] or None
        if geography is None and geo_countries:
            geography = list(geo_countries)
        elif geography is None and geo_country:
            geography = [geo_country]
        task_mode = str(kwargs.get("task_mode") or "").strip().lower()
        serve_status = str(kwargs.get("serve_status") or "served").strip().lower()
        contract_sql_only = bool(kwargs.get("contract_sql_only"))
        template_key = str(kwargs.get("template_key") or "").strip()
        heavy_path = bool(kwargs.get("heavy_path"))

        if (
            not heavy_path
            and serve_status in ("unsupported_grain", "unsupported_measure", "unsupported_dimension")
        ):
            update_current_span_metadata(
                {
                    "status": serve_status,
                    "row_count": 0,
                    "sql_source": "contract_fail_closed",
                }
            )
            self.last_sql_source = "contract_fail_closed"
            return [
                _bq_diagnostic_item(
                    status="no_valid_sql",
                    message=f"[BQ contract:{serve_status}] warehouse cell not served",
                    prep_error=serve_status,
                )
            ]

        sql_input = kwargs.get("sql")
        sql_queries: list[str] = []
        explicit_sql = False
        if isinstance(sql_input, str) and sql_input.strip():
            sql_queries = [sql_input.strip()]
            explicit_sql = True
        elif isinstance(sql_input, list):
            sql_queries = [str(s).strip() for s in sql_input if str(s).strip()]
            explicit_sql = bool(sql_queries)

        fast_fact = task_mode in ("fact_lookup", "data_export_only")
        if heavy_path:
            fast_fact = False

        template_meta: dict[str, Any] | None = None
        pattern_meta: dict[str, Any] | None = None
        ds_name = self.datasets_config.get("mart", "mart_dev")
        rows_per_query = max(1, int(os.environ.get("RAG_BQ_ROWS_PER_QUERY", "10") or 10))
        sql_source = "none"
        pattern_sqls: list[str] = []
        nl2sql_sqls: list[str] = []
        leftover_intents: list[Any] = []
        pattern_slot_ids: list[str] = []
        intent_slot_by_index: dict[int, str] = {}
        if isinstance(query_intents, list):
            for idx, intent in enumerate(query_intents):
                if isinstance(intent, dict):
                    sid = str(intent.get("subquestion_id") or "").strip()
                    if sid:
                        intent_slot_by_index[idx] = sid

        def _try_template_sql() -> list[str]:
            nonlocal template_meta, sql_source
            template_limit = 1 if fast_fact else rows_per_query
            hit = try_sql_template(
                query=query,
                project_id=self.project_id,
                dataset=ds_name,
                selected_tables=selected_tables,
                entities=entities,
                time_start=time_start,
                time_end=time_end,
                limit=template_limit,
                geo_country=geo_country,
                geo_countries=geo_countries,
                primary_measures=primary_measures,
                task_mode=task_mode,
                template_key=template_key,
            )
            if not hit:
                return []
            template_meta = hit
            sql_source = "template"
            return [str(hit["sql"])]

        if not sql_queries and not explicit_sql:
            template_sqls = _try_template_sql()
            if template_sqls:
                sql_queries = template_sqls
            else:
                pattern_cap = 1 if fast_fact else None
                pattern_hits = try_sql_patterns(
                    query_intents,
                    project_id=self.project_id,
                    dataset=ds_name,
                    query=query,
                    entities=entities,
                    time_start=time_start,
                    time_end=time_end,
                    selected_tables=selected_tables,
                    limit=rows_per_query,
                    geo_country=geo_country,
                    geo_countries=geo_countries,
                    max_queries=pattern_cap,
                    primary_measures=primary_measures,
                )
                if pattern_hits:
                    pattern_meta = {
                        "hits": pattern_hits,
                        "pattern": pattern_hits[0].get("pattern"),
                    }
                    pattern_sqls = [
                        str(h["sql"]) for h in pattern_hits if str(h.get("sql") or "").strip()
                    ]
                    pattern_slot_ids = []
                    for h in pattern_hits:
                        if not str(h.get("sql") or "").strip():
                            continue
                        ii = h.get("intent_index")
                        sid = intent_slot_by_index.get(ii, "") if isinstance(ii, int) else ""
                        pattern_slot_ids.append(sid)
                    compiled_idx = {h.get("intent_index") for h in pattern_hits}
                    if isinstance(query_intents, list):
                        leftover_intents = [
                            intent
                            for idx, intent in enumerate(query_intents)
                            if idx not in compiled_idx and isinstance(intent, dict)
                        ]
                elif isinstance(query_intents, list):
                    leftover_intents = [
                        intent for intent in query_intents if isinstance(intent, dict)
                    ]
                    sql_source = "pattern"

                leftover_tables: list[str] = []
                for intent in leftover_intents:
                    for raw in intent.get("tables") or []:
                        tid = str(raw).strip().split(".")[-1].lower()
                        if _is_mart_table_id(tid) and tid not in leftover_tables:
                            leftover_tables.append(tid)
                custom_leftover = [
                    intent
                    for intent in leftover_intents
                    if isinstance(intent, dict)
                    and str(intent.get("pattern") or "custom").strip().lower() == "custom"
                ]
                need_nl2sql = self.nl2sql_enabled and (
                    custom_leftover
                    or (not pattern_sqls and not template_sqls and not query_intents)
                )
                pm = [
                    str(m).strip().lower()
                    for m in (primary_measures or [])
                    if str(m).strip()
                ]
                if fast_fact and (
                    "market_price" in pm
                    or "food_security_ipc" in pm
                    or "food_security" in pm
                ):
                    need_nl2sql = False
                if fast_fact and not custom_leftover:
                    need_nl2sql = False
                if contract_sql_only:
                    need_nl2sql = False
                if need_nl2sql:
                    nl_tables = leftover_tables or (
                        sorted(selected_tables) if selected_tables else None
                    )
                    prev_max = os.environ.get("RAG_BQ_MAX_SQL_QUERIES")
                    if fast_fact:
                        os.environ["RAG_BQ_MAX_SQL_QUERIES"] = "1"
                    try:
                        nl2sql_sqls = self._nl_to_sql_queries(
                            query,
                            table_hints=hint_list[:1] if fast_fact and hint_list else hint_list,
                            geo_country=geo_country,
                            geo_countries=geo_countries,
                            time_start=time_start,
                            time_end=time_end,
                            entities=entities,
                            domains=domains,
                            selected_tables=nl_tables,
                            query=query,
                        )
                    finally:
                        if fast_fact:
                            if prev_max is None:
                                os.environ.pop("RAG_BQ_MAX_SQL_QUERIES", None)
                            else:
                                os.environ["RAG_BQ_MAX_SQL_QUERIES"] = prev_max
                    if nl2sql_sqls and not pattern_sqls:
                        sql_source = "nl2sql"
                if not sql_queries:
                    sql_queries = list(pattern_sqls) + list(nl2sql_sqls)

        if explicit_sql:
            sql_source = "explicit"
        elif pattern_sqls and nl2sql_sqls:
            sql_source = "pattern"
        elif pattern_sqls:
            sql_source = "pattern"
        elif nl2sql_sqls:
            sql_source = "nl2sql"
        elif sql_queries and sql_source == "template":
            sql_source = "template"
        elif sql_queries:
            sql_source = sql_source if sql_source != "none" else "nl2sql"
        else:
            sql_source = "none"

        if not sql_queries:
            reason = (
                "NL2SQL disabled and no explicit SQL provided"
                if not self.nl2sql_enabled
                else "NL2SQL produced 0 SELECT queries (no template match)"
            )
            update_current_span_metadata(
                {
                    "table_hints_count": len(hint_list or []),
                    "sql_query_count": 0,
                    "row_count": 0,
                    "latency_ms": trace_elapsed_ms(t0),
                    "status": "no_valid_sql",
                    "nl2sql_model": _nl2sql_model_id(),
                    "sql_source": sql_source,
                }
            )
            self.last_sql_source = sql_source
            self.last_bq_execute_ms = 0.0
            if self.last_bq_nl2sql_ms is None:
                self.last_bq_nl2sql_ms = 0.0
            return [
                _bq_diagnostic_item(
                    status="no_valid_sql",
                    message=reason,
                    prep_error=f"{reason}; model={_nl2sql_model_id()}",
                    nl2sql_raw="; ".join(self._last_nl2sql_raws) if self._last_nl2sql_raws else None,
                )
            ]

        max_queries = max(1, int(os.environ.get("RAG_BQ_MAX_SQL_QUERIES", "10") or 10))
        allowed = set(self.datasets_config.values())
        budget = top_k or self.max_rows
        items: list[dict[str, Any]] = []
        any_usable_rows = False
        prepared_ok = False
        queries_left = max_queries
        execute_t0 = time.perf_counter()

        def _run_sql_batch(
            batch: list[str],
            *,
            source: str,
            slot_ids: list[str] | None = None,
        ) -> None:
            nonlocal budget, any_usable_rows, prepared_ok, queries_left
            for idx, raw_sql in enumerate(batch):
                if budget <= 0 or queries_left <= 0:
                    break
                queries_left -= 1
                point_fact_cap = (
                    fast_fact
                    and template_meta
                    and template_meta.get("template") == "mart_point_fact"
                )
                limit = min(1 if point_fact_cap else rows_per_query, budget)
                validated, prep_err = self._prepare_sql(
                    raw_sql,
                    question=query,
                    table_hints=hint_list,
                    selected_tables=selected_tables,
                    allowed_datasets=allowed,
                    limit=limit,
                    client=client,
                    geo_country=geo_country,
                    geo_countries=geo_countries,
                    time_start=time_start,
                    time_end=time_end,
                    entities=entities,
                    domains=domains,
                    query=query,
                    primary_measures=primary_measures,
                    geography=geography,
                    crop_required=crop_required,
                    geography_required=geography_required,
                    sql_source=source,
                    decomposition=decomp if isinstance(decomp, dict) else None,
                )
                if validated is None:
                    logger.warning(
                        "BQ NL2SQL: validation rejected SQL #%d (%s): %s",
                        idx + 1,
                        prep_err or "unknown",
                        (raw_sql or "")[:300],
                    )
                    meta: dict[str, Any] = {
                        "sql": raw_sql,
                        "sql_index": idx + 1,
                        "sql_count": len(batch),
                        "status": "validation_failed",
                        "validation_failed": True,
                        "sql_source": source,
                        "nl2sql_model": _nl2sql_model_id(),
                    }
                    if prep_err:
                        meta["prep_error"] = prep_err[:500]
                    if template_meta and source == "template":
                        meta["template"] = template_meta.get("template")
                    if pattern_meta and source == "pattern":
                        meta["pattern"] = pattern_meta.get("pattern")
                    items.append({
                        "content": f"[BQ validation failed: {(prep_err or 'invalid SQL')[:200]}]",
                        "source": "bigquery",
                        "metadata": meta,
                    })
                    continue
                prepared_ok = True
                try:
                    from google.cloud.bigquery import QueryJobConfig as _QJC

                    _max_b = max_bytes_billed_for_source(source)
                    _jcfg = _QJC(maximum_bytes_billed=_max_b) if _max_b > 0 else None
                    job = client.query(validated, job_config=_jcfg)
                    rows = list(job.result())
                    billed = int(getattr(job, "total_bytes_billed", 0) or 0)
                    if billed:
                        logger.info("BQ bytes billed sql#%d source=%s: %s", idx + 1, source, billed)
                except Exception as exc:
                    logger.warning(
                        "BQ execution failed for validated SQL #%d: %s (sql: %s)",
                        idx + 1, str(exc)[:200], validated[:200]
                    )
                    exec_meta: dict[str, Any] = {
                        "sql": validated,
                        "sql_index": idx + 1,
                        "sql_count": len(batch),
                        "status": "execution_error",
                        "execution_error": str(exc)[:500],
                        "sql_source": source,
                        "nl2sql_model": _nl2sql_model_id(),
                    }
                    if template_meta and source == "template":
                        exec_meta["template"] = template_meta.get("template")
                    if pattern_meta and source == "pattern":
                        exec_meta["pattern"] = pattern_meta.get("pattern")
                    items.append({
                        "content": f"[BQ execution error: {str(exc)[:200]}]",
                        "source": "bigquery",
                        "metadata": exec_meta,
                    })
                    continue

                if not rows:
                    broadened = broaden_empty_sql_once(
                        validated,
                        crop_required=crop_required,
                        geography_required=geography_required,
                    )
                    if broadened:
                        revalidated, _prep_err = self._prepare_sql(
                            broadened,
                            question=query,
                            table_hints=hint_list,
                            selected_tables=selected_tables,
                            allowed_datasets=allowed,
                            limit=limit,
                            client=client,
                            geo_country=geo_country,
                            geo_countries=geo_countries,
                            time_start=time_start,
                            time_end=time_end,
                            entities=entities,
                            domains=domains,
                            query=query,
                            primary_measures=primary_measures,
                            geography=geography,
                            crop_required=crop_required,
                            geography_required=geography_required,
                            sql_source=source,
                            decomposition=decomp if isinstance(decomp, dict) else None,
                        )
                        if revalidated:
                            try:
                                from google.cloud.bigquery import QueryJobConfig as _QJC

                                _max_b = max_bytes_billed_for_source(source)
                                _jcfg = (
                                    _QJC(maximum_bytes_billed=_max_b) if _max_b > 0 else None
                                )
                                job = client.query(revalidated, job_config=_jcfg)
                                rows = list(job.result())
                                validated = revalidated
                            except Exception as exc:
                                logger.warning(
                                    "BQ broaden-once execute failed: %s",
                                    str(exc)[:200],
                                )

                for row in rows[:limit]:
                    d = dict(row)
                    meta = project_bq_row_acf(
                        {
                            **d,
                            "sql": validated,
                            "sql_index": idx + 1,
                            "sql_count": len(batch),
                            "sql_source": source,
                            "nl2sql_model": _nl2sql_model_id(),
                        }
                    )
                    slot_id = ""
                    if slot_ids and idx < len(slot_ids):
                        slot_id = str(slot_ids[idx] or "").strip()
                    if slot_id:
                        meta["subquestion_id"] = slot_id
                        meta["slot_id"] = slot_id
                    if template_meta and source == "template":
                        meta["template"] = template_meta.get("template")
                    if pattern_meta and source == "pattern":
                        meta["pattern"] = pattern_meta.get("pattern")
                    items.append({
                        "content": str(d),
                        "source": "bigquery",
                        "metadata": meta,
                    })
                    any_usable_rows = True
                    budget -= 1
                    if budget <= 0:
                        break

        if explicit_sql:
            _run_sql_batch(sql_queries, source="explicit")
        elif pattern_sqls or nl2sql_sqls:
            if pattern_sqls:
                _run_sql_batch(pattern_sqls, source="pattern", slot_ids=pattern_slot_ids or None)
            if nl2sql_sqls:
                _run_sql_batch(nl2sql_sqls, source="nl2sql")
        else:
            _run_sql_batch(sql_queries, source=sql_source)

        # After NL2SQL/pattern prepare failures or 0-row success, try deterministic SQL.
        if (
            sql_source in {"nl2sql", "pattern"}
            and not any_usable_rows
            and selected_tables
        ):
            if not pattern_sqls:
                rescue_hits = try_sql_patterns(
                    query_intents,
                    project_id=self.project_id,
                    dataset=ds_name,
                    query=query,
                    entities=entities,
                    time_start=time_start,
                    time_end=time_end,
                    selected_tables=selected_tables,
                    limit=rows_per_query,
                    geo_country=geo_country,
                    geo_countries=geo_countries,
                )
                if rescue_hits:
                    pattern_meta = {
                        "hits": rescue_hits,
                        "pattern": rescue_hits[0].get("pattern"),
                    }
                    rescue_sqls = [
                        str(h["sql"]) for h in rescue_hits if str(h.get("sql") or "").strip()
                    ]
                    if rescue_sqls:
                        _run_sql_batch(rescue_sqls, source="pattern")
            if sql_source != "template":
                tmpl_batch = _try_template_sql()
                if tmpl_batch:
                    _run_sql_batch(tmpl_batch, source="template")

        if not items:
            if prepared_ok:
                reason = "SQL executed successfully but returned no rows"
                items.append(
                    _bq_diagnostic_item(
                        status="empty_result",
                        message=reason,
                        prep_error=f"{reason}; model={_nl2sql_model_id()}",
                        nl2sql_raw="; ".join(self._last_nl2sql_raws) if self._last_nl2sql_raws else None,
                    )
                )
            else:
                reason = "All SQL attempts failed validation or execution"
                items.append(
                    _bq_diagnostic_item(
                        status="no_valid_sql",
                        message=reason,
                        prep_error=f"{reason}; model={_nl2sql_model_id()}",
                        nl2sql_raw="; ".join(self._last_nl2sql_raws) if self._last_nl2sql_raws else None,
                    )
                )

        sql_hashes = list(
            dict.fromkeys(
                sql_hash(str((item.get("metadata") or {}).get("sql") or ""))
                for item in items
                if sql_hash(str((item.get("metadata") or {}).get("sql") or ""))
            )
        )
        update_current_span_metadata(
            {
                "table_hints_count": len(hint_list or []),
                "sql_query_count": len(sql_queries),
                "row_count": len(items),
                "sql_hashes": sql_hashes[:10],
                "latency_ms": trace_elapsed_ms(t0),
                "bq_execute_ms": trace_elapsed_ms(execute_t0),
                "sql_source": sql_source,
                "nl2sql_model": _nl2sql_model_id(),
                "task_mode": task_mode or None,
            }
        )
        self.last_sql_source = sql_source
        self.last_bq_execute_ms = trace_elapsed_ms(execute_t0)
        if self.last_bq_nl2sql_ms is None:
            self.last_bq_nl2sql_ms = 0.0
        return items
