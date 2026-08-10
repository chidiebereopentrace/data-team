"""
BigQuery retriever: natural-language questions → BigQuery SQL over staging_dev only.
Uses an LLM for NL-to-SQL; validates and runs only SELECTs against BQ_DATASET_SILVER.
"""
from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ml.rag.chatbot.acf_metadata import project_bq_row_acf
from ml.rag.chatbot.bq_byte_budget import hint_max_bytes, truncate_utf8
from ml.rag.chatbot.bq_sql_validate import (
    dry_run_sql,
    sql_retry_enabled,
    validate_sql_table_allowlist,
)
from ml.rag.chatbot.query_decomposer import _NON_COUNTRY_GEO
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


def _load_dotenv() -> None:
    """Delegate to the single RAG env loader. Safe to call multiple times."""
    try:
        load_rag_dotenv(_REPO_ROOT)
    except Exception:
        pass


def _get_datasets_config() -> dict[str, str]:
    """Staging (silver) dataset ID from env — sole NL-to-SQL target."""
    return {
        "staging": os.environ.get("BQ_DATASET_SILVER", "staging_dev").strip(),
    }


def _call_llama_for_sql(messages: list[dict[str, str]], *, max_tokens: int | None = None) -> str:
    """Call LLM for NL-to-SQL; return raw text (expected to be SQL) or empty string."""
    cap = max_tokens or int(os.environ.get("RAG_BQ_NL2SQL_MAX_TOKENS", "1024") or 1024)
    bq_timeout = float(os.environ.get("RAG_BQ_NL2SQL_TIMEOUT_S", "0") or 0) or llm_default_timeout_s()
    return llm_chat_complete(
        messages,
        model=llm_model_id(),
        max_tokens=cap,
        temperature=0.0,
        timeout_s=bq_timeout,
        purpose="bq.nl2sql",
    )


# Filter-column mapping aligned with staging_dev stg_* models.
_SCHEMA_FILTER_GUIDE = """
Filter columns by question intent (use exact column names from the Schema / table hints):
- Country/region: country, country_code, country_name, admin_0, admin_1, admin_2, geographic_unit_name, fewsnet_region, fnid
- Season / time: season_name, planting_year, harvest_year, year, month, observation_year, observation_time
- Product / crop: product, product_name, item, item_code
- Markets / prices: market_name, price_type, currency, value, common_currency_price
- Food security: phase_code, phase_name, pct_phase3, pct_phase4, pct_phase5, measure_type, scenario_name

Query patterns:
- Time-bounded questions -> WHERE year/harvest_year/planting_year in range
- Which regions/countries -> GROUP BY geography column; avoid bare SELECT * LIMIT
- Trends / compare -> GROUP BY geography and year; ORDER BY year

Staging tables (use only tables in the Schema / table hints — examples):
- Yield/crop: stg_yield_raw_data
- Food security / IPC: stg_fews_food_security
- Market prices: stg_fews_market_prices, stg_wfp_vampire_prices, stg_faostat_prices
- Production / trade: stg_faostat_production, stg_faostat_trade
- Climate: stg_nasa_power, stg_copernicus_era5
- Soil: stg_isric_africa_soil, stg_isda_soil_enriched
- HDI / GDP: stg_africa_hdi, stg_africa_gdp_ppp
"""

# Forbidden SQL tokens (case-insensitive) for safety
_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|MERGE|TRUNCATE|ALTER|GRANT|REVOKE|EXEC|EXECUTE|CALL)\b",
    re.IGNORECASE,
)

_QUERY_SPLIT_RE = re.compile(r"\n---+\s*(?:QUERY)?\s*---+\n", re.IGNORECASE)


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
        "African countries using country_name (or equivalent) on the fact table; "
        "do NOT filter country_name = 'Africa'; do NOT join GDP/HDI tables for a "
        "country list unless that table is in the selected table set"
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
            "(use IN (...) or OR on country, country_name, area, adm0_name, geographic_unit_name; "
            "GROUP BY country when comparing)"
        )
    elif len(countries) == 1:
        lines.append(
            f"- REQUIRED country/area filter: {countries[0]!r} "
            "(use country, country_name, area, Area, adm0_name, or geographic_unit_name per schema)"
        )
    if time_start or time_end:
        lines.append(
            f"- REQUIRED time range: start={time_start or 'any'}, end={time_end or 'any'} "
            "(use year, planting_year, harvest_year, observation_year, TIME_PERIOD, or Y#### columns)"
        )
    if entities:
        ent = [str(e).strip() for e in entities if str(e).strip()]
        if ent:
            lines.append(f"- Key entities to cover in filters or SELECT: {', '.join(ent)}")
    if domains:
        dom = [str(d).strip() for d in domains if str(d).strip()]
        if dom:
            lines.append(f"- Topic domains: {', '.join(dom)}")
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
            if block.strip().upper().startswith("SELECT"):
                return block.strip().rstrip(";")
        text = re.sub(r"```[\s\S]*?```", "", text).strip()
    # Accept a leading SELECT or the first SELECT embedded after template/prose noise
    if text.upper().startswith("SELECT"):
        return text.rstrip(";")
    match = re.search(r"(SELECT\b[\s\S]*)", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().rstrip(";")
        if candidate.upper().startswith("SELECT"):
            return candidate
    return ""


def _parse_sql_queries(raw: str, max_queries: int) -> list[str]:
    """Parse up to max_queries SELECT statements from one LLM response."""
    if not raw or max_queries < 1:
        return []
    chunks = _QUERY_SPLIT_RE.split(raw)
    if len(chunks) <= 1:
        chunks = re.split(r"\n(?=SELECT\s)", raw, flags=re.IGNORECASE)
    seen: set[str] = set()
    out: list[str] = []
    for chunk in chunks:
        sql = _extract_single_select(chunk)
        if not sql.upper().startswith("SELECT"):
            continue
        norm = " ".join(sql.split())
        if norm in seen:
            continue
        seen.add(norm)
        out.append(sql)
        if len(out) >= max_queries:
            break
    return out


def _validate_sql(sql: str, allowed_dataset_ids: set[str], default_limit: int) -> str | None:
    """
    Ensure SQL is a safe SELECT-only query over allowed datasets. Returns cleaned SQL or None.
    """
    normalized = " ".join(sql.split()).strip()
    if not normalized.upper().startswith("SELECT"):
        return None
    if _FORBIDDEN_SQL.search(normalized):
        return None
    # Ensure referenced datasets are in the allowed set (staging_dev only for RAG)
    if allowed_dataset_ids:
        allowed_lower = {a.lower() for a in allowed_dataset_ids}
        for part in re.findall(r"`?[\w.]+`?", normalized):
            part = part.strip("`")
            if "." in part:
                segments = part.split(".")
                # dataset.table or project.dataset.table
                ds = segments[-2].lower()
                if ds not in allowed_lower:
                    return None
    if "LIMIT" not in normalized.upper():
        normalized = f"{normalized.rstrip(';')} LIMIT {default_limit}"
    return normalized


class BQRetriever(BaseRetriever):
    """
    Retrieve context by querying BigQuery. Uses staging_dev only (BQ_DATASET_SILVER).
    Uses an LLM for NL-to-SQL when no explicit sql is provided.
    """

    def __init__(
        self,
        project_id: str | None = None,
        max_rows: int = 100,
        nl2sql_enabled: bool | None = None,
    ):
        _load_dotenv()
        self.project_id = (project_id or os.environ.get("BQ_PROJECT", "")).strip()
        self.datasets_config = _get_datasets_config()
        self.max_rows = max_rows
        if nl2sql_enabled is not None:
            self.nl2sql_enabled = nl2sql_enabled
        else:
            self.nl2sql_enabled = os.environ.get("RAG_BQ_NL2SQL_ENABLED", "1").strip().lower() in ("1", "true", "on")
        self._client = None

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
        """Build a compact schema summary for staging_dev; cached via Redis when configured."""
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
            ds = self.datasets_config.get("staging", "").strip()
            return (
                f"Project: `{self.project_id}`. Staging dataset: `{ds}`. "
                "Use only the stg_* tables and columns described in the table hints below."
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
                    "\n\nSelected staging_dev table schemas from YAML "
                    "(use only these tables; honor sql_generation_hints and filtering_guidance):\n"
                    + "\n".join(packed)
                )
                if hints_truncated:
                    hints_block += "\n[bq_hint_truncated=true]"
        if multi_query:
            output_rule = (
                f"8) Output up to {max_queries} separate BigQuery SELECT queries when needed. "
                f"Put each query on its own block separated by a line containing only ---QUERY---. "
                "No explanation, no markdown fences."
            )
        else:
            output_rule = (
                "8) Output exactly one SELECT. No explanation, no markdown, no code fence."
            )
        ds = self.datasets_config.get("staging", "staging_dev")
        allowed_tables = [
            str(t).strip().split(".")[-1]
            for t in (selected_tables or [])
            if str(t).strip()
        ]
        tables_rule = (
            f"Tables you may use (exactly these stg_* tables): {', '.join(allowed_tables)}. "
            "JOINs are allowed ONLY between these tables using on= keys from semantic_relationships "
            "in the hints. Never reference a stg_* table outside this list."
            if allowed_tables
            else "Use ONLY stg_* tables and columns from the Schema / table hints."
        )
        system = (
            f"You are a BigQuery expert for OpenTrace agricultural data in the `{ds}` dataset only. "
            "Rules: "
            f"1) {tables_rule} "
            f"Use full names: `{self.project_id}.{ds}.stg_*`. "
            "2) Use exact column names from the Columns blocks — never invent columns "
            "(e.g. use country_name, not country). "
            "3) When Query constraints are present, REQUIRED country and time filters MUST appear. "
            "4) For FAOSTAT production rankings: filter element and year, then "
            "SUM(value) GROUP BY country_name ORDER BY total DESC — not MAX per product row. "
            "5) Prefer GROUP BY / ORDER BY over bare SELECT * LIMIT. "
            f"{output_rule}"
        )
        constraints_section = f"\n\n{constraints_block}\n" if constraints_block else ""
        user = (
            f"Filter and table hints:\n{_SCHEMA_FILTER_GUIDE}"
            f"{constraints_section}"
            f"{hints_block}\n\n"
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
    ) -> tuple[str | None, str | None]:
        """
        Validate SQL, enforce table allowlist, dry-run, and optionally retry once.

        Returns (validated_sql, error_message). error_message is set when SQL cannot run.
        """
        validated = _validate_sql(raw_sql, allowed_datasets, limit)
        if validated is None:
            return None, "validation_failed"

        allow_err = validate_sql_table_allowlist(validated, selected_tables)
        if allow_err:
            return None, allow_err

        dry_err = dry_run_sql(client, validated)
        if dry_err and sql_retry_enabled():
            retry_question = (
                f"{question}\n\n"
                f"Previous SQL failed BigQuery dry-run:\n{dry_err}\n\n"
                "Fix the SQL. Use ONLY columns from the Columns blocks in the table hints. "
                "JOIN only tables in the selected set using documented on= keys. "
                "Do not reference stg_* tables outside the selected set."
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
                validated = _validate_sql(retry_sql, allowed_datasets, limit)
                if validated is None:
                    return None, f"retry_validation_failed: {dry_err[:200]}"
                allow_err = validate_sql_table_allowlist(validated, selected_tables)
                if allow_err:
                    return None, allow_err
                dry_err = dry_run_sql(client, validated)
        if dry_err:
            return None, f"dry_run_failed: {dry_err[:300]}"
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

        with observed_span(
            "retrieval.bq.nl2sql",
            input_data={
                "question": question[:200],
                "table_hints_count": len(cleaned_hints),
                "mode": mode,
            },
        ):
            parallel_used = False
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
                parsed = _parse_sql_queries(_call_llama_for_sql(messages), max_queries)
                if parsed:
                    queries = parsed

            if not queries:
                hints_for_calls = cleaned_hints[:max_queries] if cleaned_hints else [None]
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
                    with ThreadPoolExecutor(max_workers=min(workers, len(hints_for_calls))) as pool:
                        futs = {
                            pool.submit(run_with_tracing_context(_gen_one, h)): h
                            for h in hints_for_calls
                        }
                        for fut in as_completed(futs):
                            sql = fut.result()
                            if not sql:
                                continue
                            norm = " ".join(sql.split())
                            if norm in seen:
                                continue
                            seen.add(norm)
                            queries.append(sql)
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
                }
            )
            return queries[:max_queries]

    @_observe_span(as_type="span", name="retrieval.bq", capture_input=False, capture_output=False)
    def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Run one or more BQ queries and return rows as context items.

        NL-to-SQL generates up to RAG_BQ_MAX_SQL_QUERIES (default 10) SELECTs from
        reasoner-selected YAML table hints. kwargs["sql"] may be a single string or list.
        Fail closed: if no validated SQL is produced, return [] (no canned fallback query).

        Optional kwargs: geo_country, time_start, time_end, entities, domains, table_hints.
        Graph node aggregates distinct executed SQL into state ``bq_sql_queries``.
        """
        t0 = time.perf_counter()
        if not self.project_id:
            update_current_span_metadata({"status": "no_project", "row_count": 0})
            return []
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
                if tid.startswith("stg_"):
                    selected_tables.add(tid)

        sql_input = kwargs.get("sql")
        sql_queries: list[str] = []
        if isinstance(sql_input, str) and sql_input.strip():
            sql_queries = [sql_input.strip()]
        elif isinstance(sql_input, list):
            sql_queries = [str(s).strip() for s in sql_input if str(s).strip()]

        if not sql_queries and self.nl2sql_enabled:
            sql_queries = self._nl_to_sql_queries(
                query,
                table_hints=hint_list,
                geo_country=geo_country,
                geo_countries=geo_countries,
                time_start=time_start,
                time_end=time_end,
                entities=entities,
                domains=domains,
                selected_tables=sorted(selected_tables) if selected_tables else None,
                query=query,
            )

        if not sql_queries:
            update_current_span_metadata(
                {
                    "table_hints_count": len(hint_list or []),
                    "sql_query_count": 0,
                    "row_count": 0,
                    "latency_ms": trace_elapsed_ms(t0),
                    "status": "no_valid_sql",
                }
            )
            return []

        max_queries = max(1, int(os.environ.get("RAG_BQ_MAX_SQL_QUERIES", "10") or 10))
        rows_per_query = max(1, int(os.environ.get("RAG_BQ_ROWS_PER_QUERY", "10") or 10))
        allowed = set(self.datasets_config.values())
        budget = top_k or self.max_rows
        items: list[dict[str, Any]] = []

        for idx, raw_sql in enumerate(sql_queries[:max_queries]):
            if budget <= 0:
                break
            limit = min(rows_per_query, budget)
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
                    "sql_count": len(sql_queries),
                    "validation_failed": True,
                }
                if prep_err:
                    meta["prep_error"] = prep_err[:500]
                items.append({
                    "content": f"[BQ validation failed: {(prep_err or 'invalid SQL')[:200]}]",
                    "source": "bigquery",
                    "metadata": meta,
                })
                continue
            try:
                job = client.query(validated)
                rows = list(job.result())
            except Exception as exc:
                logger.warning(
                    "BQ execution failed for validated SQL #%d: %s (sql: %s)",
                    idx + 1, str(exc)[:200], validated[:200]
                )
                items.append({
                    "content": f"[BQ execution error: {str(exc)[:200]}]",
                    "source": "bigquery",
                    "metadata": {
                        "sql": validated,
                        "sql_index": idx + 1,
                        "sql_count": len(sql_queries),
                        "execution_error": str(exc)[:500],
                    },
                })
                continue

            for row in rows[:limit]:
                d = dict(row)
                meta = project_bq_row_acf(
                    {
                        **d,
                        "sql": validated,
                        "sql_index": idx + 1,
                        "sql_count": len(sql_queries),
                    }
                )
                items.append({
                    "content": str(d),
                    "source": "bigquery",
                    "metadata": meta,
                })
                budget -= 1
                if budget <= 0:
                    break

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
            }
        )
        return items
