"""
BigQuery retriever: natural-language questions → BigQuery SQL over the bronze dataset only.
Uses Llama 3.1 via Hugging Face for NL-to-SQL; validates and runs only SELECTs.
"""
from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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

# Production note: The main API (app/api.py) calls load_rag_dotenv early using the
# unified loader (respecting config/.env and pure env vars for GCE).
# This private helper is retained for direct script usage / backward compat only.
# It now delegates to the standard loader so there is no separate data/local logic
# in production code paths.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv() -> None:
    """Delegate to the single RAG env loader. Safe to call multiple times."""
    try:
        load_rag_dotenv(_REPO_ROOT)
    except Exception:
        # Never break callers; vars may already be present via container env.
        pass


def _get_datasets_config() -> dict[str, str]:
    """Bronze dataset IDs from env (for NL-to-SQL and validation)."""
    return {
        "bronze": os.environ.get("BQ_DATASET_BRONZE", "bronze").strip(),
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
        purpose="nl2sql",
    )


# Filter-column mapping aligned with bronze dbt sources. Use exact names from the Schema section.
_SCHEMA_FILTER_GUIDE = """
Filter columns by question intent (use exact column names from the schema below):
- Country/region: country, country_code, country_name, area (FAO tables), adm0_name, reporting_country, Reference area, geographic_unit_name, fewsnet_region, admin_1, admin_2, admin_0
- Season / time: season_name, planting_year, harvest_year, year, TIME_PERIOD, period_date, projection_start, projection_end, reporting_date, mp_year, mp_month, first_period_date, last_period_date
- Product / crop: product, item (FAO), Commodity, cpcv2_description, product_name, cm_name
- Admin/geography: admin_1, admin_2, admin_0, geographic_unit_name, fewsnet_region, admin_region, agroecological_zone
- Scenario: scenario_name

Query patterns (use these so the result directly answers the question):
- "Past decade" / "over the past N years" -> add WHERE planting_year >= (e.g. 2014) OR harvest_year BETWEEN ... OR year >= ... so only recent data is returned.
- "Which regions" / "which countries" / "which districts" -> GROUP BY the region column (country, admin_1, geographic_unit_name, etc.) and return one row per region; do not use SELECT * LIMIT 10.
- "Most significant changes" / "biggest changes" / "largest increase" -> compute a change metric (e.g. MAX(yield) - MIN(yield), or (yield in latest year - yield in earliest year)), GROUP BY region, ORDER BY that change DESC, LIMIT 10 or 20.
- "Compare" / "trends over time" -> GROUP BY region and year (or period), optionally aggregate (AVG(yield), SUM(production)), ORDER BY year/period.
- Always filter time when the question mentions a period; always aggregate and order when the question asks for "which" or "most".

Table hints (bronze dataset only — use only tables that appear in the Schema section):
- Yield/crop production: yield_raw_data (country, product, season_name, planting_year, harvest_year, area, production, yield)
- Food security / IPC: fews_net_food_security_master (country, geographic_unit_name, scenario_name, projection_start/end, ipc_phase_value)
- FAO: fao_rfn, fao_rl, fao_rp, fao_tcl, fao_ti, fao_fbs, fao_qcl (area/country_name, item, year, value)
- Cropland: cropland_area_summary_2019_africa (agroecological_zone and related columns per schema)
- GDP / development: africa_gdp_ppp, africa_Human_development_index (country, year)
"""

# Forbidden SQL tokens (case-insensitive) for safety
_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|MERGE|TRUNCATE|ALTER|GRANT|REVOKE|EXEC|EXECUTE|CALL)\b",
    re.IGNORECASE,
)

_QUERY_SPLIT_RE = re.compile(r"\n---+\s*(?:QUERY)?\s*---+\n", re.IGNORECASE)


def _format_query_constraints(
    *,
    geo_country: str | None,
    geo_countries: list[str] | None = None,
    time_start: str | None,
    time_end: str | None,
    entities: list[str] | None,
    domains: list[str] | None,
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
    # Ensure referenced datasets are in the allowed set (e.g. bronze only for RAG)
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
    Retrieve context by querying BigQuery. Uses the bronze dataset only (BQ_DATASET_BRONZE).
    Uses Llama 3.1 (HF) for NL-to-SQL when no explicit sql is provided.
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
        """Build a compact schema summary for configured datasets (bronze only); cached across processes via Redis when configured."""
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
        """Compact schema text; skip live BQ catalog when rich per-table hints are present."""
        skip_live = os.environ.get("RAG_BQ_SKIP_LIVE_SCHEMA", "on").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
        )
        if skip_live and table_hints:
            ds = self.datasets_config.get("bronze", "").strip()
            return (
                f"Project: `{self.project_id}`. Bronze dataset: `{ds}`. "
                "Use only the table and columns described in the table hint below."
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
    ) -> list[dict[str, str]]:
        schema_text = self._schema_for_nl2sql(table_hints)
        constraints_block = _format_query_constraints(
            geo_country=geo_country,
            geo_countries=geo_countries,
            time_start=time_start,
            time_end=time_end,
            entities=entities,
            domains=domains,
        )
        hints_block = ""
        if table_hints:
            cleaned = [str(h).strip() for h in table_hints if str(h).strip()]
            if cleaned:
                per_hint_cap = int(os.environ.get("RAG_BQ_HINT_MAX_CHARS", "2500") or 2500)
                hints_block = (
                    "\n\nPrioritized table descriptions from the knowledge base "
                    "(use the table(s) below; honor sql_generation_hints, filtering_guidance, "
                    "and aggregation_rules):\n"
                    + "\n".join(f"- {h[:per_hint_cap]}" for h in cleaned)
                )
        if multi_query:
            output_rule = (
                f"6) Output up to {max_queries} separate BigQuery SELECT queries — one per relevant "
                f"table hint when possible. Put each query on its own block separated by a line "
                f"containing only ---QUERY---. No explanation, no markdown fences."
            )
        else:
            output_rule = (
                "6) Output exactly one SELECT for the single table hint provided. "
                "No explanation, no markdown, no code fence."
            )
        system = (
            "You are a BigQuery expert for OpenTrace agricultural and food-security data in the bronze dataset only. "
            "Rules: "
            "1) Use ONLY tables and columns from the Schema section (bronze dataset). "
            "Use full names: `project.dataset.table`. "
            "2) When Query constraints are present, REQUIRED country and time filters MUST appear in every SELECT. "
            "3) Match country columns to schema (country, country_name, Area, adm0_name, geographic_unit_name). "
            "4) Match time columns to schema (year, planting_year, harvest_year, observation_year, TIME_PERIOD). "
            "5) Use table hints for table/column choice; prefer GROUP BY / ORDER BY over bare SELECT * LIMIT. "
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
        )
        raw = _call_llama_for_sql(messages)
        sql = _extract_single_select(raw)
        if not sql and raw:
            logger.warning("NL-to-SQL: LLM returned non-SELECT text (first 200 chars): %s", raw[:200])
        return sql

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
    ) -> list[str]:
        """
        Generate up to RAG_BQ_MAX_SQL_QUERIES (default 10) SELECT statements via NL-to-SQL.

        Mode RAG_BQ_NL2SQL_MODE:
        - per_hint (default): one LLM call per vector-matched table hint (up to max queries).
        - batch: one LLM call returning multiple queries separated by ---QUERY---.
        """
        max_queries = max(1, int(os.environ.get("RAG_BQ_MAX_SQL_QUERIES", "10") or 10))
        mode = os.environ.get("RAG_BQ_NL2SQL_MODE", "per_hint").strip().lower()
        cleaned_hints = [str(h).strip() for h in (table_hints or []) if str(h).strip()]
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
                        table_hints=[hint] if hint else None,
                        geo_country=geo_country,
                        geo_countries=geo_countries,
                        time_start=time_start,
                        time_end=time_end,
                        entities=entities,
                        domains=domains,
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

    def _fallback_sql(self, question: str) -> str:
        """Minimal fallback when NL-to-SQL returns nothing (connectivity / LLM down)."""
        q = (question or "").lower()
        proj = self.project_id
        bronze_ds = self.datasets_config.get("bronze", "").strip()
        if not bronze_ds:
            return ""
        if any(x in q for x in ("which region", "which country", "which district", "most significant change", "past decade", "over the past")) and any(x in q for x in ("yield", "productivity", "crop", "production")):
            return (
                f"SELECT country, "
                f"MAX(yield) - MIN(yield) AS yield_change, "
                f"ROUND(AVG(yield), 2) AS avg_yield, "
                f"COUNT(*) AS n_obs "
                f"FROM `{proj}.{bronze_ds}.yield_raw_data` "
                f"WHERE planting_year >= 2014 AND yield IS NOT NULL "
                f"GROUP BY country "
                f"ORDER BY yield_change DESC "
                f"LIMIT {min(20, self.max_rows)}"
            )
        try:
            from google.cloud.bigquery import DatasetReference
            for _layer, ds_id in self.datasets_config.items():
                if not ds_id:
                    continue
                try:
                    dataset_ref = DatasetReference(proj, ds_id)
                    table_list = list(self._get_client().list_tables(dataset_ref))
                    if table_list:
                        t = table_list[0]
                        return f"SELECT * FROM `{proj}.{ds_id}.{t.table_id}` LIMIT {min(10, self.max_rows)}"
                except Exception:
                    continue
        except Exception:
            pass
        return ""

    @_observe_span(as_type="span", name="retrieval.bq", capture_input=False, capture_output=False)
    def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Run one or more BQ queries and return rows as context items.

        NL-to-SQL generates up to RAG_BQ_MAX_SQL_QUERIES (default 10) SELECTs, typically
        one per vector-matched table hint. kwargs["sql"] may be a single string or list.

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
            )
        if not sql_queries:
            fb = self._fallback_sql(query)
            if fb:
                sql_queries = [fb]

        if not sql_queries:
            update_current_span_metadata(
                {
                    "table_hints_count": len(hint_list or []),
                    "sql_query_count": 0,
                    "row_count": 0,
                    "latency_ms": trace_elapsed_ms(t0),
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
            validated = _validate_sql(raw_sql, allowed, limit)
            if validated is None:
                logger.warning(
                    "BQ NL2SQL: validation rejected SQL #%d (first 300 chars): %s",
                    idx + 1, (raw_sql or "")[:300]
                )
                # Still surface the attempted SQL for debugging (software team handoff)
                items.append({
                    "content": "[BQ validation failed for this query]",
                    "source": "bigquery",
                    "metadata": {
                        "sql": raw_sql,
                        "sql_index": idx + 1,
                        "sql_count": len(sql_queries),
                        "validation_failed": True,
                    },
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
                items.append({
                    "content": str(d),
                    "source": "bigquery",
                    "metadata": {
                        **d,
                        "sql": validated,
                        "sql_index": idx + 1,
                        "sql_count": len(sql_queries),
                    },
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
