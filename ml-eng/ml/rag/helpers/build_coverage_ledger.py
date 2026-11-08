"""Build the coverage ledger: every indicator class and mart object with disposition and readiness.

Run from ``ml-eng`` (no BigQuery required; reads repository contracts and the coverage snapshot):
  python -m ml.rag.helpers.build_coverage_ledger

Writes ``ml/rag/artifacts/coverage_ledger.json`` and ``coverage_ledger.md``. Readiness is
earned, never declared: ``contracted`` needs a class, measures and registry operations;
``fixture_verified`` needs an offline probe bind that compiles and passes the SQL AST gate;
``integration_verified`` needs a live read-only result in ``artifacts/live_verification.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml.rag.chatbot.bq_sql_validate import validate_sql_column_allowlist
from ml.rag.chatbot.bq_table_schema_yaml import (
    TableBindContract,
    compile_table_bind_contract,
    discriminator_columns,
    load_mart_table_schema,
    measure_columns,
    measure_columns_mart,
    year_column,
)
from ml.rag.chatbot.compile_sql_from_bind import compile_sql_from_bind
from ml.rag.chatbot.coverage_ledger import LEDGER_PATH, OPERATIONS, min_readiness
from ml.rag.chatbot.sql_ast_validate import parse_select, physical_tables

RAG_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RAG_DIR.parents[2]
YAML_DIR = RAG_DIR / "bq_mart_tables_yaml_files"
CLASSES_PATH = RAG_DIR / "helpers" / "mart_indicator_classes.yaml"
REGISTRY_PATH = RAG_DIR / "helpers" / "capability_registry.yaml"
CORPUS_POLICY_PATH = RAG_DIR / "helpers" / "class_corpus_policy.yaml"
CARDS_DIR = RAG_DIR / "schema_cards"
COVERAGE_PATH = RAG_DIR / "artifacts" / "mart_coverage.json"
LIVE_PATH = RAG_DIR / "artifacts" / "live_verification.json"
DBT_MART_DIR = REPO_DIR / "data-eng" / "dbt" / "models" / "mart_dev"
SUMMARY_PATH = LEDGER_PATH.with_suffix(".md")

_REF_FROM_RE = re.compile(r"\bfrom\s+\{\{\s*ref\('([a-z0-9_]+)'\)\s*\}\}", re.I)
_TIME_COLS = frozenset({"year", "month", "date", "as_of_date", "period", "season_year", "time_key", "date_key"})
_GEO_COLS = frozenset({"country_iso3", "country_name", "geography_key", "geo_key", "admin_1_name", "admin_2_name", "city_name", "market_name", "region"})
_LINEAGE_COLS = frozenset({"source_key", "loaded_at", "row_count", "audit_key", "ingested_at"})
_PROBE_ISO3, _PROBE_COUNTRY, _PROBE_YEAR = "KEN", "Kenya", 2020


def _yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def input_paths() -> list[Path]:
    paths = [CLASSES_PATH, REGISTRY_PATH, CORPUS_POLICY_PATH, *YAML_DIR.glob("*.yml"), *CARDS_DIR.glob("*.yaml")]
    paths += [p for p in (COVERAGE_PATH, LIVE_PATH) if p.is_file()]
    paths += (DBT_MART_DIR / "aggregates").glob("*.sql")
    return sorted(paths)


def snapshot_hash() -> str:
    """Hash of every ledger input; line endings are normalized so checkouts agree."""
    digest = hashlib.sha256()
    for path in input_paths():
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


def _lineage_parents() -> dict[str, str]:
    out: dict[str, str] = {}
    for sql in sorted((DBT_MART_DIR / "aggregates").glob("agg_*.sql")):
        match = _REF_FROM_RE.search(sql.read_text(encoding="utf-8"))
        if match:
            out[sql.stem] = match.group(1).lower()
    return out


def _disposition(table: str, classes: list[str]) -> tuple[str, str]:
    if table.startswith("dim_"):
        return "dimension", ""
    if table.startswith("bridge_"):
        return "bridge", ""
    if classes:
        return "serving", ""
    return "unsupported", "no indicator class lists or labels this table"


def _column_roles(table: str, schema: dict[str, Any], excluded: set[str]) -> dict[str, str]:
    measures = {c.lower() for c in (measure_columns_mart(table) or measure_columns(table))}
    filters = {c.lower() for c in discriminator_columns(table)}
    roles: dict[str, str] = {}
    for col in schema.get("columns") or []:
        name = str(col.get("name") or "").strip().lower()
        if not name:
            continue
        if name in _TIME_COLS:
            roles[name] = "time"
        elif name in _GEO_COLS:
            roles[name] = "geo"
        elif name in _LINEAGE_COLS:
            roles[name] = "lineage"
        elif name in measures:
            roles[name] = "measure"
        elif name in excluded:
            roles[name] = "excluded"
        elif name.endswith("_key"):
            roles[name] = "key"
        elif name in filters or f"{name}_value_samples" in schema:
            roles[name] = "filter"
        else:
            roles[name] = "attribute"
    return roles


def _derived_operations(table: str, roles: dict[str, str]) -> dict[str, tuple[bool, str]]:
    """What the table's shape permits; the registry declares what is served."""
    has_measure = "measure" in roles.values()
    has_country = "geo" in roles.values()
    has_time = year_column(table) is not None or "time" in roles.values()
    snapshot = table.endswith("_latest")
    has_category = bool(discriminator_columns(table))
    no_measure = (False, "no measure column")
    return {
        "lookup": (True, "") if has_measure else no_measure,
        "latest": (True, "") if has_measure and has_time else (False, "no time column"),
        "series": (False, "single snapshot table") if snapshot else ((True, "") if has_measure and has_time else (False, "no time column")),
        "compare": (True, "") if has_measure and has_country else (False, "no geography column"),
        "rank": (True, "") if has_measure and has_country else (False, "no geography column"),
        "categorical": (True, "") if has_category else (False, "no categorical discriminator"),
    }


def _declared_operations(registry: dict[str, Any], table: str) -> dict[str, Any] | None:
    block = (registry.get("table_operations") or {}).get(table)
    return block if isinstance(block, dict) else None


def _probe(table: str) -> tuple[bool, str, str]:
    """Compile a Kenya/2020 bind through the production compiler and gate it offline."""
    facets = {
        "geography": [_PROBE_COUNTRY],
        "time_start": f"{_PROBE_YEAR}-01-01",
        "time_end": f"{_PROBE_YEAR}-12-31",
    }
    contract = compile_table_bind_contract(
        table,
        facets=facets,
        query=f"value in {_PROBE_COUNTRY} in {_PROBE_YEAR}",
        country_labels=[_PROBE_ISO3],
    )
    if not contract.measure_columns:
        contract = TableBindContract.from_dict(
            {**contract.to_dict(), "measure_columns": measure_columns_mart(table) or measure_columns(table)}
        ) or contract
    sql = compile_sql_from_bind(contract)
    if not sql:
        return False, "bind compiler produced no SQL for the probe request", ""
    root, err = parse_select(sql)
    if err:
        return False, f"AST gate: {err}", sql
    tables = {ref.name.lower() for ref in physical_tables(root)}
    unknown = sorted(t for t in tables if load_mart_table_schema(t) is None)
    if unknown:
        return False, f"probe reads tables without a mart YAML: {', '.join(unknown)}", sql
    col_err = validate_sql_column_allowlist(sql)
    if col_err:
        return False, col_err, sql
    return True, "", sql


def _coverage_summary(coverage: dict[str, Any], table: str, parent: str | None) -> dict[str, Any]:
    entries = [e for e in coverage.get("entries") or [] if e.get("table_id") == table]
    source = table
    if not entries and parent:
        entries = [e for e in coverage.get("entries") or [] if e.get("table_id") == parent]
        source = parent if entries else ""
    if not entries:
        return {"source": "", "members": 0, "min_year": None, "max_year": None}
    years = [(e.get("min_year"), e.get("max_year")) for e in entries if e.get("min_year") is not None]
    return {
        "source": source,
        "members": len({e.get("iso3") or e.get("geo_value") for e in entries}),
        "min_year": min((y[0] for y in years), default=None),
        "max_year": max((y[1] for y in years), default=None),
    }


def _class_tables(spec: dict[str, Any]) -> set[str]:
    """Fact and aggregate tables a taxonomy class routes to: primary, companion and family tables."""
    raw = [*(spec.get("primary_facts") or []), *(spec.get("companion_facts") or [])]
    raw += [f.get("table") for f in spec.get("families") or [] if isinstance(f, dict)]
    bare = {str(t).split(".")[-1].lower() for t in raw if t}
    return {t for t in bare if t.startswith(("fct_", "agg_"))}


def _table_classes(taxonomy: dict[str, Any], table: str, schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    listed = sorted(
        code
        for code, spec in (taxonomy.get("classes") or {}).items()
        if isinstance(spec, dict) and table in _class_tables(spec)
    )
    labelled = sorted(str(c).upper() for c in schema.get("indicator_classes") or [])
    return listed, labelled


def build_ledger() -> dict[str, Any]:
    taxonomy = _yaml(CLASSES_PATH)
    registry = _yaml(REGISTRY_PATH)
    corpus_policy = _yaml(CORPUS_POLICY_PATH)
    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8")) if COVERAGE_PATH.is_file() else {}
    live = json.loads(LIVE_PATH.read_text(encoding="utf-8")) if LIVE_PATH.is_file() else {}
    parents = _lineage_parents()
    cards = {p.stem.upper(): _yaml(p) for p in sorted(CARDS_DIR.glob("*.yaml"))}

    objects: dict[str, Any] = {}
    disagreements: list[dict[str, Any]] = []
    for path in sorted(YAML_DIR.glob("*.yml")):
        table = path.stem.lower()
        schema = _yaml(path)
        listed, labelled = _table_classes(taxonomy, table, schema)
        classes = sorted(set(listed) | set(labelled))
        if listed != labelled and (listed or labelled) and not table.startswith(("dim_", "bridge_")):
            disagreements.append(
                {"table": table, "kind": "class_membership", "taxonomy": listed, "table_yaml": labelled}
            )
        disposition, reason = _disposition(table, classes)
        excluded: set[str] = set()
        for code in classes:
            excluded |= {str(c).lower() for c in (cards.get(code) or {}).get("exclude_prompt_columns") or []}
        roles = _column_roles(table, schema, excluded)
        entry: dict[str, Any] = {
            "disposition": disposition,
            "classes": classes,
            "lineage_parent": parents.get(table),
            "grain": str(schema.get("grain") or ""),
            "column_roles": roles,
        }
        if reason:
            entry["reason"] = reason
        if disposition != "serving":
            entry["readiness"] = "discovered"
            objects[table] = entry
            continue

        derived = _derived_operations(table, roles)
        declared = _declared_operations(registry, table)
        operations: dict[str, Any] = {}
        blockers: list[str] = []
        for op in OPERATIONS:
            possible, why_not = derived[op]
            if declared is None:
                operations[op] = {"supported": False, "reason": "not declared in capability_registry table_operations"}
                continue
            if op in (declared.get("supported") or []):
                if not possible:
                    blockers.append(f"{op} declared but table shape forbids it: {why_not}")
                operations[op] = {"supported": possible, "reason": "" if possible else why_not}
            else:
                note = str((declared.get("unsupported") or {}).get(op) or why_not or "not declared")
                operations[op] = {"supported": False, "reason": note}
        entry["operations"] = operations
        entry["coverage"] = _coverage_summary(coverage, table, parents.get(table))

        if declared is None:
            blockers.append("no table_operations declaration")
        if "measure" not in roles.values():
            blockers.append("no measure column")
        if blockers:
            entry["readiness"] = "blocked" if declared is not None else "discovered"
            entry["blockers"] = blockers
            objects[table] = entry
            continue
        ok, probe_err, probe_sql = _probe(table)
        entry["probe"] = {"ok": ok, "error": probe_err, "sql": probe_sql}
        if not ok:
            entry["readiness"] = "blocked"
            entry["blockers"] = [probe_err]
        elif (live.get("tables") or {}).get(table, {}).get("ok"):
            entry["readiness"] = "integration_verified"
        else:
            entry["readiness"] = "fixture_verified"
        objects[table] = entry

    class_rows: dict[str, Any] = {}
    policy_classes = corpus_policy.get("classes") or {}
    default_corpora = (corpus_policy.get("defaults") or {}).get("corpora") or []
    for code, spec in sorted((taxonomy.get("classes") or {}).items()):
        if not isinstance(spec, dict):
            continue
        serving = sorted(t for t, e in objects.items() if e["disposition"] == "serving" and code in e["classes"])
        missing = sorted(t for t in _class_tables(spec) if t not in objects)
        if missing:
            disagreements.append({"class": code, "kind": "missing_table_yaml", "tables": missing})
        card = cards.get(code) or {}
        card_tables = {str(t).split(".")[-1].lower() for t in card.get("tables") or []}
        card_default = str(card.get("default_table") or "").split(".")[-1].lower()
        if card_default and card_default not in serving:
            disagreements.append({"class": code, "kind": "card_default_not_serving", "table": card_default})
        states = [objects[t]["readiness"] for t in serving]
        class_rows[code] = {
            "name": spec.get("name"),
            "serving_tables": serving,
            "default_table": card_default,
            "card_tables": sorted(card_tables),
            "corpora": list((policy_classes.get(code) or {}).get("corpora") or default_corpora),
            "readiness": min_readiness(states) if serving else "blocked",
            "default_table_readiness": (objects.get(card_default) or {}).get("readiness", "blocked"),
        }

    return {
        "built_at": datetime.now(UTC).isoformat(),
        "snapshot_hash": snapshot_hash(),
        "coverage_built_at": coverage.get("built_at"),
        "classes": class_rows,
        "objects": objects,
        "disagreements": disagreements,
    }


def render_summary(ledger: dict[str, Any]) -> str:
    lines = [
        "# Coverage ledger summary",
        "",
        f"Snapshot `{ledger['snapshot_hash']}`, coverage built {ledger.get('coverage_built_at') or 'never'}.",
        "",
        "| Class | Readiness | Default table | Default readiness | Serving tables |",
        "|---|---|---|---|---|",
    ]
    for code, row in ledger["classes"].items():
        lines.append(
            f"| {code} | {row['readiness']} | {row['default_table'] or '-'} | "
            f"{row['default_table_readiness']} | {len(row['serving_tables'])} |"
        )
    blocked = {t: e for t, e in ledger["objects"].items() if e.get("readiness") == "blocked"}
    lines += ["", "## Blocked serving tables", ""]
    lines += [f"- `{t}`: {'; '.join(e.get('blockers') or [])}" for t, e in sorted(blocked.items())] or ["- none"]
    lines += ["", "## Unsupported mart objects", ""]
    lines += [
        f"- `{t}`: {e.get('reason')}"
        for t, e in sorted(ledger["objects"].items())
        if e["disposition"] == "unsupported"
    ] or ["- none"]
    lines += ["", "## Contract disagreements", ""]
    lines += [f"- {json.dumps(d, sort_keys=True)}" for d in ledger["disagreements"]] or ["- none"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    ledger = build_ledger()
    LEDGER_PATH.write_text(json.dumps(ledger, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(render_summary(ledger), encoding="utf-8")
    print(f"Wrote {LEDGER_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
