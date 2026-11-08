"""Parsed BigQuery SQL checks: read-only single SELECT, permitted functions, scoped columns.

Every check works on the sqlglot AST, so table aliases, CTE names and derived-table
outputs are resolved by scope instead of being mistaken for column names.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import Scope, traverse_scope

_DIALECT = "bigquery"

_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Export,
    exp.Into,
)
_FORBIDDEN_FUNCS: tuple[type[exp.Expression], ...] = (
    exp.Predict,
    exp.SessionUser,
)
# Functions sqlglot does not model for BigQuery and that are safe in read-only analytics.
_ANONYMOUS_ALLOWED = frozenset({"SAFE_OFFSET", "SAFE_ORDINAL", "OFFSET", "ORDINAL"})


@dataclass(frozen=True)
class TableRef:
    project: str
    dataset: str
    name: str


def parse_select(sql: str) -> tuple[exp.Expression | None, str | None]:
    """Parse exactly one read-only query; return (ast, error)."""
    try:
        statements = [s for s in sqlglot.parse(sql or "", read=_DIALECT) if s is not None]
    except ParseError as exc:
        return None, f"sql_parse_failed: {str(exc).splitlines()[0][:200]}"
    if len(statements) != 1:
        return None, "SQL must be exactly one statement"
    root = statements[0]
    if not isinstance(root, exp.Query):
        return None, f"SQL must be a read-only query, not {type(root).__name__.upper()}"
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return None, f"SQL contains a forbidden {type(node).__name__.upper()} clause"
        if isinstance(node, _FORBIDDEN_FUNCS):
            return None, f"SQL calls a forbidden function {type(node).__name__.upper()}"
        if isinstance(node, exp.Anonymous) and str(node.name).upper() not in _ANONYMOUS_ALLOWED:
            return None, f"SQL calls a function that is not permitted: {node.name}"
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return None, "SQL uses a table-valued function, which is not permitted"
        if "information_schema" in f"{table.db}.{table.name}".lower():
            return None, "SQL reads INFORMATION_SCHEMA, which is not permitted"
    return root, None


def physical_tables(root: exp.Expression) -> list[TableRef]:
    """Warehouse tables the statement reads, excluding CTE references."""
    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    out: list[TableRef] = []
    for table in root.find_all(exp.Table):
        if not table.db and table.name.lower() in cte_names:
            continue
        ref = TableRef(project=table.catalog, dataset=table.db, name=table.name.lower())
        if ref not in out:
            out.append(ref)
    return out


def _scope_source_columns(
    scope: Scope,
    columns_for_table: Callable[[str], set[str] | None],
) -> tuple[dict[str, set[str] | None], set[str] | None]:
    """Columns visible per source alias; ``None`` means unknown (no YAML), so unchecked."""
    by_alias: dict[str, set[str] | None] = {}
    for alias, source in scope.sources.items():
        if isinstance(source, exp.Table):
            known = columns_for_table(source.name.lower())
            by_alias[alias.lower()] = {c.lower() for c in known} if known else None
        elif isinstance(source, Scope):
            selects = source.expression.named_selects if isinstance(source.expression, exp.Query) else []
            if isinstance(source.expression, exp.Query) and source.expression.is_star:
                by_alias[alias.lower()] = None
            else:
                by_alias[alias.lower()] = {s.lower() for s in selects}
        else:
            by_alias[alias.lower()] = None
    if any(v is None for v in by_alias.values()):
        return by_alias, None
    union: set[str] = set()
    for cols in by_alias.values():
        union |= cols or set()
    return by_alias, union


def unknown_columns(
    root: exp.Expression,
    columns_for_table: Callable[[str], set[str] | None],
) -> list[str]:
    """Column references that no source in their scope exposes."""
    bad: list[str] = []
    for scope in traverse_scope(root):
        by_alias, union = _scope_source_columns(scope, columns_for_table)
        projections = {
            sel.alias.lower()
            for sel in getattr(scope.expression, "selects", [])
            if isinstance(sel, exp.Alias)
        }
        own_select = scope.expression if isinstance(scope.expression, exp.Select) else None
        for column in scope.columns:
            if own_select is not None and column.find_ancestor(exp.Select) is not own_select:
                continue
            name = column.name.lower()
            if not name or isinstance(column.this, exp.Star):
                continue
            qualifier = column.table.lower()
            if qualifier:
                if qualifier not in by_alias:
                    bad.append(f"{column.table}.{column.name}")
                    continue
                visible = by_alias[qualifier]
                if visible is not None and name not in visible:
                    bad.append(f"{column.table}.{column.name}")
                continue
            if union is None or name in union or name in projections:
                continue
            bad.append(column.name)
    seen: set[str] = set()
    out: list[str] = []
    for item in bad:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


__all__ = ["TableRef", "parse_select", "physical_tables", "unknown_columns"]
