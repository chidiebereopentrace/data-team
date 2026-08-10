"""Per-table YAML loader for BigQuery semantic schemas under ``ml/rag/bq_tables_yaml_files``.

Staging_dev ``stg_*`` YAMLs are the sole table catalog for Ask ADZA NL-to-SQL
(no Qdrant table-description matching). Each YAML carries grain, keys, hints,
and columns for the SQL reasoner and NL-to-SQL prompts.

Public API:
- ``load_table_schema(name)``       -> raw dict for the table, or None.
- ``format_table_schema(name, ...)`` -> compact SQL-prompt block string, or "".
- ``known_table_names()``           -> set[str] of all indexed names (bare + FQN).
- ``list_staging_table_index()``    -> compact index rows for the SQL reasoner.
- ``pack_selected_table_hints(...)`` -> byte-capped full YAML packs for NL2SQL.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ml.rag.chatbot.bq_byte_budget import hint_max_bytes, pack_lines, reasoner_index_max_bytes, truncate_utf8, utf8_len
from ml.rag.helpers.staging_semantic_relationships import compact_rels_summary

# bq_table_schema_yaml.py lives at ml/rag/chatbot/, YAMLs live at ml/rag/bq_tables_yaml_files/.
_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "bq_tables_yaml_files"

# Cache: (cache_key, index)
_cache: tuple[tuple[Any, ...], dict[str, dict[str, Any]]] | None = None


def _yaml_dir() -> Path:
    raw = os.environ.get("RAG_BQ_TABLES_YAML_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_DIR.resolve()


def _strip_fqn(table_name: str) -> str:
    """Return the bare table name (last dotted segment) from a possibly fully-qualified id."""
    text = (table_name or "").strip().strip("`")
    if not text:
        return ""
    return text.split(".")[-1]


def _index_yaml_files(directory: Path) -> dict[str, dict[str, Any]]:
    """Build name -> table_schema_dict index, keyed by both bare and fully-qualified names."""
    out: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return out
    for path in directory.glob("*.yml"):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        bare = path.stem
        # File-name key is authoritative; the explicit table_name field (often a FQN) is an alias.
        out[bare] = data
        declared = str(data.get("table_name") or "").strip().strip("`")
        if declared:
            out[declared] = data
            short = _strip_fqn(declared)
            if short and short != bare:
                out[short] = data
    return out


def _build_index() -> dict[str, dict[str, Any]]:
    """Load all YAML files; cache by (dir_path, dir_mtime, sorted file mtimes)."""
    global _cache
    directory = _yaml_dir()
    try:
        dir_mtime = directory.stat().st_mtime_ns if directory.is_dir() else None
    except OSError:
        dir_mtime = None
    file_sig: tuple[tuple[str, int], ...] = tuple()
    if dir_mtime is not None and directory.is_dir():
        try:
            file_sig = tuple(
                sorted(
                    (p.name, p.stat().st_mtime_ns)
                    for p in directory.glob("*.yml")
                )
            )
        except OSError:
            file_sig = tuple()
    cache_key = ("v1", str(directory), dir_mtime, file_sig)
    if _cache is not None and _cache[0] == cache_key:
        return _cache[1]
    index = _index_yaml_files(directory)
    _cache = (cache_key, index)
    return index


def known_table_names() -> set[str]:
    """All names (bare + FQN aliases) for which a YAML schema is available."""
    return set(_build_index().keys())


def load_table_schema(table_name: str) -> dict[str, Any] | None:
    """Resolve a table name to its raw YAML dict (trying FQN, bare, file-stem)."""
    name = (table_name or "").strip()
    if not name:
        return None
    index = _build_index()
    if name in index:
        return index[name]
    bare = _strip_fqn(name)
    if bare and bare in index:
        return index[bare]
    return None


# --- formatting -------------------------------------------------------------

_MAX_LINE = 140
_MAX_COLUMNS = 30


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _format_list_field(label: str, value: Any) -> str | None:
    """Render scalar/list/dict YAML node as a single compact line."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        if not text:
            return None
        return f"{label}: {_truncate(text, _MAX_LINE)}"
    if isinstance(value, list):
        flat: list[str] = []
        for item in value:
            if isinstance(item, (str, int, float, bool)):
                flat.append(str(item).strip())
            elif isinstance(item, dict):
                pair = next(iter(item.items()), None)
                if pair is not None:
                    k, v = pair
                    flat.append(f"{k}={_truncate(str(v), 60)}")
            if not flat:
                continue
        if not flat:
            return None
        return f"{label}: " + _truncate(", ".join(filter(None, flat)), _MAX_LINE)
    if isinstance(value, dict):
        bits: list[str] = []
        for k, v in value.items():
            if isinstance(v, (str, int, float, bool)):
                bits.append(f"{k}={_truncate(str(v), 60)}")
            elif isinstance(v, list):
                inner = ", ".join(str(x) for x in v if isinstance(x, (str, int, float, bool)))
                if inner:
                    bits.append(f"{k}=[{_truncate(inner, 80)}]")
        if not bits:
            return None
        return f"{label}: " + _truncate("; ".join(bits), _MAX_LINE)
    return None


def _format_columns(columns: Any, *, max_columns: int = _MAX_COLUMNS) -> str:
    """Render a YAML columns list as `name (type, role): description` lines."""
    if not isinstance(columns, list):
        return ""
    lines: list[str] = []
    for col in columns[:max_columns]:
        if not isinstance(col, dict):
            continue
        name = str(col.get("name") or "").strip()
        if not name:
            continue
        typ = str(col.get("type") or "").strip()
        role = str(col.get("semantic_role") or "").strip()
        desc = str(col.get("description") or "").strip()
        example = col.get("example")
        head = name
        meta_bits = []
        if typ:
            meta_bits.append(typ)
        if role:
            meta_bits.append(role)
        if meta_bits:
            head = f"{name} ({', '.join(meta_bits)})"
        tail = desc
        if example not in (None, ""):
            ex = _truncate(str(example), 40)
            tail = f"{tail} [ex: {ex}]" if tail else f"[ex: {ex}]"
        line = f"  - {head}" + (f": {_truncate(tail, _MAX_LINE - len(head) - 4)}" if tail else "")
        lines.append(line)
    if isinstance(columns, list) and len(columns) > max_columns:
        lines.append(f"  - … {len(columns) - max_columns} more columns")
    return "\n".join(lines)


# Section ordering optimized for NL-to-SQL prompt usefulness.
_SECTION_ORDER: list[tuple[str, str]] = [
    ("description", "Description"),
    ("grain", "Grain"),
    ("primary_keys", "Primary keys"),
    ("relationships", "Relationships"),
    ("semantic_relationships", "Semantic relationships"),
    ("join_logic", "Join logic"),
    ("time_dimensions", "Time dimensions"),
    ("geography", "Geography columns"),
    ("metrics", "Metric columns"),
    ("scenario_context", "Scenario context"),
    ("semantic_role", "Semantic role"),
    ("business_questions_supported", "Business questions supported"),
    ("aggregation_rules", "Aggregation rules"),
    ("filtering_guidance", "Filtering guidance"),
    ("sql_generation_hints", "SQL generation hints"),
    ("data_quality", "Data quality"),
    ("temporal_model", "Temporal model"),
]


def _format_semantic_relationships(value: Any) -> str | None:
    """Compact multi-table relationship block for NL2SQL / reasoner packs."""
    if not isinstance(value, dict):
        return None
    lines: list[str] = ["Semantic relationships:"]
    joins = value.get("joins_with")
    if isinstance(joins, list) and joins:
        lines.append("  joins_with:")
        for item in joins[:8]:
            if not isinstance(item, dict):
                continue
            table = str(item.get("table") or "").strip()
            if not table:
                continue
            on = item.get("on")
            on_s = ",".join(str(x) for x in on) if isinstance(on, list) else str(on or "")
            how = str(item.get("how") or "").strip()
            note = str(item.get("note") or "").strip()
            lines.append(
                f"    - {table} on=[{on_s}] how={how}" + (f" ({note})" if note else "")
            )
    comps = value.get("companions")
    if isinstance(comps, list) and comps:
        lines.append("  companions:")
        for item in comps[:6]:
            if not isinstance(item, dict):
                continue
            table = str(item.get("table") or "").strip()
            when = str(item.get("when") or "").strip()
            role = str(item.get("role") or "").strip()
            if table:
                lines.append(f"    - {table} when={when}" + (f" role={role}" if role else ""))
    avoid = value.get("do_not_join")
    if isinstance(avoid, list) and avoid:
        lines.append("  do_not_join:")
        for item in avoid[:6]:
            if not isinstance(item, dict):
                continue
            table = str(item.get("table") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if table:
                lines.append(f"    - {table}: {reason}" if reason else f"    - {table}")
    return "\n".join(lines) if len(lines) > 1 else None


def format_table_schema(
    table_name: str,
    *,
    max_chars: int = 2400,
    max_bytes: int | None = None,
    include_columns: bool = True,
) -> str:
    """Compact, SQL-prompt-friendly rendering of a per-table YAML schema.

    Returns "" when no YAML is known for the table. Output is bounded by
    ``max_bytes`` (preferred) or ``max_chars``.
    """
    schema = load_table_schema(table_name)
    if not schema:
        return ""

    fqn = str(schema.get("table_name") or table_name).strip().strip("`")
    header = f"Table: {fqn or table_name}"
    parts: list[str] = [header]

    for key, label in _SECTION_ORDER:
        if key not in schema:
            continue
        if key == "semantic_relationships":
            block = _format_semantic_relationships(schema[key])
            if block:
                parts.append(block)
            continue
        line = _format_list_field(label, schema[key])
        if line:
            parts.append(line)

    if include_columns and isinstance(schema.get("columns"), list):
        col_block = _format_columns(schema["columns"])
        if col_block:
            parts.append("Columns:")
            parts.append(col_block)

    text = "\n".join(parts)
    budget = max_bytes if max_bytes is not None else max_chars
    if budget <= 0:
        return ""
    if max_bytes is not None:
        out, _ = truncate_utf8(text, budget)
        return out
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 1)].rstrip() + "…"


def list_staging_table_index() -> list[dict[str, Any]]:
    """Compact catalog for the SQL reasoner (one row per unique ``stg_*`` YAML)."""
    index = _build_index()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key, schema in index.items():
        if not isinstance(schema, dict):
            continue
        fqn = str(schema.get("table_name") or "").strip().strip("`")
        bare = _strip_fqn(fqn) or (key if key.startswith("stg_") else "")
        if not bare.startswith("stg_") or bare in seen:
            continue
        seen.add(bare)
        raw_role = schema.get("semantic_role")
        role: dict[str, Any] = raw_role if isinstance(raw_role, dict) else {}
        raw_tags = role.get("supports")
        tags: list[Any] = raw_tags if isinstance(raw_tags, list) else []
        raw_source = schema.get("source")
        source: dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
        domain = str(
            role.get("primary_domain")
            or source.get("domain")
            or schema.get("entity_type")
            or ""
        ).strip()
        rows.append(
            {
                "table_id": bare,
                "fqn": fqn or bare,
                "description": str(schema.get("description") or "").strip(),
                "grain": str(schema.get("grain") or "").strip(),
                "domain": domain,
                "tags": [str(t).strip() for t in tags if str(t).strip()],
                "rels": compact_rels_summary(bare),
            }
        )
    rows.sort(key=lambda r: r["table_id"])
    return rows


def format_reasoner_index(*, max_bytes: int | None = None) -> tuple[str, bool]:
    """Byte-capped one-line-per-table index for the SQL reasoner prompt."""
    budget = reasoner_index_max_bytes() if max_bytes is None else max(0, max_bytes)
    lines: list[str] = []
    for row in list_staging_table_index():
        tags = ", ".join(row.get("tags") or [])
        desc = str(row.get("description") or "")
        if len(desc) > 120:
            desc = desc[:119].rstrip() + "…"
        rels = str(row.get("rels") or compact_rels_summary(str(row["table_id"])))
        if len(rels) > 140:
            rels = rels[:139].rstrip() + "…"
        lines.append(
            f"- {row['table_id']} | domain={row.get('domain') or '-'} | "
            f"grain={row.get('grain') or '-'} | tags={tags or '-'} | "
            f"rels={rels} | {desc}"
        )
    return pack_lines(lines, budget)


def pack_selected_table_hints(
    table_ids: list[str],
    *,
    max_bytes: int | None = None,
) -> tuple[list[str], bool]:
    """Full YAML packs for selected tables, truncated to the NL2SQL hint byte budget."""
    budget = hint_max_bytes() if max_bytes is None else max(0, max_bytes)
    if budget <= 0 or not table_ids:
        return [], bool(table_ids)
    per = max(400, budget // max(1, len(table_ids)))
    hints: list[str] = []
    used = 0
    truncated = False
    known_count = 0
    for tid in table_ids:
        if not load_table_schema(tid):
            continue
        known_count += 1
        if used > 0:
            remain_total = budget - used - 1  # newline separator when joined
        else:
            remain_total = budget - used
        if remain_total <= 0:
            truncated = True
            break
        block = format_table_schema(tid, max_bytes=min(per, remain_total), include_columns=True)
        if not block:
            continue
        cost = utf8_len(block) + (1 if hints else 0)
        if used + cost > budget:
            frag, _ = truncate_utf8(block, remain_total)
            if frag:
                hints.append(frag)
            truncated = True
            break
        hints.append(block)
        used += cost
    return hints, truncated or len(hints) < known_count
