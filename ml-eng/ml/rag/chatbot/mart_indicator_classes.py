"""OpenTrace indicator class taxonomy for mart BQ table routing."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_YAML_PATH = Path(__file__).resolve().parents[1] / "helpers" / "mart_indicator_classes.yaml"


def _bare_table(table_id: str) -> str:
    return (table_id or "").strip().split(".")[-1].lower()


@lru_cache(maxsize=1)
def _load_taxonomy() -> dict[str, Any]:
    if not _YAML_PATH.is_file():
        return {"classes": {}}
    data = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"classes": {}}


def all_class_codes() -> list[str]:
    classes = _load_taxonomy().get("classes") or {}
    return sorted(str(k) for k in classes.keys())


def class_for_query(text: str) -> list[str]:
    """Return indicator class codes whose aliases match the query (ordered by match strength)."""
    q = (text or "").lower()
    if not q:
        return []
    scored: list[tuple[int, str]] = []
    classes = _load_taxonomy().get("classes") or {}
    for code, spec in classes.items():
        if not isinstance(spec, dict):
            continue
        best = 0
        for alias in spec.get("aliases") or []:
            al = str(alias).lower().strip()
            if not al:
                continue
            if " " in al:
                if al in q:
                    best = max(best, 10 + len(al))
            elif re.search(rf"\b{re.escape(al)}\b", q):
                best = max(best, 8 + min(len(al), 12))
        if best > 0:
            scored.append((best, str(code)))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [code for _, code in scored]


def _class_spec(code: str) -> dict[str, Any] | None:
    classes = _load_taxonomy().get("classes") or {}
    spec = classes.get(code.upper())
    return spec if isinstance(spec, dict) else None


def facts_for_class(code: str) -> list[str]:
    spec = _class_spec(code)
    if not spec:
        return []
    facts = list(spec.get("primary_facts") or [])
    facts.extend(spec.get("companion_facts") or [])
    # Dedupe preserving order; companion_facts may include dims
    seen: set[str] = set()
    out: list[str] = []
    for f in facts:
        bare = _bare_table(str(f))
        if bare and bare not in seen:
            seen.add(bare)
            out.append(bare)
    return out


def facts_for_classes(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for code in codes:
        for tid in facts_for_class(code):
            if tid not in seen:
                seen.add(tid)
                out.append(tid)
    return out


def families_for_fact(table_id: str) -> list[dict[str, Any]]:
    bare = _bare_table(table_id)
    out: list[dict[str, Any]] = []
    classes = _load_taxonomy().get("classes") or {}
    for code, spec in classes.items():
        if not isinstance(spec, dict):
            continue
        for fam in spec.get("families") or []:
            if not isinstance(fam, dict):
                continue
            fam_table = str(fam.get("table") or "").strip()
            if fam_table and _bare_table(fam_table) != bare:
                continue
            if not fam_table and bare not in facts_for_class(str(code)):
                continue
            entry = dict(fam)
            entry["class_code"] = str(code)
            out.append(entry)
    return out


def indicator_classes_for_table(table_id: str) -> list[str]:
    bare = _bare_table(table_id)
    codes: list[str] = []
    classes = _load_taxonomy().get("classes") or {}
    for code, spec in classes.items():
        if not isinstance(spec, dict):
            continue
        primary = {_bare_table(str(t)) for t in (spec.get("primary_facts") or [])}
        if bare in primary:
            codes.append(str(code))
    return codes


def do_not_mix_tables(table_a: str, table_b: str) -> str | None:
    a, b = _bare_table(table_a), _bare_table(table_b)
    if not a or not b or a == b:
        return None
    classes = _load_taxonomy().get("classes") or {}
    for spec in classes.values():
        if not isinstance(spec, dict):
            continue
        for rule in spec.get("do_not_mix") or []:
            if not isinstance(rule, dict):
                continue
            pair = {_bare_table(str(t)) for t in (rule.get("tables") or [])}
            if a in pair and b in pair:
                return str(rule.get("reason") or "do_not_mix")
    # Cross-class PROD yield vs production
    if {a, b} == {"fct_production", "fct_yield"}:
        return "FAOSTAT country-year production vs FNID-season yield — never mix"
    return None


def family_do_not_mix(family_id: str, other_family_id: str) -> bool:
    classes = _load_taxonomy().get("classes") or {}
    for spec in classes.values():
        if not isinstance(spec, dict):
            continue
        for fam in spec.get("families") or []:
            if not isinstance(fam, dict):
                continue
            if str(fam.get("id")) != family_id:
                continue
            blocked = {str(x) for x in (fam.get("do_not_mix_with") or [])}
            return other_family_id in blocked
    return False


__all__ = [
    "all_class_codes",
    "class_for_query",
    "do_not_mix_tables",
    "facts_for_class",
    "facts_for_classes",
    "families_for_fact",
    "family_do_not_mix",
    "indicator_classes_for_table",
]
