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


def taxonomy_version() -> str:
    return str(_load_taxonomy().get("version") or "")


def all_class_codes() -> list[str]:
    classes = _load_taxonomy().get("classes") or {}
    return sorted(str(k) for k in classes.keys())


def class_for_query(text: str) -> list[str]:
    """Return indicator class codes whose aliases or family terms match the query."""
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
        best = max(best, class_family_score(str(code), q))
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


def companion_facts_for_class(code: str) -> list[str]:
    """Mart fact/agg companion_facts only (skip dims/bridges)."""
    spec = _class_spec(code)
    if not spec:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in spec.get("companion_facts") or []:
        bare = _bare_table(str(raw))
        if not bare or bare in seen:
            continue
        if bare.startswith(("dim_", "bridge_", "gold_dim_")):
            continue
        if not (bare.startswith("fct_") or bare.startswith("agg_")):
            continue
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


def families_for_class(code: str) -> list[dict[str, Any]]:
    spec = _class_spec(code)
    if not spec:
        return []
    return [f for f in (spec.get("families") or []) if isinstance(f, dict)]


_GRAIN_TOKENS = frozenset(
    {
        "agg",
        "area",
        "country",
        "fct",
        "geo",
        "grain",
        "item",
        "key",
        "month",
        "national",
        "panel",
        "point",
        "product",
        "share",
        "source",
        "stock",
        "time",
        "year",
    }
)
_GLUE_TOKENS = frozenset({"and", "by", "for", "not", "of", "per", "the", "via", "vs"})
_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {}


def _token_in_query(token: str, query: str) -> bool:
    pattern = _TOKEN_PATTERNS.get(token)
    if pattern is None:
        body = "off[-\\s]?farm" if token == "offfarm" else re.escape(token)
        pattern = re.compile(rf"\b{body}\b", re.IGNORECASE)
        _TOKEN_PATTERNS[token] = pattern
    return bool(pattern.search(query or ""))


def family_declared_tokens(family: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Subject tokens select the family. Grain tokens only raise an already matching family."""
    raw = str(family.get("id") or "").lower().split("_")
    tokens = [token for token in raw if token not in _GLUE_TOKENS and len(token) >= 3]
    subjects = [token for token in tokens if token not in _GRAIN_TOKENS]
    grain = [token for token in tokens if token in _GRAIN_TOKENS]
    return subjects, grain


def family_match_score(family: dict[str, Any], query: str) -> int:
    """Score a family from its own id. Zero unless every subject token is in the question."""
    if not str(family.get("table") or "").strip():
        return 0
    subjects, grain = family_declared_tokens(family)
    if subjects:
        if not all(_token_in_query(token, query) for token in subjects):
            return 0
        score = 10 * len(subjects)
        if len(subjects) > 1:
            score += 20
    elif grain and all(_token_in_query(token, query) for token in grain):
        score = 60
    else:
        return 0
    if subjects and grain and all(_token_in_query(token, query) for token in grain):
        score += 50
    return score


def family_question_phrase(family: dict[str, Any]) -> str:
    subjects, grain = family_declared_tokens(family)
    return " ".join([*subjects, *grain])


def class_family_score(code: str, query: str) -> int:
    """Best subject-bearing family score. Grain-only families do not select a class."""
    best = 0
    for family in families_for_class(code):
        subjects, _grain = family_declared_tokens(family)
        if not subjects:
            continue
        best = max(best, family_match_score(family, query))
    return best


def best_declared_family(code: str, query: str) -> tuple[dict[str, Any], int] | None:
    """Highest-scoring family on this class. The earlier declaration wins a tie."""
    best: dict[str, Any] | None = None
    best_score = 0
    for family in families_for_class(code):
        score = family_match_score(family, query)
        if score > best_score:
            best = family
            best_score = score
    if best is None:
        return None
    return best, best_score


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
    "best_declared_family",
    "class_family_score",
    "class_for_query",
    "companion_facts_for_class",
    "do_not_mix_tables",
    "family_declared_tokens",
    "family_match_score",
    "family_question_phrase",
    "facts_for_class",
    "facts_for_classes",
    "families_for_class",
    "families_for_fact",
    "family_do_not_mix",
    "indicator_classes_for_table",
]
