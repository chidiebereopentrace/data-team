"""Intent bundles: multi-measure turn handles (not a table map).

Tables resolve via agri_measure_ontology.get_measure().candidate_tables.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import yaml

_BUNDLES_PATH = Path(__file__).resolve().parents[1] / "helpers" / "intent_bundles.yaml"


@dataclass(frozen=True)
class IntentBundleSpec:
    id: str
    trigger_patterns: tuple[str, ...]
    required_measures: tuple[str, ...] = field(default_factory=tuple)
    optional_measures: tuple[str, ...] = field(default_factory=tuple)
    excluded_primary_measures: tuple[str, ...] = field(default_factory=tuple)
    requires_breakdown: tuple[str, ...] = field(default_factory=tuple)
    vector_only: bool = False


@dataclass(frozen=True)
class MatchedBundle:
    spec: IntentBundleSpec
    score: int = 1


@lru_cache(maxsize=1)
def _load_bundle_specs() -> tuple[IntentBundleSpec, ...]:
    if not _BUNDLES_PATH.is_file():
        return ()
    with _BUNDLES_PATH.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    specs: list[IntentBundleSpec] = []
    for entry in raw.get("bundles") or []:
        if not isinstance(entry, dict):
            continue
        bid = str(entry.get("id") or "").strip()
        if not bid:
            continue
        patterns = tuple(
            str(p).strip().lower()
            for p in (entry.get("trigger_patterns") or [])
            if str(p).strip()
        )
        specs.append(
            IntentBundleSpec(
                id=bid,
                trigger_patterns=patterns,
                required_measures=_tuple_str(entry.get("required_measures")),
                optional_measures=_tuple_str(entry.get("optional_measures")),
                excluded_primary_measures=_tuple_str(entry.get("excluded_primary_measures")),
                requires_breakdown=_tuple_str(entry.get("requires_breakdown")),
                vector_only=bool(entry.get("vector_only")),
            )
        )
    return tuple(specs)


def _tuple_str(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(x).strip().lower() for x in raw if str(x).strip())


def _match_blob(
    query: str,
    decomposition: dict[str, Any] | None,
    *,
    breakdown: Sequence[str] | None = None,
) -> str:
    dec = decomposition if isinstance(decomposition, dict) else {}
    parts = [query or ""]
    for key in ("entities", "domains"):
        val = dec.get(key)
        if isinstance(val, list):
            parts.extend(str(x) for x in val if str(x).strip())
    if breakdown:
        parts.extend(str(b) for b in breakdown if str(b).strip())
    return " ".join(parts).lower()


def _pattern_score(blob: str, pattern: str) -> int:
    p = pattern.strip().lower()
    if not p:
        return 0
    if " " in p or "-" in p:
        return 10 + len(p) if p in blob else 0
    if re.search(rf"\b{re.escape(p)}\b", blob):
        return 8 + min(len(p), 12)
    return 0


def match_intent_bundles(
    query: str,
    decomposition: dict[str, Any] | None = None,
    *,
    breakdown: Sequence[str] | None = None,
) -> tuple[MatchedBundle, ...]:
    """Return matched bundles sorted by score (deterministic checklist input)."""
    blob = _match_blob(query, decomposition, breakdown=breakdown)
    if not blob.strip():
        return ()
    bd = {str(b).strip().lower() for b in (breakdown or []) if str(b).strip()}
    matched: list[MatchedBundle] = []
    for spec in _load_bundle_specs():
        if spec.requires_breakdown and not all(r in bd for r in spec.requires_breakdown):
            continue
        score = max((_pattern_score(blob, p) for p in spec.trigger_patterns), default=0)
        if score > 0:
            matched.append(MatchedBundle(spec=spec, score=score))
    matched.sort(key=lambda m: (-m.score, m.spec.id))
    return tuple(matched)


def bundle_required_measures(bundles: tuple[MatchedBundle, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for mb in bundles:
        for mid in mb.spec.required_measures:
            if mid not in seen:
                seen.add(mid)
                out.append(mid)
    return tuple(out)


def bundles_block_primary(measure_id: str, bundles: tuple[MatchedBundle, ...]) -> bool:
    mid = (measure_id or "").strip().lower()
    if not mid:
        return False
    for mb in bundles:
        if mid in mb.spec.excluded_primary_measures:
            return True
    return False


def has_bundle(bundles: tuple[MatchedBundle, ...], bundle_id: str) -> bool:
    return any(mb.spec.id == bundle_id for mb in bundles)


__all__ = [
    "IntentBundleSpec",
    "MatchedBundle",
    "match_intent_bundles",
    "bundle_required_measures",
    "bundles_block_primary",
    "has_bundle",
]
