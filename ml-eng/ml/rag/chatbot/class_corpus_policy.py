"""YAML-driven vector corpus selection by indicator class."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_YAML_PATH = Path(__file__).resolve().parents[1] / "helpers" / "class_corpus_policy.yaml"


@lru_cache(maxsize=1)
def _load_policy() -> dict[str, Any]:
    if not _YAML_PATH.is_file():
        return {"defaults": {"corpora": ["public_reports", "academic_papers", "ota"]}, "classes": {}}
    data = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def corpora_for_class(class_code: str) -> list[str]:
    code = (class_code or "").strip().upper()
    classes = _load_policy().get("classes") or {}
    spec = classes.get(code) if isinstance(classes, dict) else None
    if isinstance(spec, dict) and spec.get("corpora"):
        return [str(c).strip() for c in spec["corpora"] if str(c).strip()]
    defaults = _load_policy().get("defaults") or {}
    return [str(c).strip() for c in (defaults.get("corpora") or ["public_reports", "academic_papers", "ota"]) if str(c).strip()]


def corpora_for_classes(
    primary: tuple[str, ...] | list[str],
    *,
    secondary: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Union corpora for routed classes; primary class order wins."""
    seen: set[str] = set()
    out: list[str] = []
    for code in (*primary, *secondary):
        for corp in corpora_for_class(str(code)):
            if corp not in seen:
                seen.add(corp)
                out.append(corp)
    return out


__all__ = ["corpora_for_class", "corpora_for_classes"]
