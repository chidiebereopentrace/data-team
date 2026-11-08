"""Coverage of the requested facets across warehouse rows and document passages.

A cell is supported when the facet text is in the evidence, absent when retrieval
succeeded without it, and unknown when that lane failed. One country in the text
does not cover another country the question asked for.
"""
from __future__ import annotations

from typing import Any

FacetStatus = str  # supported | partial | absent | unknown | incompatible


def _blob(items: Any) -> str:
    parts: list[str] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or ""))
    return "\n".join(parts).casefold()


def _names(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _lane_status(name: str, blob: str, *, failed: bool) -> FacetStatus:
    if name.casefold() in blob:
        return "supported"
    if failed or not blob.strip():
        return "unknown"
    return "absent"


def build_facet_matrix(state: dict[str, Any]) -> list[dict[str, str]]:
    """One cell per requested geography and entity."""
    dec = state.get("decomposition") if isinstance(state.get("decomposition"), dict) else {}
    warehouse = _blob(state.get("bq_results"))
    documents = _blob(state.get("reranked_context")) + "\n" + _blob(state.get("merged_context"))
    debug = state.get("bq_sql_debug")
    warehouse_failed = isinstance(debug, list) and bool(debug) and not warehouse.strip()
    cells: list[dict[str, str]] = []
    for geo in _names(dec.get("geography")):
        cells.append(
            {
                "facet": "geography",
                "name": geo,
                "status": _lane_status(geo, warehouse + "\n" + documents, failed=warehouse_failed and not documents.strip()),
            }
        )
    for entity in _names(dec.get("entities")):
        cells.append(
            {
                "facet": "entity",
                "name": entity,
                "status": _lane_status(entity, warehouse + "\n" + documents, failed=False),
            }
        )
    return cells


def missing_required_facets(state: dict[str, Any]) -> list[dict[str, str]]:
    """Geographies must all be present. Entities are missing only when none of several appear."""
    cells = build_facet_matrix(state)
    geos = [cell for cell in cells if cell["facet"] == "geography" and cell["status"] != "supported"]
    entities = [cell for cell in cells if cell["facet"] == "entity"]
    missing = list(geos)
    if len(entities) >= 2 and entities and all(cell["status"] != "supported" for cell in entities):
        missing.extend(entities)
    return missing


__all__ = ["build_facet_matrix", "missing_required_facets"]
