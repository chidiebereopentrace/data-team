"""Convert known weight units on warehouse price rows before generation.

Currency is never converted. Documents are not rewritten. ``common_unit_price``
is not the source — only ``avg_price`` / ``min_price`` / ``max_price`` and ``unit``.
"""
from __future__ import annotations

import re
from typing import Any

_ASKED_UNIT_RE = re.compile(
    r"\b(?:per|/)\s*(?:a\s+)?(?P<pack>\d+(?:\.\d+)?\s*)?(?P<unit>kg|kilograms?|g|grams?|t|tonnes?|tons?)\b",
    re.IGNORECASE,
)
_PACK_ASK_RE = re.compile(
    r"\b(?P<pack>\d+(?:\.\d+)?)\s*(?P<unit>kg|kilograms?)\s+(?:bag|sack|pack)?\b",
    re.IGNORECASE,
)
_ROW_UNIT_RE = re.compile(
    r"^(?P<pack>\d+(?:\.\d+)?\s*)?(?P<unit>kg|kilogram|g|gram|t|tonne|ton)\b",
    re.IGNORECASE,
)
_PRICE_KEYS = ("avg_price", "min_price", "max_price", "value", "price")
_TO_KG = {
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "t": 1000.0,
    "ton": 1000.0,
    "tonne": 1000.0,
    "tonnes": 1000.0,
    "tons": 1000.0,
}
_CANON = {
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "t": "t",
    "ton": "t",
    "tonne": "t",
    "tonnes": "t",
    "tons": "t",
}


def requested_weight_unit(query: str) -> tuple[float, str] | None:
    """Pack size and unit the question asks for (e.g. 1 kg, 50 kg)."""
    text = query or ""
    match = _ASKED_UNIT_RE.search(text) or _PACK_ASK_RE.search(text)
    if not match:
        return None
    unit = _CANON.get(str(match.group("unit") or "").casefold())
    if not unit:
        return None
    pack_raw = match.groupdict().get("pack")
    pack = float(str(pack_raw).strip()) if pack_raw and str(pack_raw).strip() else 1.0
    return pack, unit


def _row_weight(unit_text: str) -> tuple[float, str] | None:
    text = (unit_text or "").strip()
    if not text:
        return None
    match = _ROW_UNIT_RE.match(text)
    if not match:
        return None
    unit = _CANON.get(str(match.group("unit") or "").casefold())
    if not unit:
        return None
    pack_raw = match.group("pack")
    pack = float(str(pack_raw).strip()) if pack_raw and str(pack_raw).strip() else 1.0
    return pack, unit


def _kg_mass(pack: float, unit: str) -> float | None:
    factor = _TO_KG.get(unit)
    if factor is None:
        return None
    return pack * factor


def convert_price_row(row: dict[str, Any], *, query: str) -> dict[str, Any]:
    """Rewrite one warehouse price row onto the asked weight unit when both are known."""
    asked = requested_weight_unit(query)
    out = dict(row)
    unit_text = str(out.get("unit") or "").strip()
    held = _row_weight(unit_text)
    if asked is None:
        return out
    ask_pack, ask_unit = asked
    ask_kg = _kg_mass(ask_pack, ask_unit)
    if ask_kg is None or ask_kg <= 0:
        return out
    if held is None:
        out["unit_note"] = f"asked unit {ask_pack:g} {ask_unit} is not in the row"
        return out
    held_pack, held_unit = held
    held_kg = _kg_mass(held_pack, held_unit)
    if held_kg is None or held_kg <= 0:
        out["unit_note"] = f"asked unit {ask_pack:g} {ask_unit} is not in the row"
        return out
    if abs(held_kg - ask_kg) < 1e-12:
        return out
    scale = held_kg / ask_kg
    for key in _PRICE_KEYS:
        if key in out and isinstance(out[key], (int, float)):
            out[key] = float(out[key]) / scale
    target = f"{ask_pack:g} {ask_unit}" if ask_pack != 1 else ask_unit
    source = f"{held_pack:g} {held_unit}" if held_pack != 1 else held_unit
    out["unit"] = target if ask_pack != 1 else ask_unit
    out["converted_from"] = source
    out["unit_note"] = f"converted from {source}"
    return out


def convert_price_rows(rows: list[dict[str, Any]], *, query: str) -> list[dict[str, Any]]:
    return [convert_price_row(row, query=query) for row in rows]


__all__ = [
    "convert_price_row",
    "convert_price_rows",
    "requested_weight_unit",
]
