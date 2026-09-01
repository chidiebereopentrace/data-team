"""Unified slot validation for TurnContract compilation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ml.rag.chatbot.agri_entities import CROP_ENTITY_RE, query_has_crop_or_commodity
from ml.rag.chatbot.agri_measure_ontology import MeasureSpec
from ml.rag.chatbot.continental_scope import decomposition_has_africa_scope
from ml.rag.chatbot.geo_regions import detect_regions_in_text
from ml.rag.chatbot.turn_contract import TurnContract

SlotOutcome = Literal["served", "clarify", "unsupported"]


@dataclass(frozen=True)
class SlotValidation:
    outcome: SlotOutcome
    reason: str = ""


def _geo_list(decomposition: dict[str, Any] | None) -> list[str]:
    if not isinstance(decomposition, dict):
        return []
    geo = decomposition.get("geography")
    if not isinstance(geo, list):
        return []
    return [str(g).strip() for g in geo if str(g).strip()]


def geography_satisfied(
    *,
    query: str,
    decomposition: dict[str, Any] | None,
    contract: TurnContract,
    measure: MeasureSpec | None,
) -> bool:
    if measure is not None and measure.country_is_answer:
        return True
    if measure is not None and not measure.geography_required:
        return True
    if contract.geo:
        return True
    if contract.geo_grain in ("africa", "region"):
        return True
    if decomposition_has_africa_scope(decomposition):
        return True
    if _geo_list(decomposition):
        return True
    if detect_regions_in_text(query or ""):
        return True
    return False


def crop_satisfied(
    *,
    query: str,
    decomposition: dict[str, Any] | None,
    contract: TurnContract,
    measure: MeasureSpec | None,
    multi_measure_panel: bool = False,
) -> bool:
    if measure is None or not measure.crop_required:
        return True
    if contract.entities and multi_measure_panel:
        return True
    if query_has_crop_or_commodity(query, decomposition):
        return True
    if CROP_ENTITY_RE.search(query or ""):
        return True
    return bool(contract.entities)


def validate_turn_slots(
    contract: TurnContract,
    measure: MeasureSpec | None,
    *,
    query: str = "",
    decomposition: dict[str, Any] | None = None,
    multi_measure_panel: bool = False,
) -> SlotValidation:
    if contract.job in ("help", "social", "clarify"):
        return SlotValidation("clarify", contract.job)
    if measure is None and contract.job in ("fact", "trend", "rank", "list", "compare"):
        return SlotValidation("clarify", "measure_unresolved")
    if measure is not None:
        if not crop_satisfied(
            query=query,
            decomposition=decomposition,
            contract=contract,
            measure=measure,
            multi_measure_panel=multi_measure_panel,
        ):
            return SlotValidation("clarify", "crop_required")
        if not geography_satisfied(
            query=query,
            decomposition=decomposition,
            contract=contract,
            measure=measure,
        ):
            return SlotValidation("clarify", "geography_required")
    return SlotValidation("served")


__all__ = [
    "SlotOutcome",
    "SlotValidation",
    "crop_satisfied",
    "geography_satisfied",
    "validate_turn_slots",
]
