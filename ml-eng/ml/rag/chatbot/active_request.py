"""Active request: the resolved slots of a conversation, carried turn to turn.

Each slot records the turn that set it and how (asked in the question, carried from an
earlier turn, or an explicit correction). A follow-up keeps every slot it does not
restate; a restated slot overrides; a shift to another indicator clears the slots that
only apply to the old one (price type belongs to prices). Adapters persist the dict
between turns; the stateless chat_history path folds it from the user turns instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ml.rag.chat_history import normalize_messages
from ml.rag.chatbot.agri_entities import CROP_ENTITY_RE
from ml.rag.chatbot.assistant_identity import is_meta_query
from ml.rag.chatbot.geo_iso3 import countries_named_in_text, iso3_to_canonical_name, name_to_iso3
from ml.rag.chatbot.product_knowledge import is_help_query, is_product_query
from ml.rag.chatbot.query_gate import is_greeting_query

SlotSource = Literal["question", "carried", "correction"]
SLOT_NAMES: tuple[str, ...] = (
    "commodity",
    "geographies",
    "indicator",
    "price_type",
    "time_phrase",
    "operation",
)
# Slots that only mean something for one indicator; a shift away clears them.
_INDICATOR_SCOPED: dict[str, str] = {"price_type": "price"}

_PRICE_TYPE_RE = re.compile(r"\b(retail|wholesale|producer|farm[\s-]?gate|consumer|export|import)\b", re.I)
_PRICE_TYPE_CANON = {"farm gate": "producer", "farmgate": "producer", "farm-gate": "producer", "consumer": "retail"}
_INDICATOR_RE = re.compile(
    r"\b(prices?|production|produced|yields?|harvest(?:ed)?|area|exports?|imports?|trade|"
    r"rainfall|precipitation|temperature|food\s+insecurity|food\s+security|ipc|hunger|"
    r"employment|income|emissions|soil|fertili[sz]er|hdi)\b",
    re.I,
)
_INDICATOR_CANON = {
    "prices": "price",
    "produced": "production",
    "yields": "yield",
    "harvest": "production",
    "harvested": "production",
    "exports": "trade",
    "imports": "trade",
    "export": "trade",
    "import": "trade",
    "precipitation": "rainfall",
    "food insecurity": "food security",
    "ipc": "food security",
    "hunger": "food security",
    "fertiliser": "fertilizer",
}
_TIME_RE = re.compile(
    r"\b(now|currently|today|lately|recent(?:ly)?|latest|this\s+(?:year|month|season)|"
    r"last\s+(?:year|month|season)|(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2})?)\b",
    re.I,
)
_OPERATION_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|trend|over\s+time|rank|ranking|highest|lowest|top)\b", re.I)
_OPERATION_CANON = {
    "comparison": "compare",
    "versus": "compare",
    "vs": "compare",
    "vs.": "compare",
    "over time": "trend",
    "ranking": "rank",
    "highest": "rank",
    "lowest": "rank",
    "top": "rank",
}
_CORRECTION_RE = re.compile(
    r"\b(instead|actually|i\s+mean|i\s+meant|rather|not\s+\w+\s*,?\s*but)\b|^\s*no\b",
    re.I,
)
_ANAPHORA_RE = re.compile(
    r"\b(that\s+country|that\s+nation|that\s+place|those\s+countries|the\s+same\s+crop|that\s+crop|"
    r"same\s+(?:crop|commodity|country)|there|for\s+that|about\s+that|and\s+for\s+\w+)\b",
    re.I,
)
_ELLIPTICAL_RE = re.compile(
    r"^\s*(what\s+about\s+[\w\s]+|how\s+about\s+[\w\s]+|same\s+for\s+[\w\s]+|"
    r"country\s+is\s+\w+|and\s+(?:in\s+)?[\w\s]{2,40}|for\s+[\w\s]{2,40})\??\s*$",
    re.I,
)
_SUBJECT_OR_VERB_RE = re.compile(
    r"\b(i|my|we|our|he|she|it|they|his|her|their|is|are|was|were|am|has|have|had|do|does|did|"
    r"can|could|will|would|should|treat|cure|fix|help|sick|dying|dead|infected|injured|need|want|buy|sell)\b",
    re.I,
)


@dataclass(frozen=True)
class Slot:
    value: Any
    turn: int
    source: SlotSource

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "turn": self.turn, "source": self.source}


@dataclass
class ActiveRequest:
    slots: dict[str, Slot] = field(default_factory=dict)
    turn: int = 0
    state_version: int = 0
    pending_clarification: str | None = None

    def value(self, name: str) -> Any:
        slot = self.slots.get(name)
        return slot.value if slot else None

    def carried(self) -> dict[str, Any]:
        return {n: s.value for n, s in self.slots.items() if s.source == "carried"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": {n: s.to_dict() for n, s in self.slots.items()},
            "turn": self.turn,
            "state_version": self.state_version,
            "pending_clarification": self.pending_clarification,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ActiveRequest | None:
        if not isinstance(raw, dict):
            return None
        slots: dict[str, Slot] = {}
        for name, spec in (raw.get("slots") or {}).items():
            if name in SLOT_NAMES and isinstance(spec, dict) and spec.get("value") not in (None, "", []):
                source = spec.get("source") if spec.get("source") in ("question", "carried", "correction") else "carried"
                slots[name] = Slot(spec["value"], int(spec.get("turn") or 0), source)
        return cls(
            slots=slots,
            turn=int(raw.get("turn") or 0),
            state_version=int(raw.get("state_version") or 0),
            pending_clarification=raw.get("pending_clarification") or None,
        )


def _canon(value: str, table: dict[str, str]) -> str:
    low = re.sub(r"\s+", " ", value.strip().lower())
    return table.get(low, low)


def extract_turn_slots(text: str) -> dict[str, Any]:
    """Slots a single user turn states in its own words."""
    q = text or ""
    out: dict[str, Any] = {}
    commodities = [m.group(1).lower() for m in CROP_ENTITY_RE.finditer(q)]
    if commodities:
        out["commodity"] = list(dict.fromkeys(commodities))
    geos = sorted(
        {iso3_to_canonical_name(code) or code for code in (name_to_iso3(n) for n in countries_named_in_text(q)) if code}
    )
    if geos:
        out["geographies"] = geos
    if m := _INDICATOR_RE.search(q):
        out["indicator"] = _canon(m.group(1), _INDICATOR_CANON)
    if m := _PRICE_TYPE_RE.search(q):
        out["price_type"] = _canon(m.group(1), _PRICE_TYPE_CANON)
        out.setdefault("indicator", "price")
    if m := _TIME_RE.search(q):
        out["time_phrase"] = m.group(1).lower()
    if m := _OPERATION_RE.search(q):
        out["operation"] = _canon(m.group(1), _OPERATION_CANON)
    return out


def slots_from_decomposition(dec: dict[str, Any] | None) -> dict[str, Any]:
    """Slots the decomposer resolved for this turn (entities and places it understood)."""
    d = dec if isinstance(dec, dict) else {}
    out: dict[str, Any] = {}
    entities = [str(e).strip().lower() for e in d.get("entities") or [] if str(e).strip()]
    if entities:
        out["commodity"] = list(dict.fromkeys(entities))
    geos = [str(g).strip() for g in d.get("geography") or [] if str(g).strip()]
    if geos:
        out["geographies"] = geos
    return out


def is_follow_up(text: str, stated: dict[str, Any]) -> bool:
    """A turn continues the active request when it states a slot or leans on earlier turns."""
    q = (text or "").strip()
    if not q or is_greeting_query(q) or is_meta_query(q) or is_help_query(q) or is_product_query(q, None):
        return False
    if _ANAPHORA_RE.search(q) or _ELLIPTICAL_RE.match(q) or _CORRECTION_RE.search(q):
        return True
    return (
        len(q.split()) <= 4
        and not re.search(r"\b(what|which|how|why|when|show|give|compare|analy)\b", q, re.I)
        and not _SUBJECT_OR_VERB_RE.search(q)
    )


def add_decomposed_slots(active: ActiveRequest, dec: dict[str, Any] | None) -> None:
    """Record slots the decomposer understood that the turn text did not state or carry."""
    for name, value in slots_from_decomposition(dec).items():
        if name not in active.slots:
            active.slots[name] = Slot(value, active.turn, "question")


def merge_turn(prior: ActiveRequest | None, stated: dict[str, Any], *, text: str) -> ActiveRequest:
    """Fold one user turn into the active request. A turn naming its own commodity and place starts fresh."""
    base = prior or ActiveRequest()
    if "commodity" in stated and "geographies" in stated:
        base = ActiveRequest(turn=base.turn, state_version=base.state_version)
    turn = base.turn + 1
    correcting = bool(_CORRECTION_RE.search(text or ""))
    new_indicator = stated.get("indicator")
    slots: dict[str, Slot] = {}
    for name, slot in base.slots.items():
        scope = _INDICATOR_SCOPED.get(name)
        if scope and new_indicator and new_indicator != scope:
            continue
        slots[name] = Slot(slot.value, slot.turn, "carried")
    for name, value in stated.items():
        if name not in SLOT_NAMES or value in (None, "", []):
            continue
        overrides = name in base.slots and base.slots[name].value != value
        source: SlotSource = "correction" if overrides and correcting else "question"
        slots[name] = Slot(value, turn, source)
    return ActiveRequest(
        slots=slots,
        turn=turn,
        state_version=base.state_version,
        pending_clarification=None,
    )


def fold_history(recent_turns: list[dict[str, Any]] | None) -> ActiveRequest | None:
    """Rebuild the active request from user turns when no persisted state exists."""
    active: ActiveRequest | None = None
    for message in normalize_messages(recent_turns):
        if str(message.get("role") or "").lower() != "user":
            continue
        content = str(message.get("content") or "")
        stated = extract_turn_slots(content)
        if stated:
            active = merge_turn(active, stated, text=content)
    return active


def resolve_turn(
    user_query: str,
    *,
    persisted: dict[str, Any] | None,
    recent_turns: list[dict[str, Any]] | None,
) -> tuple[ActiveRequest, ActiveRequest]:
    """(slots for this turn, state to persist). An unrelated turn carries nothing and keeps prior state."""
    prior = ActiveRequest.from_dict(persisted) or fold_history(recent_turns)
    stated = extract_turn_slots(user_query)
    if not is_follow_up(user_query, stated):
        turn_only = merge_turn(None, stated, text=user_query)
        # A self-contained question becomes the active request. A social turn keeps the previous one.
        if stated:
            return turn_only, turn_only
        return turn_only, prior or ActiveRequest()
    merged = merge_turn(prior, stated, text=user_query)
    return merged, merged


_LABELS = {
    "commodity": "commodity",
    "geographies": "place",
    "indicator": "measure",
    "price_type": "price type",
    "time_phrase": "time",
    "operation": "operation",
}


def _fmt(value: Any) -> str:
    return ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)


def resolved_request_text(user_query: str, active: ActiveRequest) -> str:
    """The question plus the slots it carries, for decomposition and SQL binding."""
    carried = active.carried()
    if not carried:
        return ""
    parts = [f"{_LABELS[n]}: {_fmt(carried[n])}" for n in SLOT_NAMES if n in carried]
    return f"{user_query.strip()} (carried from earlier turns: {'; '.join(parts)})"


def fill_decomposition(dec: dict[str, Any], active: ActiveRequest) -> dict[str, Any]:
    """Fill facets the turn left empty from carried slots; stated facets always win."""
    out = dict(dec)
    carried = active.carried()
    if carried.get("commodity") and not out.get("entities"):
        out["entities"] = list(carried["commodity"])
    if carried.get("geographies") and not out.get("geography"):
        out["geography"] = list(carried["geographies"])
    if carried.get("price_type") and not out.get("price_type"):
        out["price_type"] = carried["price_type"]
    out["active_request"] = active.to_dict()
    return out


__all__ = [
    "ActiveRequest",
    "SLOT_NAMES",
    "add_decomposed_slots",
    "Slot",
    "extract_turn_slots",
    "fill_decomposition",
    "fold_history",
    "is_follow_up",
    "merge_turn",
    "resolve_turn",
    "resolved_request_text",
    "slots_from_decomposition",
]
