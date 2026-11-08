"""Three query views: user_query, context_rewrite, detail_rewrite.

user_query is never replaced as the sole embed string.
context_rewrite: related history + profile scope hints (vector + BQ/decompose).
detail_rewrite: deterministic enrichment twin (vector only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ml.rag.chatbot.active_request import ActiveRequest, resolve_turn, resolved_request_text
from ml.rag.chatbot.query_normalize import normalize_query_text

_ENTITY_BAG_MAX_TOKENS = 6
_LIVESTOCK_RE = re.compile(
    r"\b(livestock|cattle|goat|goats|sheep|poultry|chicken|dairy|herd)\b",
    re.I,
)
_CROP_FAMILY_RE = re.compile(
    r"\b(crop|crops|cereal|cereals|grain|grains|maize|rice|cassava|wheat|"
    r"sorghum|millet|coffee|cocoa|cotton|soy|soybean|groundnut)\b",
    re.I,
)


@dataclass(frozen=True)
class QueryViews:
    user_query: str
    context_rewrite: str
    detail_rewrite: str
    context_applied: bool
    carried_slots: dict[str, Any] = field(default_factory=dict)

    def vector_texts(self) -> list[str]:
        """Distinct non-empty texts for vector retrieve; user_query always first."""
        out: list[str] = []
        seen: set[str] = set()
        for text in (self.user_query, self.context_rewrite, self.detail_rewrite):
            t = (text or "").strip()
            if not t:
                continue
            key = t.casefold()
            if key in seen:
                continue
            if out and not _materially_different(t, out[0]) and key != out[0].casefold():
                # Near-duplicate of user_query — skip companion
                continue
            if out and any(not _materially_different(t, prev) for prev in out):
                continue
            seen.add(key)
            out.append(t)
        uq = self.user_query.strip()
        if uq and not any(t.casefold() == uq.casefold() for t in out):
            out.insert(0, uq)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "context_rewrite": self.context_rewrite,
            "detail_rewrite": self.detail_rewrite,
            "context_applied": self.context_applied,
            "carried_slots": dict(self.carried_slots),
            "user_query_len": len(self.user_query or ""),
            "context_rewrite_len": len(self.context_rewrite or ""),
            "detail_rewrite_len": len(self.detail_rewrite or ""),
        }


def _materially_different(a: str, b: str) -> bool:
    left = (a or "").strip().casefold()
    right = (b or "").strip().casefold()
    if not left or not right:
        return bool(left) and left != right
    if left == right:
        return False
    # Near-duplicate: one is a short prefix of the other with small delta
    if left in right or right in left:
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if len(longer) - len(shorter) < 12 and shorter:
            return False
    return True


def _is_entity_bag(text: str) -> bool:
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", text or "") if len(t) > 1]
    return 0 < len(tokens) <= _ENTITY_BAG_MAX_TOKENS and " " not in (text or "").strip()


def _profile_scope_hints(
    *,
    user_profile: dict[str, Any] | None,
    plan_type: str | None,
    category: str | None,
) -> str:
    profile = user_profile if isinstance(user_profile, dict) else {}
    parts: list[str] = []
    country = str(profile.get("country") or "").strip()
    if country:
        parts.append(f"Focus country: {country}")
    cat = str(category or profile.get("category") or "").strip()
    if cat:
        parts.append(f"Stakeholder: {cat}")
    pt = str(plan_type or "").strip()
    if pt:
        parts.append(f"Plan: {pt}")
    return ". ".join(parts)


def build_context_rewrite(
    user_query: str,
    *,
    active: ActiveRequest | None = None,
    conversation_summary: str | None = None,
    recent_turns: list[dict[str, Any]] | None = None,
    user_profile: dict[str, Any] | None = None,
    plan_type: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """The question plus its carried slots and profile scope; empty string when nothing applies."""
    if active is None and (recent_turns or conversation_summary):
        history = list(recent_turns or [])
        if conversation_summary:
            history = [{"role": "user", "content": conversation_summary}, *history]
        active, _persisted = resolve_turn(user_query, persisted=None, recent_turns=history)
    context = resolved_request_text(user_query, active) if active else ""
    applied = bool(context) and _materially_different(context, user_query)
    if not applied:
        context = ""
    hints = _profile_scope_hints(
        user_profile=user_profile, plan_type=plan_type, category=category
    )
    if applied and hints:
        context = f"{context} ({hints})"
    elif not applied and hints and _looks_needs_profile_scope(user_query):
        # Elliptical geo-less follow-up: attach profile as scope companion only
        context = f"{user_query.strip()}. {hints}"
        if _materially_different(context, user_query):
            applied = True
        else:
            context = ""
            applied = False
    if context and not _materially_different(context, user_query):
        context = ""
        applied = False
    return {"context_rewrite": context, "context_applied": applied}


def _looks_needs_profile_scope(query: str) -> bool:
    q = (query or "").strip()
    if len(q.split()) > 12:
        return False
    if re.search(
        r"\b(kenya|nigeria|ghana|ethiopia|senegal|uganda|tanzania|cameroon|"
        r"mali|niger|chad|somalia|zambia|malawi)\b",
        q,
        re.I,
    ):
        return False
    return bool(
        re.search(
            r"\b(what about|how about|and for|same for|there|that country|"
            r"my country|the country)\b",
            q,
            re.I,
        )
        or len(q.split()) <= 5
    )


def build_detail_rewrite(
    user_query: str,
    decomposition: dict[str, Any] | None = None,
) -> str:
    """Deterministic enrichment twin for vector search; empty if bag or equal to user."""
    q = (user_query or "").strip()
    if not q:
        return ""
    dec = decomposition if isinstance(decomposition, dict) else {}
    parts: list[str] = [q]

    geos: list[str] = []
    for g in dec.get("geography") or []:
        s = str(g).strip()
        if s and s.casefold() not in q.casefold():
            geos.append(s)
    if geos:
        parts.append("Geography: " + ", ".join(geos[:12]))

    entities: list[str] = []
    for e in dec.get("entities") or []:
        s = str(e).strip()
        if s and s.casefold() not in q.casefold():
            entities.append(s)
    if entities:
        parts.append("Entities: " + ", ".join(entities[:16]))

    years: list[str] = []
    for key in ("time_start", "time_end"):
        v = str(dec.get(key) or "").strip()[:10]
        if v and v not in q:
            years.append(v)
    if years:
        parts.append("Years: " + " to ".join(years))

    measures = dec.get("primary_measures") or dec.get("primary_measure_hints") or []
    families: list[str] = []
    has_dec_signal = bool(geos or entities or years or measures)
    if has_dec_signal:
        if _CROP_FAMILY_RE.search(q) or any(
            _CROP_FAMILY_RE.search(str(e)) for e in (dec.get("entities") or [])
        ):
            families.append("crop")
        if _LIVESTOCK_RE.search(q) or any(
            _LIVESTOCK_RE.search(str(e)) for e in (dec.get("entities") or [])
        ):
            families.append("livestock")
        for m in measures:
            s = str(m).strip().lower()
            if s:
                families.append(s)
    if families:
        parts.append("Topics: " + ", ".join(dict.fromkeys(families)))

    twin = " | ".join(parts)
    if not _materially_different(twin, q):
        return ""
    # Reject pure entity bags as the twin body after the question
    tail = twin[len(q) :].strip(" |")
    if tail and _is_entity_bag(tail.replace("|", " ").replace(",", " ")):
        # Still allow if geos+livestock both present in twin (panel fidelity)
        low = twin.casefold()
        geo_n = sum(1 for g in (dec.get("geography") or []) if str(g).strip().casefold() in low)
        if geo_n < 2 and "livestock" not in low:
            return ""
    return twin


def build_query_views(
    raw_query: str,
    *,
    active: ActiveRequest | None = None,
    conversation_summary: str | None = None,
    recent_turns: list[dict[str, Any]] | None = None,
    user_profile: dict[str, Any] | None = None,
    plan_type: str | None = None,
    category: str | None = None,
    decomposition: dict[str, Any] | None = None,
) -> QueryViews:
    user_query = normalize_query_text(raw_query)
    ctx = build_context_rewrite(
        user_query,
        active=active,
        conversation_summary=conversation_summary,
        recent_turns=recent_turns,
        user_profile=user_profile,
        plan_type=plan_type,
        category=category,
    )
    detail = build_detail_rewrite(user_query, decomposition)
    return QueryViews(
        user_query=user_query,
        context_rewrite=str(ctx.get("context_rewrite") or ""),
        detail_rewrite=detail,
        context_applied=bool(ctx.get("context_applied")),
        carried_slots=active.carried() if active else {},
    )


def user_query_dropped(views: QueryViews | dict[str, Any] | None, vector_texts_used: list[str] | None) -> bool:
    """True when retrieve omitted the full user_query."""
    if views is None:
        return True
    if isinstance(views, QueryViews):
        uq = views.user_query.strip()
    else:
        uq = str(views.get("user_query") or "").strip()
    if not uq:
        return False
    used = vector_texts_used or []
    if not used:
        return True
    return not any((t or "").strip().casefold() == uq.casefold() for t in used)


__all__ = [
    "QueryViews",
    "build_context_rewrite",
    "build_detail_rewrite",
    "build_query_views",
    "user_query_dropped",
]
