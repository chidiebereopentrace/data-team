"""
Extract ACF claim/finding + structured D/M from chunk text at preprocess time.

Hybrid C:
  - Rules always run.
  - LLM only when rules miss usable direction/magnitude AND
    ``RAG_ACF_CLAIM_EXTRACT=llm`` and an LLM backend is configured.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_DIRECTIONS = frozenset({"increasing", "decreasing", "stable", "unknown"})

_INCREASE_RE = re.compile(
    r"\b(increas(?:e|ed|es|ing)|ris(?:e|es|ing|en)|grew|growth|up(?:ward)?|"
    r"higher|improv(?:e|ed|es|ing)|surge(?:d|s)?|climb(?:ed|s|ing)?|"
    r"gain(?:ed|s|ing)?|expand(?:ed|s|ing)?)\b",
    re.IGNORECASE,
)
_DECREASE_RE = re.compile(
    r"\b(decreas(?:e|ed|es|ing)|declin(?:e|ed|es|ing)|drop(?:ped|s|ping)?|"
    r"fall(?:s|ing|en)?|fell|down(?:ward)?|lower|reduc(?:e|ed|es|ing)|"
    r"shrink(?:s|ing)?|slump(?:ed|s)?|contract(?:ed|s|ing)?|worsen(?:ed|s|ing)?)\b",
    re.IGNORECASE,
)
_STABLE_RE = re.compile(
    r"\b(stable|unchanged|flat|steady|stagnant|plateau(?:ed)?|no\s+change|"
    r"remain(?:ed|s|ing)?\s+(?:the\s+same|stable|flat))\b",
    re.IGNORECASE,
)

# Signed percentage near trend language: -12%, +8.5 percent, 12% YoY
# Note: no \b after "%" — "%" is non-word, so \b fails before whitespace.
_PCT_RE = re.compile(
    r"(?P<sign>[+\-−–]?)\s*(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?:%|percent(?:age)?(?:\s+points?)?|pp)"
    r"(?:\s*(?P<yoy>yoy|year[- ]on[- ]year|year[- ]over[- ]year))?",
    re.IGNORECASE,
)

def claim_extract_mode() -> str:
    """Return ``off`` (rules only) or ``llm`` (hybrid)."""
    raw = os.environ.get("RAG_ACF_CLAIM_EXTRACT", "").strip().lower()
    if raw in ("llm", "on", "1", "true", "hybrid"):
        return "llm"
    return "off"


def _truncate_finding(text: str, limit: int = 300) -> str:
    s = " ".join((text or "").split())
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _sentence_for_span(text: str, start: int, end: int) -> str:
    if not text:
        return ""
    left = text.rfind(".", 0, start)
    left = 0 if left < 0 else left + 1
    right = text.find(".", end)
    if right < 0:
        right = len(text)
    else:
        right += 1
    return _truncate_finding(text[left:right].strip() or text[start:end])


def _metric_from_structured_meta(meta: dict[str, Any] | None) -> str | None:
    """Metric from structured keys only (not domains / general)."""
    if not meta:
        return None
    for key in ("metric", "metric_text", "Measure", "measure", "indicator", "element", "item", "product"):
        val = str(meta.get(key) or "").strip()
        if val:
            # metric_text can be long; take first clause
            return val.split(".")[0].strip()[:128]
    return None


def _metric_from_domains(meta: dict[str, Any] | None) -> str | None:
    if not meta:
        return None
    domains = meta.get("domains")
    if isinstance(domains, str) and domains.strip():
        return domains.split(";")[0].strip().split(",")[0].strip()[:128] or None
    if isinstance(domains, (list, tuple)) and domains:
        first = str(domains[0]).strip()
        return first[:128] if first else None
    return None


def _metric_from_meta(meta: dict[str, Any] | None, *, allow_domains: bool = True) -> str | None:
    structured = _metric_from_structured_meta(meta)
    if structured:
        return structured
    if allow_domains:
        return _metric_from_domains(meta)
    return None


def has_usable_claim_signal(result: dict[str, Any]) -> bool:
    """True when direction is known or a magnitude is present."""
    direction = str(result.get("direction") or "unknown")
    has_mag = result.get("magnitude") is not None
    return direction != "unknown" or has_mag


def _direction_from_text(text: str) -> str:
    inc = bool(_INCREASE_RE.search(text))
    dec = bool(_DECREASE_RE.search(text))
    stab = bool(_STABLE_RE.search(text))
    if stab and not inc and not dec:
        return "stable"
    if inc and not dec:
        return "increasing"
    if dec and not inc:
        return "decreasing"
    if inc and dec:
        # Prefer the earlier match
        mi = _INCREASE_RE.search(text)
        md = _DECREASE_RE.search(text)
        if mi and md:
            return "increasing" if mi.start() <= md.start() else "decreasing"
    return "unknown"


def _magnitude_from_text(text: str, direction: str) -> tuple[float | None, str | None, re.Match[str] | None]:
    m = _PCT_RE.search(text or "")
    if not m:
        return None, None, None
    sign_raw = (m.group("sign") or "").strip().replace("−", "-").replace("–", "-")
    num = float(m.group("num"))
    if sign_raw == "-":
        mag = -abs(num)
    elif sign_raw == "+":
        mag = abs(num)
    elif direction == "decreasing":
        mag = -abs(num)
    elif direction == "increasing":
        mag = abs(num)
    else:
        mag = abs(num)
    unit = "pct_yoy" if m.group("yoy") else "pct"
    return mag, unit, m


def _rules_extract(text: str, meta: dict[str, Any] | None) -> dict[str, Any]:
    body = (text or "").strip()
    direction = _direction_from_text(body) if body else "unknown"
    magnitude, unit, pct_match = _magnitude_from_text(body, direction)

    # If we only have a signed percentage, infer direction from sign
    if direction == "unknown" and magnitude is not None:
        if magnitude < 0:
            direction = "decreasing"
        elif magnitude > 0:
            direction = "increasing"

    finding = ""
    if pct_match is not None:
        finding = _sentence_for_span(body, pct_match.start(), pct_match.end())
    elif direction != "unknown":
        for rx in (_DECREASE_RE, _INCREASE_RE, _STABLE_RE):
            dm = rx.search(body)
            if dm:
                finding = _sentence_for_span(body, dm.start(), dm.end())
                break
    # No first-sentence fallback — omit over invent when no trend span.

    usable = direction != "unknown" or magnitude is not None
    metric = _metric_from_meta(meta, allow_domains=usable) if usable else _metric_from_structured_meta(meta)

    out: dict[str, Any] = {
        "direction": direction,
    }
    if finding:
        out["finding"] = finding
    if metric:
        out["metric"] = metric
    if magnitude is not None:
        out["magnitude"] = magnitude
    if unit:
        out["unit"] = unit
    return out


def _rules_missed(result: dict[str, Any]) -> bool:
    """True when hybrid LLM fallback should be considered."""
    return not has_usable_claim_signal(result)


_LLM_SYSTEM = (
    "Extract one agricultural/economic claim from the text for confidence scoring. "
    "Reply with JSON only, no markdown, keys: "
    'finding (string <= 300 chars), metric (short snake or phrase), '
    'direction (increasing|decreasing|stable|unknown), '
    "magnitude (number or null), unit (string or null, e.g. pct_yoy, pct)."
)


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return data


def _llm_extract(text: str, meta: dict[str, Any] | None) -> dict[str, Any] | None:
    from ml.rag.llm_chat import llm_chat_complete, llm_configured, llm_default_timeout_s, llm_model_id

    if not llm_configured():
        return None
    snippet = " ".join((text or "").split())[:2500]
    if not snippet:
        return None
    hint = _metric_from_meta(meta) or ""
    user = f"Text:\n{snippet}\n"
    if hint:
        user += f"\nHint metric/domain: {hint}\n"
    try:
        raw = llm_chat_complete(
            [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=llm_model_id(),
            max_tokens=256,
            temperature=0.0,
            timeout_s=min(60.0, llm_default_timeout_s()),
            purpose="acf.claim_extract",
        )
    except Exception as exc:
        logger.debug("ACF claim LLM extract failed: %s", exc)
        return None
    data = _parse_llm_json(raw)
    if not data:
        return None
    direction = str(data.get("direction") or "unknown").strip().lower()
    if direction not in _DIRECTIONS:
        direction = "unknown"
    mag: float | None = None
    mag_raw = data.get("magnitude")
    if mag_raw is not None and mag_raw != "":
        try:
            mag = float(mag_raw)
        except (TypeError, ValueError):
            mag = None
    usable = direction != "unknown" or mag is not None
    out: dict[str, Any] = {"direction": direction}
    if usable:
        finding_raw = str(data.get("finding") or "").strip()
        if finding_raw:
            out["finding"] = _truncate_finding(finding_raw)
        metric = str(data.get("metric") or _metric_from_meta(meta, allow_domains=True) or "").strip()[:128]
        if metric:
            out["metric"] = metric
    if mag is not None:
        out["magnitude"] = mag
    unit = data.get("unit")
    if unit is not None and str(unit).strip() and usable:
        out["unit"] = str(unit).strip()[:64]
    return out


def extract_acf_claim(
    text: str,
    *,
    meta: dict[str, Any] | None = None,
    force_llm: bool | None = None,
) -> dict[str, Any]:
    """
    Extract finding + metric/direction/magnitude/unit from chunk text.

    Parameters
    ----------
    text:
        Chunk body (or preferred lane text for OTA).
    meta:
        Existing chunk metadata (domains, metric_text, …).
    force_llm:
        Override env for tests; ``True`` enables hybrid LLM on miss,
        ``False`` forces rules-only, ``None`` uses ``RAG_ACF_CLAIM_EXTRACT``.
    """
    result = _rules_extract(text, meta)
    use_llm = claim_extract_mode() == "llm" if force_llm is None else bool(force_llm)
    if use_llm and _rules_missed(result):
        llm_result = _llm_extract(text, meta)
        if llm_result:
            # Merge: LLM fills gaps; keep rules magnitude/unit if LLM omitted
            merged = dict(result)
            for key in ("finding", "metric", "direction", "magnitude", "unit"):
                if llm_result.get(key) is not None and llm_result.get(key) != "":
                    if key == "direction" and llm_result[key] == "unknown" and merged.get("direction") != "unknown":
                        continue
                    merged[key] = llm_result[key]
            return merged
    return result


def apply_claim_extract_to_meta(
    meta: dict[str, Any],
    text: str,
    *,
    corpus: str | None = None,
    force_llm: bool | None = None,
) -> dict[str, Any]:
    """
    Stamp claim fields onto metadata. Idempotent if finding+direction already set.
    """
    out = dict(meta or {})

    # Prefer OTA lane text for extraction when present
    extract_text = text
    for key in ("metric_text", "insight_text", "recommendation_text"):
        lane = str(out.get(key) or "").strip()
        if lane:
            extract_text = lane
            break

    already = str(out.get("finding") or "").strip() and str(out.get("direction") or "").strip()
    if already:
        return out

    claim = extract_acf_claim(extract_text or text, meta=out, force_llm=force_llm)
    if not has_usable_claim_signal(claim):
        # Omit over invent: leave claim keys absent on no-signal chunks.
        return out
    if claim.get("finding") and not str(out.get("finding") or "").strip():
        out["finding"] = claim["finding"]
    if claim.get("metric") and not str(out.get("metric") or "").strip():
        out["metric"] = claim["metric"]
    if claim.get("direction") and not str(out.get("direction") or "").strip():
        out["direction"] = claim["direction"]
    if out.get("magnitude") is None and claim.get("magnitude") is not None:
        out["magnitude"] = claim["magnitude"]
    if not str(out.get("unit") or "").strip() and claim.get("unit"):
        out["unit"] = claim["unit"]
    return out
