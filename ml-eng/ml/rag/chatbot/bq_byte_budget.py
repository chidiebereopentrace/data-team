"""UTF-8 byte budgets for BQ reasoner / NL2SQL hints / result context."""
from __future__ import annotations

import os
from typing import Any


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return default


def reasoner_index_max_bytes() -> int:
    return _env_int("RAG_BQ_REASONER_INDEX_MAX_BYTES", 12000)


def hint_max_bytes() -> int:
    # Prefer new byte env; fall back to legacy char env when set.
    raw = os.environ.get("RAG_BQ_HINT_MAX_BYTES", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return _env_int("RAG_BQ_HINT_MAX_CHARS", 8000)


def context_max_bytes() -> int:
    return _env_int("RAG_BQ_CONTEXT_MAX_BYTES", 6000)


def utf8_len(text: str) -> int:
    return len((text or "").encode("utf-8"))


def truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    """Return (truncated_text, was_truncated)."""
    raw = text or ""
    if max_bytes <= 0:
        return "", bool(raw)
    data = raw.encode("utf-8")
    if len(data) <= max_bytes:
        return raw, False
    ellipsis = "…"
    ellipsis_bytes = ellipsis.encode("utf-8")
    keep = max(0, max_bytes - len(ellipsis_bytes))
    cut = data[:keep]
    while cut and (cut[-1] & 0xC0) == 0x80:
        cut = cut[:-1]
    out = cut.decode("utf-8", errors="ignore").rstrip() + ellipsis
    # Final safety clamp if still over (should be rare).
    while utf8_len(out) > max_bytes and len(out) > 1:
        out = out[:-2] + ellipsis if out.endswith(ellipsis) else out[:-1]
    return out, True


def pack_lines(lines: list[str], max_bytes: int) -> tuple[str, bool]:
    """Join lines until the UTF-8 budget is exhausted."""
    if max_bytes <= 0:
        return "", bool(lines)
    text = ""
    truncated = False
    for line in lines:
        candidate = line if not text else text + "\n" + line
        if utf8_len(candidate) <= max_bytes:
            text = candidate
            continue
        truncated = True
        remain = max_bytes - utf8_len(text) - (1 if text else 0)
        if remain > 16:
            frag, _ = truncate_utf8(line, remain)
            text = frag if not text else text + "\n" + frag
        break
    return text, truncated


def trim_bq_result_contents(
    rows: list[dict[str, Any]],
    *,
    max_bytes: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Trim serialized BQ row content to a total UTF-8 byte budget (priority order preserved)."""
    budget = context_max_bytes() if max_bytes is None else max(0, max_bytes)
    out: list[dict[str, Any]] = []
    used = 0
    truncated = False
    for row in rows:
        content = str(row.get("content") or "")
        cost = utf8_len(content)
        if used + cost <= budget:
            out.append(row)
            used += cost
            continue
        remain = budget - used
        if remain > 32:
            frag, _ = truncate_utf8(content, remain)
            trimmed = dict(row)
            trimmed["content"] = frag
            meta = dict(trimmed.get("metadata") or {})
            meta["bq_context_truncated"] = True
            trimmed["metadata"] = meta
            out.append(trimmed)
            used = budget
        truncated = True
        break
    return out, truncated or len(out) < len(rows)
