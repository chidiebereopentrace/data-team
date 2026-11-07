"""User-visible copy for warehouse failures. Keep jargon out of the spoken answer."""
from __future__ import annotations

USER_BQ_RETRY = (
    "I couldn't pull a warehouse number for that yet. "
    "Try a specific country, crop, and year (e.g. maize in Nigeria in 2024)."
)

_JARGON = (
    "YAML Columns",
    "failed validation",
    "was not submitted",
    "SQL uses columns not in",
)


def scrub_spoken_answer(prose: str) -> str:
    """Replace a spoken answer that leaked validation jargon with one retry sentence."""
    text = prose or ""
    if any(marker in text for marker in _JARGON):
        return USER_BQ_RETRY
    return prose
