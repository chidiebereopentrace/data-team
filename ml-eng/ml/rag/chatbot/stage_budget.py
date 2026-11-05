"""Per-stage latency budgets. An overrun keeps the evidence already in hand."""
from __future__ import annotations

import os


def stage_over_budget(elapsed_s: float, *, env_var: str, default_s: float) -> bool:
    raw = os.environ.get(env_var, "").strip()
    try:
        budget = float(raw) if raw else float(default_s)
    except ValueError:
        budget = float(default_s)
    return elapsed_s > max(0.0, budget)
