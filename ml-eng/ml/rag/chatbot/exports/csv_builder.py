"""Build CSV exports from tabular data."""
from __future__ import annotations

import io
from typing import Any

import pandas as pd


def build_csv(
    rows: list[dict[str, Any]],
    *,
    filename: str = "export.csv",
) -> tuple[bytes, str]:
    if not rows:
        raise ValueError("No tabular data available for CSV export")
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    name = filename if filename.endswith(".csv") else f"{filename}.csv"
    return buf.getvalue(), name


__all__ = ["build_csv"]
