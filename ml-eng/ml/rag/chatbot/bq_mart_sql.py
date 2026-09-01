"""Mart table FQN helpers — no class_engines dependency."""
from __future__ import annotations

import os


def mart_dataset() -> str:
    return (
        os.environ.get("BQ_DATASET_GOLD", "").strip()
        or os.environ.get("BQ_DATASET", "").strip()
        or "mart_dev"
    )


def mart_table_fqn(table_id: str) -> str:
    project = os.environ.get("BQ_PROJECT", "opentrace-prod-5ga4").strip()
    bare = table_id.split(".")[-1]
    return f"`{project}.{mart_dataset()}.{bare}`"


__all__ = ["mart_dataset", "mart_table_fqn"]
