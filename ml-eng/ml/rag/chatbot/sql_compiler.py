"""Card-driven SQL compiler — SqlRequest validation only; SQL from dynamic stack."""
from __future__ import annotations

import os
from typing import Any

from ml.rag.chatbot.sql_request import SqlRequest, Shape, default_measures_for_shape


def sql_compiler_enabled() -> bool:
    return os.environ.get("RAG_SQL_COMPILER", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def assemble_sql(req: SqlRequest, card: dict[str, Any]) -> str:
    """Deprecated: class engines no longer emit SELECT strings."""
    raise NotImplementedError("assemble_sql removed; use TableBindContract + template/NL2SQL path")


def compile_sql(
    req: SqlRequest,
    card: dict[str, Any],
) -> tuple[str | None, str]:
    """Engines use planned bind contracts; compiler path does not assemble SQL."""
    return None, "use_bind_contract_path"


__all__ = [
    "Shape",
    "SqlRequest",
    "assemble_sql",
    "compile_sql",
    "default_measures_for_shape",
    "sql_compiler_enabled",
]
