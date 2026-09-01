"""Import smoke tests — no class_engines circular import."""
from __future__ import annotations


def test_sql_compiler_import_without_class_engines_cycle() -> None:
    from ml.rag.chatbot.sql_compiler import assemble_sql, compile_sql, sql_compiler_enabled

    assert sql_compiler_enabled() is True
    assert callable(assemble_sql)
    assert callable(compile_sql)


def test_bq_retriever_import_after_sql_compiler() -> None:
    from ml.rag.retrievers.bq_retriever import BQRetriever

    assert BQRetriever is not None


def test_registry_covers_all_indicator_classes() -> None:
    from ml.rag.chatbot.class_engines.registry import ENGINE_BY_CODE
    from ml.rag.chatbot.mart_indicator_classes import all_class_codes

    codes = set(all_class_codes())
    assert codes
    assert codes == set(ENGINE_BY_CODE.keys())
