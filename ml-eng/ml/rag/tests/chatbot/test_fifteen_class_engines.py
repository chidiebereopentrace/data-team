"""Registry coverage for 15 indicator class engines."""
from __future__ import annotations

from ml.rag.chatbot.class_engines.registry import ENGINE_BY_CODE, engine_for_class
from ml.rag.chatbot.mart_indicator_classes import all_class_codes


def test_all_fifteen_classes_have_engines() -> None:
    codes = all_class_codes()
    assert len(codes) == 15
    for code in codes:
        assert code in ENGINE_BY_CODE
        eng = engine_for_class(code)
        assert eng.class_code == code or getattr(eng, "class_code", "") == code


def test_prc_and_fs_are_dedicated_engines() -> None:
    from ml.rag.chatbot.class_engines.fs import FsEngine
    from ml.rag.chatbot.class_engines.prc import PrcEngine

    assert isinstance(engine_for_class("PRC"), PrcEngine)
    assert isinstance(engine_for_class("FS"), FsEngine)
