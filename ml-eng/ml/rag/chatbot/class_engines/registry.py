"""Class engine registry."""
from __future__ import annotations

from ml.rag.chatbot.class_engines.base import ClassEngine
from ml.rag.chatbot.class_engines.fvc import FvcEngine
from ml.rag.chatbot.class_engines.generic import GenericEngine
from ml.rag.chatbot.class_engines.prod import ProdEngine
from ml.rag.chatbot.mart_indicator_classes import all_class_codes

_OVERRIDES: dict[str, ClassEngine] = {
    "FVC": FvcEngine(),
    "PROD": ProdEngine(),
}

_ENGINE_BY_CODE: dict[str, ClassEngine] = dict(_OVERRIDES)
for code in all_class_codes():
    if code not in _ENGINE_BY_CODE:
        _ENGINE_BY_CODE[code] = GenericEngine(code)


def engine_for_class(class_code: str) -> ClassEngine:
    return _ENGINE_BY_CODE.get((class_code or "").upper(), GenericEngine((class_code or "PROD").upper()))


__all__ = ["engine_for_class", "ENGINE_BY_CODE"]

ENGINE_BY_CODE = _ENGINE_BY_CODE
