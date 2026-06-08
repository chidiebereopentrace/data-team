"""Unit tests for dense embedding backend routing."""
from __future__ import annotations

from unittest.mock import patch

from ml.rag.retrievers.vector_retriever import _resolve_embed_backend


def test_resolve_embed_backend_fastembed() -> None:
    assert _resolve_embed_backend("fastembed") == "fastembed"


def test_resolve_embed_backend_hf_api_redirects_to_fastembed() -> None:
    assert _resolve_embed_backend("hf_api") == "fastembed"


def test_resolve_embed_backend_local_without_torch() -> None:
    with patch(
        "ml.rag.retrievers.vector_retriever._sentence_transformers_available",
        return_value=False,
    ):
        assert _resolve_embed_backend("local") == "fastembed"


def test_resolve_embed_backend_local_with_torch() -> None:
    with patch(
        "ml.rag.retrievers.vector_retriever._sentence_transformers_available",
        return_value=True,
    ):
        assert _resolve_embed_backend("local") == "local"
