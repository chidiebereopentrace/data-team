"""HTTP 500 detail hides exception text unless RAG_DEBUG is on."""
from __future__ import annotations

from ml.rag.app.api import _GENERIC_500_DETAIL, _augment_http_500_detail


def test_nameerror_unsourced_figures_has_grounding_message() -> None:
    out = _augment_http_500_detail(
        "name 'unsourced_figures' is not defined",
        debug=False,
    )
    assert out == "Answer grounding failed unexpectedly; retry the question."


def test_exception_text_hidden_without_debug() -> None:
    out = _augment_http_500_detail(
        "403 Access Denied: Table opentrace-prod-5ga4:mart_dev.fct_prices",
        debug=False,
    )
    assert out == _GENERIC_500_DETAIL
    assert "mart_dev" not in out


def test_exception_text_shown_with_debug() -> None:
    out = _augment_http_500_detail("boom", debug=True)
    assert out.startswith("boom")
