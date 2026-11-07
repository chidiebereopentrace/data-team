"""HTTP 500 detail must not blame PyTorch for an unbound app name."""
from __future__ import annotations

from ml.rag.app.api import _augment_http_500_detail


def test_nameerror_unsourced_figures_no_pytorch_hint() -> None:
    out = _augment_http_500_detail(
        "name 'unsourced_figures' is not defined",
        debug=False,
    )
    assert "PyTorch" not in out
    assert "pip install torch" not in out
    assert out == "Answer grounding failed unexpectedly; retry the question."


def test_torch_import_error_still_gets_hint() -> None:
    out = _augment_http_500_detail(
        "No module named 'torch'",
        debug=False,
    )
    assert "pip install torch" in out
