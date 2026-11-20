"""Truncation detection and honest length-cap messaging (ML-054)."""

from __future__ import annotations

import os
from unittest import mock

from ml.rag.chatbot.generator import (
    _append_truncation_notice,
    _generate_max_tokens,
)
from ml.rag.llm_chat import (
    answer_was_length_capped,
    length_capped_call_count,
    last_finish_reason,
    record_finish_reason,
    reset_llm_truncation,
)


def test_finish_reason_length_is_recorded() -> None:
    reset_llm_truncation()
    assert answer_was_length_capped() is False
    assert length_capped_call_count() == 0
    record_finish_reason("length")
    assert answer_was_length_capped() is True
    assert length_capped_call_count() == 1
    assert last_finish_reason() == "length"


def test_finish_reason_stop_is_not_truncation() -> None:
    reset_llm_truncation()
    record_finish_reason("stop")
    assert answer_was_length_capped() is False
    assert length_capped_call_count() == 0


def test_reset_clears_truncation_state() -> None:
    reset_llm_truncation()
    record_finish_reason("length")
    assert answer_was_length_capped() is True
    reset_llm_truncation()
    assert answer_was_length_capped() is False
    assert length_capped_call_count() == 0
    assert last_finish_reason() == ""


def test_empty_or_missing_finish_reason_is_ignored() -> None:
    reset_llm_truncation()
    record_finish_reason(None)
    record_finish_reason("")
    record_finish_reason("   ")
    assert last_finish_reason() == ""
    assert length_capped_call_count() == 0


def test_truncation_notice_appended_only_when_length_capped() -> None:
    reset_llm_truncation()
    answer = "Rwanda leads on adaptation finance. Senegal follows."
    # finish_reason=stop -> untouched
    record_finish_reason("stop")
    assert _append_truncation_notice(answer) == answer
    # finish_reason=length -> notice appended
    reset_llm_truncation()
    record_finish_reason("length")
    out = _append_truncation_notice(answer)
    assert out != answer
    assert "reached the maximum" in out
    assert out.startswith(answer)


def test_truncation_notice_is_not_duplicated() -> None:
    reset_llm_truncation()
    record_finish_reason("length")
    once = _append_truncation_notice("Partial answer about Uganda")
    twice = _append_truncation_notice(once)
    assert twice.count("reached the maximum") == 1


def test_truncation_notice_skips_empty_answer() -> None:
    reset_llm_truncation()
    record_finish_reason("length")
    assert _append_truncation_notice("") == ""


def test_french_truncation_notice_stays_in_french() -> None:
    reset_llm_truncation()
    record_finish_reason("length")
    answer = _append_truncation_notice("Analyse partielle.", lang="fr")
    assert "longueur maximale" in answer
    assert "fewer countries" not in answer
    assert _append_truncation_notice(answer, lang="fr") == answer


def test_multipart_modes_use_env_ceiling_not_clamped_default() -> None:
    # Regression: analytical was hard-clamped to 1536 even with a higher env
    # ceiling, truncating multi-country comparisons mid-sentence.
    with mock.patch.dict(os.environ, {"RAG_GENERATE_MAX_TOKENS": "4096"}, clear=False):
        assert _generate_max_tokens("analytical") == 4096
        assert _generate_max_tokens("research") == 4096
        assert _generate_max_tokens("briefing") == 4096
        # Short modes stay capped by their per-mode default.
        assert _generate_max_tokens("fact_lookup") == 512
        assert _generate_max_tokens("chat") == 512
        assert _generate_max_tokens("data_export_only") == 256


def test_multipart_mode_never_drops_below_its_default() -> None:
    # A low/misconfigured env ceiling must not shrink analytical below 1536.
    with mock.patch.dict(os.environ, {"RAG_GENERATE_MAX_TOKENS": "256"}, clear=False):
        assert _generate_max_tokens("analytical") == 1536


def test_decomposition_token_cap_does_not_mark_the_answer_as_truncated() -> None:
    reset_llm_truncation()
    from ml.rag.llm_chat import llm_chat_complete

    response = mock.Mock(status_code=200)
    response.json.return_value = {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
    with mock.patch.dict(os.environ, {"RAG_LLM_BASE_URL": "https://openrouter.ai/api/v1"}):
        with mock.patch("ml.rag.llm_chat.requests.post", return_value=response):
            llm_chat_complete([{"role": "user", "content": "decompose"}], purpose="decompose")
    assert not answer_was_length_capped()
    assert length_capped_call_count() == 1


def test_deadline_exhaustion_is_recorded_without_an_upstream_call() -> None:
    from ml.rag.llm_chat import llm_chat_complete

    reset_llm_truncation()
    with mock.patch.dict(os.environ, {"RAG_LLM_BASE_URL": "https://openrouter.ai/api/v1"}):
        with mock.patch("ml.rag.llm_chat.remaining_seconds", return_value=0):
            with mock.patch("ml.rag.llm_chat._HTTP_WORKERS.try_submit") as submit:
                assert llm_chat_complete([{"role": "user", "content": "Explain soil health"}], purpose="generate") == ""
    submit.assert_not_called()
    assert last_finish_reason() == "budget_exhausted"
    assert not answer_was_length_capped()
