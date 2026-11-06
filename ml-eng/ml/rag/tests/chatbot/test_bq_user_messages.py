"""Spoken answers must not repeat warehouse validation jargon."""
from __future__ import annotations

from ml.rag.chatbot.bq_user_messages import USER_BQ_RETRY, scrub_spoken_answer


def test_scrub_hides_yaml_column_jargon() -> None:
    raw = "SQL uses columns not in the YAML Columns blocks: cou, pro, wfp, ye"
    assert scrub_spoken_answer(raw) == USER_BQ_RETRY
    assert "YAML" not in scrub_spoken_answer(raw)
    submitted = "OpenTrace warehouse SQL was compiled but was not submitted to the warehouse."
    assert scrub_spoken_answer(submitted) == USER_BQ_RETRY
    assert "was not submitted" not in scrub_spoken_answer(submitted)


def test_scrub_leaves_plain_maize_sentence() -> None:
    plain = "The average maize price in Nigeria in 2024 was 450 naira per kg."
    assert scrub_spoken_answer(plain) == plain
