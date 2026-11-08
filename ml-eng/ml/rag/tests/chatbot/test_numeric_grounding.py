"""Unsourced-figure detection (ML-058, Sprint 2 Week 2).

Internal Test 2: the system stated specific statistics with no traceable
source. These tests pin both halves of the contract -- blatant invented
figures get flagged, and ordinary prose does not trip the guard.
"""

from __future__ import annotations

from ml.rag.chatbot.numeric_grounding import (
    _candidate_figures,
    strip_unsourced_figures,
    unsourced_figures,
)


def _ref(text: str) -> dict[str, object]:
    return {"text": text, "title": "Source", "doc_kind": "academic_article"}


# --- detection -------------------------------------------------------------


def test_invented_figure_is_flagged() -> None:
    answer = "Maize production reached 4200 tonnes last season."
    cited = [_ref("Maize production in the region has been rising steadily.")]
    assert unsourced_figures(answer, cited) == ["4200"]


def test_figure_present_in_source_is_not_flagged() -> None:
    answer = "Maize production reached 4200 tonnes last season."
    cited = [_ref("Official records put maize production at 4200 tonnes.")]
    assert unsourced_figures(answer, cited) == []


def test_thousands_separator_matches_plain_digits() -> None:
    answer = "Output was 1,234 tonnes."
    cited = [_ref("Output was 1234 tonnes that year.")]
    assert unsourced_figures(answer, cited) == []


def test_percentage_traced_to_source_passes() -> None:
    answer = "Yields rose by 32% across the sampled districts."
    cited = [_ref("A 32 percent increase was recorded across sampled districts.")]
    assert unsourced_figures(answer, cited) == []


def test_multiple_unsourced_figures_all_returned() -> None:
    answer = "Production hit 4200 tonnes and exports reached 1850 tonnes."
    cited = [_ref("Production and exports both grew.")]
    assert sorted(unsourced_figures(answer, cited)) == ["1850", "4200"]


# --- false-positive guards -------------------------------------------------


def test_years_are_not_treated_as_claims() -> None:
    answer = "Between 2019 and 2024 the sector expanded."
    cited = [_ref("The sector expanded over recent years.")]
    assert unsourced_figures(answer, cited) == []


def test_year_shaped_quantity_with_a_unit_is_still_checked() -> None:
    """Regression: 1850/2000 look like years but "1850 tonnes" is a figure."""
    answer = "Exports reached 1850 tonnes."
    cited = [_ref("Exports grew last season.")]
    assert unsourced_figures(answer, cited) == ["1850"]


def test_year_shaped_quantity_in_source_is_not_flagged() -> None:
    answer = "Exports reached 1850 tonnes."
    cited = [_ref("Export volume was 1850 tonnes.")]
    assert unsourced_figures(answer, cited) == []


def test_year_shaped_percentage_is_checked() -> None:
    answer = "Coverage rose 2000%."
    cited = [_ref("Coverage rose sharply.")]
    assert unsourced_figures(answer, cited) == ["2000"]


def test_bare_year_after_a_unit_free_context_stays_exempt() -> None:
    answer = "The 2024 harvest was strong and 2019 was weak."
    cited = [_ref("Harvest quality varied by season.")]
    assert unsourced_figures(answer, cited) == []


def test_small_counts_are_ignored() -> None:
    answer = "There are 3 main drivers behind this trend."
    cited = [_ref("Several drivers explain the trend.")]
    assert unsourced_figures(answer, cited) == []


def test_markdown_list_markers_are_not_figures() -> None:
    answer = "1. First driver\n2. Second driver\n3. Third driver"
    cited = [_ref("Drivers are discussed at length.")]
    assert unsourced_figures(answer, cited) == []


def test_citation_markers_are_not_figures() -> None:
    answer = "Production rose sharply [1] across the region [2]."
    cited = [_ref("Production rose sharply across the region.")]
    assert unsourced_figures(answer, cited) == []


def test_no_citations_does_not_flag_everything() -> None:
    """The no-citations path already forces no_evidence; don't double-punish."""
    answer = "Production reached 4200 tonnes."
    assert unsourced_figures(answer, []) == []


def test_answer_without_numbers_is_clean() -> None:
    answer = "Maize production has been rising steadily across the region."
    cited = [_ref("Production is rising.")]
    assert unsourced_figures(answer, cited) == []


def test_structured_row_values_count_as_evidence() -> None:
    """Warehouse rows carry figures in arbitrary numeric fields."""
    answer = "Production was 4200 tonnes."
    cited = [{"content": "maize row", "value": 4200, "metadata": {"unit": "tonnes"}}]
    assert unsourced_figures(answer, cited) == []


# --- candidate extraction --------------------------------------------------


def test_candidate_figures_skips_years_and_small_ints() -> None:
    figures = _candidate_figures("In 2024, 3 regions produced 4200 tonnes.")
    assert "2024" not in figures
    assert "3" not in figures
    assert "4200" in figures


def test_candidate_figures_keeps_decimals() -> None:
    assert "2.5" in _candidate_figures("Yield averaged 2.5 t/ha.")


# --- caveat ----------------------------------------------------------------


def test_unsourced_sentence_is_removed() -> None:
    out = strip_unsourced_figures(
        "Production was 4200 tonnes. Rainfall was described as average.",
        ["4200"],
    )
    assert "4200" not in out
    assert "average" in out
    assert "indicative" not in out


def test_nothing_flagged_keeps_the_answer() -> None:
    answer = "Production was 4200 tonnes."
    assert strip_unsourced_figures(answer, []) == answer


def test_every_sentence_unsourced_becomes_a_limitation() -> None:
    out = strip_unsourced_figures("Production was 4200 tonnes.", ["4200"])
    assert "4200" not in out
    assert "does not support a numerical claim" in out
