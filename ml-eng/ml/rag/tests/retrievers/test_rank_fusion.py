"""Lexical, dense, and fused orderings on one frozen candidate set."""
from __future__ import annotations

from ml.rag.retrievers.rank_fusion import reciprocal_rank_fusion

LEXICAL = ["ghana-yam", "methane-yam", "nigeria-yam"]
DENSE = ["nigeria-yam", "ghana-yam", "methane-yam"]


def test_hybrid_keeps_both_countries_ahead_of_the_keyword_only_hit() -> None:
    fused = [item_id for item_id, _score in reciprocal_rank_fusion([LEXICAL, DENSE])]
    assert set(fused[:2]) == {"ghana-yam", "nigeria-yam"}
    assert fused[-1] == "methane-yam"


def test_lexical_only_and_dense_only_stay_distinct() -> None:
    lexical = [item_id for item_id, _score in reciprocal_rank_fusion([LEXICAL])]
    dense = [item_id for item_id, _score in reciprocal_rank_fusion([DENSE])]
    assert lexical == LEXICAL
    assert dense == DENSE
    assert lexical != dense
