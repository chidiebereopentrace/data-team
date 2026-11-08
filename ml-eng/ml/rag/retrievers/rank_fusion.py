"""Reciprocal rank fusion of independent retriever orderings.

Dense and lexical scores are not on one scale. Fusion uses rank position only,
and each source ranking is kept so a later stage can show where a hit came from.
"""
from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Returns ids best-first with their fused score."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item_id in enumerate(ranking, start=1):
            key = str(item_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


__all__ = ["reciprocal_rank_fusion"]
