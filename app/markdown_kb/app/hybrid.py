from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Protocol, TypeVar


RRF_K = 60


class IdentifiedSection(Protocol):
    id: str


SectionT = TypeVar("SectionT", bound=IdentifiedSection)


@dataclass(frozen=True)
class HybridResult(Generic[SectionT]):
    section: SectionT
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class _Accumulator(Generic[SectionT]):
    section: SectionT
    score: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    vector_score: float | None = None


def merge_rrf(
    bm25_results: Iterable[tuple[SectionT, float]],
    vector_results: Iterable[tuple[SectionT, float]],
    k: int = 3,
    rrf_k: int = RRF_K,
) -> list[HybridResult[SectionT]]:
    """Merge BM25 and vector rankings with reciprocal rank fusion.

    Raw scores are preserved for debugging but never compared directly; only the
    per-ranker ranks contribute to the fused score.
    """

    if k <= 0:
        return []

    merged: dict[str, _Accumulator[SectionT]] = {}

    def add_results(results: Iterable[tuple[SectionT, float]], ranker: str) -> None:
        for rank, (section, raw_score) in enumerate(results, start=1):
            if raw_score <= 0:
                continue
            source_id = section.id
            entry = merged.setdefault(source_id, _Accumulator(section=section))
            entry.score += 1.0 / (rrf_k + rank)
            if ranker == "bm25" and entry.bm25_rank is None:
                entry.bm25_rank = rank
                entry.bm25_score = raw_score
            elif ranker == "vector" and entry.vector_rank is None:
                entry.vector_rank = rank
                entry.vector_score = raw_score

    add_results(bm25_results, "bm25")
    add_results(vector_results, "vector")

    results = [
        HybridResult(
            section=entry.section,
            score=entry.score,
            bm25_rank=entry.bm25_rank,
            vector_rank=entry.vector_rank,
            bm25_score=entry.bm25_score,
            vector_score=entry.vector_score,
            rrf_score=entry.score,
        )
        for entry in merged.values()
        if entry.score > 0
    ]

    def sort_key(result: HybridResult[SectionT]) -> tuple[float, int, str]:
        ranks = [rank for rank in (result.bm25_rank, result.vector_rank) if rank is not None]
        best_rank = min(ranks) if ranks else 10**9
        return (-result.score, best_rank, result.section.id)

    results.sort(key=sort_key)
    return results[:k]
