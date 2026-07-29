from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Iterable

from .hybrid import HybridResult


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


def rerank_hybrid_results(query: str, fused_results: Iterable[HybridResult], k: int = 3) -> list[HybridResult]:
    """Rerank RRF-fused candidates with a deterministic query/context scorer.

    RRF is great for merging rankers with incomparable score scales. This second
    pass looks at the actual candidate text so an exact, answer-bearing section
    can beat a broad semantic hit. The original RRF score and BM25/vector debug
    fields are preserved on every returned result.
    """

    if k <= 0:
        return []

    query_terms = _expanded_terms(_tokens(query))
    scored: list[tuple[HybridResult, float]] = []
    for result in fused_results:
        rerank_score = _rerank_score(query_terms, result)
        if rerank_score <= 0 or not math.isfinite(rerank_score):
            continue
        scored.append((result, rerank_score))

    scored.sort(key=lambda item: _sort_key(item[0], item[1]))
    reranked: list[HybridResult] = []
    for rank, (result, score) in enumerate(scored[:k], start=1):
        reranked.append(
            replace(
                result,
                score=score,
                rrf_score=result.rrf_score if result.rrf_score is not None else result.score,
                rerank_rank=rank,
                rerank_score=score,
            )
        )
    return reranked


def _rerank_score(query_terms: set[str], result: HybridResult) -> float:
    section = result.section
    heading_terms = _expanded_terms(_tokens(" ".join(getattr(section, "heading_path", []) or [])))
    content_terms = _expanded_terms(_tokens(getattr(section, "content", "")))
    candidate_terms = heading_terms | content_terms
    if not query_terms or not candidate_terms:
        return 0.0

    overlap = query_terms & candidate_terms
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(query_terms)
    precision = len(overlap) / len(candidate_terms)
    heading_coverage = len(query_terms & heading_terms) / len(query_terms)

    # Signals are deliberately bounded and comparable. Raw BM25/vector scores are
    # kept as weak tie-breakers; the text match is the primary rerank signal.
    bm25_signal = _bounded(result.bm25_score)
    vector_signal = _bounded(result.vector_score)
    rrf_signal = _bounded(result.rrf_score if result.rrf_score is not None else result.score)

    return (
        coverage * 2.0
        + precision * 0.5
        + heading_coverage * 0.35
        + bm25_signal * 0.10
        + vector_signal * 0.08
        + rrf_signal * 0.05
    )


def _sort_key(result: HybridResult, score: float) -> tuple[float, float, int, str]:
    ranks = [rank for rank in (result.bm25_rank, result.vector_rank) if rank is not None]
    best_rank = min(ranks) if ranks else 10**9
    rrf_score = result.rrf_score if result.rrf_score is not None else result.score
    return (-score, -rrf_score, best_rank, getattr(result.section, "id", ""))


def _tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOP_WORDS]


def _expanded_terms(tokens: Iterable[str]) -> set[str]:
    terms: set[str] = set()
    for token in tokens:
        terms.add(token)
        if token.endswith("s") and len(token) > 3:
            terms.add(token[:-1])
    return terms


def _bounded(value: float | None) -> float:
    if value is None or value <= 0 or not math.isfinite(value):
        return 0.0
    return value / (1.0 + value)
