"""Retrieval metrics.

retrieval metrics:
"Mean reciprocal rank (MRR), recall@K, precision@K, normalized discounted
cumulative gain (nDCG)". All four are implemented here over an offline query set
with labelled relevant chunks.

Relevance is judged at the granularity the index is built at -- the chunk id --
so a metric change can always be traced to a specific passage.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = len(relevant.intersection(retrieved_ids[:k]))
    return hits / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    hits = len(relevant.intersection(retrieved_ids[:k]))
    return hits / min(k, max(len(retrieved_ids), 1))


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@k."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_queries(results: List[Dict[str, object]], k: int) -> Dict[str, float]:
    """Aggregate per-query metrics into the component-level report.

    ``results`` items carry ``retrieved_ids`` and ``relevant_ids``. Queries with
    no labelled relevant chunk are *negative* cases -- deliberately off-corpus
    questions -- and are scored separately: what matters for them is that the
    retriever's top score stays below the evidence threshold, not that it ranks
    anything.
    """
    positives = [r for r in results if r.get("relevant_ids")]
    negatives = [r for r in results if not r.get("relevant_ids")]

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    report = {
        "queries": float(len(results)),
        "positive_queries": float(len(positives)),
        "negative_queries": float(len(negatives)),
        f"recall_at_{k}": mean(
            [recall_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in positives]  # type: ignore[arg-type]
        ),
        f"precision_at_{k}": mean(
            [precision_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in positives]  # type: ignore[arg-type]
        ),
        "mrr": mean(
            [reciprocal_rank(r["retrieved_ids"], r["relevant_ids"]) for r in positives]  # type: ignore[arg-type]
        ),
        f"ndcg_at_{k}": mean(
            [ndcg_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in positives]  # type: ignore[arg-type]
        ),
    }

    if negatives:
        report["negative_max_score"] = max(float(r.get("top_score", 0.0)) for r in negatives)
        report["negative_mean_top_score"] = mean([float(r.get("top_score", 0.0)) for r in negatives])
    return report