"""Pipeline step 7 -- evaluate retrieval offline.

This step covers the first two levels, which are the levels ingestion owns:

* **module** -- chunk and embedding health (coverage of documents/sections,
  zero vectors, chunk size distribution)
* **component** -- retrieval quality against a labelled query set, scored with
  recall@K / precision@K / MRR / nDCG@K

End-to-end response quality (faithfulness, hallucination rate) is a runtime
concern and is observed by ``agent/observability.py``, not here.

Evaluation runs against an exact in-memory store built from the same chunks and
vectors that were indexed. That removes approximate-nearest-neighbour
tie-breaking from the measurement, so a metric change means the corpus, chunker
or embedder changed -- not the index's mood.
"""

# NOTE: no `from __future__ import annotations` in this module, deliberately.
# It turns every annotation into a string, and ZenML resolves step signatures
# without evaluating strings on some versions (0.92 does not, 0.96 does). The
# symptoms are remote from the cause: a two-artifact step silently registers a
# single output called "output", and single-output steps fail inside the
# materializer registry with "'str' object has no attribute '__mro__'".

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ingestion.evaluation.metrics import evaluate_queries
from shared.embeddings import build_embedder, coerce_embedder_state
from shared.schema import PolicyChunk
from shared.vector_store import InMemoryPolicyVectorStore


class RetrievalQualityGate(RuntimeError):
    """Raised when retrieval quality falls below the configured thresholds.

    Failing the pipeline is the point: an index that cannot retrieve the right
    policy should not become the corpus a governance workflow reasons over.
    """


def evaluate_retrieval(
    chunks: List[Dict[str, Any]],
    vectors: List[List[float]],
    eval_set_path: str,
    provider: str = "hashing",
    model_name: str = "",
    dimension: int = 16384,
    k: int = 5,
    min_recall_at_k: float = 0.8,
    min_mrr: float = 0.6,
    min_ndcg_at_k: float = 0.6,
    weak_evidence_score: float = 0.55,
    strong_evidence_score: float = 1.00,
    fail_below_threshold: bool = True,
    embedder_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    eval_set = _load_eval_set(eval_set_path)
    k = int(eval_set.get("k", k))

    parsed = [PolicyChunk.from_dict(raw) for raw in chunks]
    # The same fitted state the index was built with, so evaluation measures the
    # retrieval the runtime will actually perform.
    embedder = build_embedder(
        provider, model_name, dimension, state=coerce_embedder_state(embedder_state)
    )
    store = InMemoryPolicyVectorStore(
        collection="eval",
        embedding_model=provider,
        similarity=getattr(embedder, "similarity", "cosine"),
    )
    store.upsert(parsed, vectors)

    per_query: List[Dict[str, Any]] = []
    for query in eval_set.get("queries", []):
        relevant_ids = _resolve_labels(parsed, query.get("relevant") or [])
        retrieved = store.query(embedder.embed_query(query["question"]), k=k)
        retrieved_ids = [r.chunk.chunk_id for r in retrieved]
        per_query.append(
            {
                "id": query.get("id", query["question"][:40]),
                "question": query["question"],
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids,
                "top_score": retrieved[0].score if retrieved else 0.0,
                "top_citation": retrieved[0].chunk.citation() if retrieved else "",
            }
        )

    component = evaluate_queries(per_query, k=k)
    module = _module_metrics(parsed, vectors)

    calibration = _calibration(per_query, weak_evidence_score, strong_evidence_score)
    failures = _check_thresholds(
        component, calibration, k, min_recall_at_k, min_mrr, min_ndcg_at_k, strong_evidence_score
    )

    report: Dict[str, Any] = {
        "k": k,
        "module_level": module,
        "component_level": component,
        "calibration": calibration,
        "thresholds": {
            f"min_recall_at_{k}": min_recall_at_k,
            "min_mrr": min_mrr,
            f"min_ndcg_at_{k}": min_ndcg_at_k,
            "weak_evidence_score": weak_evidence_score,
            "strong_evidence_score": strong_evidence_score,
        },
        "failures": failures,
        "passed": not failures,
        "per_query": per_query,
    }

    if failures and fail_below_threshold:
        raise RetrievalQualityGate(
            "Retrieval evaluation failed; index not accepted:\n  - " + "\n  - ".join(failures)
        )
    return report


def _calibration(
    per_query: List[Dict[str, Any]], weak_evidence_score: float, strong_evidence_score: float
) -> Dict[str, Any]:
    """Check the runtime's evidence thresholds against measured scores.

    The evidence grade is an absolute threshold on the retrieval score, so it is
    only meaningful if it is calibrated against the score distribution the
    current corpus, chunker and embedder actually produce. Calibrating it here --
    inside the pipeline that produces all three -- is what stops the thresholds
    in ``config/policy_review.yaml`` from being magic numbers that quietly rot
    when the corpus grows or the embedder is swapped.

    ``separation_margin`` is the headline number: the gap between the weakest
    on-corpus question and the strongest off-corpus one. It can legitimately be
    negative for a lexical retriever -- an off-corpus question sharing vocabulary
    with the corpus will outscore a genuine question phrased in words the policy
    does not use -- which is why the *gate* asserts the safety property below
    rather than demanding a positive margin.
    """
    positives = [float(q["top_score"]) for q in per_query if q.get("relevant_ids")]
    negatives = [float(q["top_score"]) for q in per_query if not q.get("relevant_ids")]

    calibration: Dict[str, Any] = {
        "weak_evidence_score": weak_evidence_score,
        "strong_evidence_score": strong_evidence_score,
        "positive_min_top_score": min(positives) if positives else 0.0,
        "positive_max_top_score": max(positives) if positives else 0.0,
        "negative_max_top_score": max(negatives) if negatives else 0.0,
        # How many labelled queries each threshold would admit. An operator
        # reading the report can see the effect of moving a threshold without
        # re-running anything.
        "positives_reaching_strong": sum(1 for s in positives if s >= strong_evidence_score),
        "positives_below_weak": sum(1 for s in positives if s < weak_evidence_score),
        "negatives_reaching_strong": sum(1 for s in negatives if s >= strong_evidence_score),
        "negatives_reaching_weak": sum(1 for s in negatives if s >= weak_evidence_score),
    }
    if positives and negatives:
        calibration["separation_margin"] = min(positives) - max(negatives)
    return calibration


def _check_thresholds(
    component: Dict[str, float],
    calibration: Dict[str, Any],
    k: int,
    min_recall: float,
    min_mrr: float,
    min_ndcg: float,
    strong_evidence_score: float,
) -> List[str]:
    failures: List[str] = []

    # --- ranking quality (RAGOps Table 1, component level) ------------------
    for metric, floor in ((f"recall_at_{k}", min_recall), ("mrr", min_mrr), (f"ndcg_at_{k}", min_ndcg)):
        value = component.get(metric, 0.0)
        if value < floor:
            failures.append(f"{metric}={value:.3f} below threshold {floor:.3f}")

    # --- gating safety ------------------------------------------------------
    # The property that must hold: an off-corpus question must never produce
    # STRONG evidence, because strong evidence is what licenses the workflow to
    # assert a policy position. A negative landing at "weak" is acceptable and
    # handled downstream -- the assessment step is required to return
    # "permitted: unknown" and say what the evidence does not settle.
    if calibration.get("negatives_reaching_strong", 0):
        failures.append(
            f"{calibration['negatives_reaching_strong']} off-corpus query/queries reach the "
            f"strong-evidence threshold {strong_evidence_score:.2f} "
            f"(max negative score {calibration.get('negative_max_top_score', 0.0):.2f}); "
            "the assistant would state a policy position on a question the corpus does not cover"
        )
    return failures


def _module_metrics(chunks: List[PolicyChunk], vectors: List[List[float]]) -> Dict[str, Any]:
    sizes = [len(c.text) for c in chunks]
    return {
        "chunks": len(chunks),
        "documents": len({c.metadata.document_id for c in chunks}),
        "sections": len({(c.metadata.document_id, c.metadata.section) for c in chunks}),
        "mean_chunk_chars": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
        "max_chunk_chars": max(sizes) if sizes else 0,
        "min_chunk_chars": min(sizes) if sizes else 0,
        "zero_vectors": sum(1 for v in vectors if not any(v)),
        "chunks_missing_metadata": sum(1 for c in chunks if c.metadata.missing_fields()),
    }


def _resolve_labels(chunks: List[PolicyChunk], labels: List[Dict[str, str]]) -> List[str]:
    """Map (document_id, section_contains) labels onto current chunk ids."""
    resolved: List[str] = []
    for label in labels:
        document_id = label.get("document_id", "")
        needle = label.get("section_contains", "").lower()
        matches = [
            c.chunk_id
            for c in chunks
            if c.metadata.document_id == document_id and needle in c.metadata.section.lower()
        ]
        if not matches:
            raise ValueError(
                f"eval label {label} matches no chunk in the index; the eval set is stale "
                "relative to the corpus"
            )
        resolved.extend(matches)
    return sorted(set(resolved))


def _load_eval_set(path: str) -> Dict[str, Any]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Retrieval evaluation set not found: {file}")
    return yaml.safe_load(file.read_text(encoding="utf-8")) or {}


try:  # pragma: no cover
    from zenml import step

    @step(enable_cache=False)
    def evaluate_retrieval_step(
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        eval_set_path: str,
        provider: str = "hashing",
        model_name: str = "",
        dimension: int = 16384,
        k: int = 5,
        min_recall_at_k: float = 0.8,
        min_mrr: float = 0.6,
        min_ndcg_at_k: float = 0.6,
        weak_evidence_score: float = 0.55,
        strong_evidence_score: float = 1.00,
        fail_below_threshold: bool = True,
        embedder_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return evaluate_retrieval(
            chunks=chunks,
            vectors=vectors,
            eval_set_path=eval_set_path,
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            k=k,
            min_recall_at_k=min_recall_at_k,
            min_mrr=min_mrr,
            min_ndcg_at_k=min_ndcg_at_k,
            weak_evidence_score=weak_evidence_score,
            strong_evidence_score=strong_evidence_score,
            fail_below_threshold=fail_below_threshold,
            embedder_state=embedder_state,
        )

except ImportError:  # pragma: no cover
    evaluate_retrieval_step = None  # type: ignore[assignment]