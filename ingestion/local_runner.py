"""Run the ingestion steps without the ZenML orchestrator.

This composes the *same* step functions in the *same* order as
``ingestion/pipelines/policy_ingestion.py``. It exists so the corpus can be
indexed in environments where a ZenML stack is not provisioned (CI, a fresh
clone, a test fixture) without duplicating any ingestion logic.

It is a convenience, not the architecture: it deliberately has no versioning,
caching or artifact tracking. Production ingestion runs the ZenML pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config.settings import Settings, load_settings
from ingestion.steps.chunk_documents import chunk_documents
from ingestion.steps.embed_chunks import embed_chunks
from ingestion.steps.evaluate_retrieval import evaluate_retrieval
from ingestion.steps.index_chunks import index_chunks
from ingestion.steps.load_documents import load_documents
from ingestion.steps.parse_documents import parse_documents
from ingestion.steps.verify_documents import verify_documents


def run_ingestion_locally(
    settings: Optional[Settings] = None,
    strict_verification: bool = True,
    fail_below_threshold: Optional[bool] = None,
) -> Dict[str, Any]:
    settings = settings or load_settings()
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)

    raw = load_documents(str(settings.corpus_path))
    parsed = parse_documents(raw)
    verified, verification_report = verify_documents(parsed, strict=strict_verification)
    chunks = chunk_documents(
        verified,
        max_chunk_chars=settings.max_chunk_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    vectors, embedding_report = embed_chunks(
        chunks,
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    index_report = index_chunks(
        chunks,
        vectors,
        backend=settings.vector_store_backend,
        persist_directory=str(settings.vector_store_path),
        collection=settings.vector_store_collection,
        embedding_model=settings.embedding_provider,
        embedder_state=embedding_report,
    )
    evaluation_report = evaluate_retrieval(
        chunks=chunks,
        vectors=vectors,
        eval_set_path=str(settings.eval_set_full_path),
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        k=settings.evaluation.k,
        min_recall_at_k=settings.evaluation.min_recall_at_k,
        min_mrr=settings.evaluation.min_mrr,
        min_ndcg_at_k=settings.evaluation.min_ndcg_at_k,
        weak_evidence_score=settings.retrieval.weak_evidence_score,
        strong_evidence_score=settings.retrieval.strong_evidence_score,
        fail_below_threshold=(
            settings.evaluation.fail_pipeline_below_threshold
            if fail_below_threshold is None
            else fail_below_threshold
        ),
        embedder_state=embedding_report,
    )

    return {
        "verification": verification_report,
        "embedding": embedding_report,
        "index": index_report,
        "evaluation": evaluation_report,
    }