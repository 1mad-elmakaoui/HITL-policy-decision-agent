"""The ZenML policy-ingestion pipeline.

    Load -> Parse -> Verify -> Chunk -> Embed -> Index -> Evaluate

This is the whole of the offline knowledge lifecycle. It runs on demand and when
policy documents change; it does **not** run as part of a user request.

Ordering note: the specification's outline lists Load -> Parse -> Chunk -> Embed
-> Index -> Evaluate, we insert Verification between
ingestion and the retrieval sources. Verification is placed after parsing here
because we fold format conversion into ingestion, and because
the completeness check ("All expected fields, including metadata ... should be
present") can only run once front matter has been parsed into fields.

Every step is also importable as a plain function, so each is unit-testable
without an orchestrator (see ``tests/ingestion/``).
"""

# NOTE: no `from __future__ import annotations` in this module, deliberately.
# It turns every annotation into a string, and ZenML resolves step signatures
# without evaluating strings on some versions (0.92 does not, 0.96 does). The
# symptoms are remote from the cause: a two-artifact step silently registers a
# single output called "output", and single-output steps fail inside the
# materializer registry with "'str' object has no attribute '__mro__'".

from typing import Any, Dict, Optional

from zenml import pipeline

from config.settings import Settings, load_settings
from ingestion.steps.chunk_documents import chunk_documents_step
from ingestion.steps.embed_chunks import embed_chunks_step
from ingestion.steps.evaluate_retrieval import evaluate_retrieval_step
from ingestion.steps.index_chunks import index_chunks_step
from ingestion.steps.load_documents import load_documents_step
from ingestion.steps.parse_documents import parse_documents_step
from ingestion.steps.verify_documents import verify_documents_step


@pipeline(name="policy_ingestion", enable_cache=True)
def policy_ingestion_pipeline(
    corpus_dir: str,
    vector_store_backend: str,
    vector_store_dir: str,
    vector_store_collection: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimension: int,
    max_chunk_chars: int,
    chunk_overlap_chars: int,
    eval_set_path: str,
    eval_k: int,
    min_recall_at_k: float,
    min_mrr: float,
    min_ndcg_at_k: float,
    weak_evidence_score: float,
    strong_evidence_score: float,
    fail_below_threshold: bool,
    strict_verification: bool = True,
) -> None:
    """Versioned, re-runnable ingestion of the policy corpus.

    ZenML gives this the properties the architecture depends on: each run is
    versioned, each step's inputs and outputs are tracked artifacts, and an
    unchanged corpus re-runs from cache -- so re-ingesting after a single policy
    edit is cheap enough to do on every change.
    """
    raw_documents = load_documents_step(corpus_dir=corpus_dir)
    parsed_documents = parse_documents_step(records=raw_documents)
    verified_documents, _verification_report = verify_documents_step(
        parsed=parsed_documents, strict=strict_verification
    )
    chunks = chunk_documents_step(
        documents=verified_documents,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=chunk_overlap_chars,
    )
    vectors, embedding_report = embed_chunks_step(
        chunks=chunks,
        provider=embedding_provider,
        model_name=embedding_model,
        dimension=embedding_dimension,
    )
    index_chunks_step(
        chunks=chunks,
        vectors=vectors,
        backend=vector_store_backend,
        persist_directory=vector_store_dir,
        collection=vector_store_collection,
        embedding_model=embedding_provider,
        embedder_state=embedding_report,
    )
    evaluate_retrieval_step(
        chunks=chunks,
        vectors=vectors,
        eval_set_path=eval_set_path,
        provider=embedding_provider,
        model_name=embedding_model,
        dimension=embedding_dimension,
        k=eval_k,
        min_recall_at_k=min_recall_at_k,
        min_mrr=min_mrr,
        min_ndcg_at_k=min_ndcg_at_k,
        weak_evidence_score=weak_evidence_score,
        strong_evidence_score=strong_evidence_score,
        fail_below_threshold=fail_below_threshold,
        embedder_state=embedding_report,
    )


def run_policy_ingestion(
    settings: Optional[Settings] = None, strict_verification: bool = True
) -> Dict[str, Any]:
    """Run the pipeline with the project's configuration."""
    settings = settings or load_settings()
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)

    policy_ingestion_pipeline(
        corpus_dir=str(settings.corpus_path),
        vector_store_backend=settings.vector_store_backend,
        vector_store_dir=str(settings.vector_store_path),
        vector_store_collection=settings.vector_store_collection,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
        max_chunk_chars=settings.max_chunk_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        eval_set_path=str(settings.eval_set_full_path),
        eval_k=settings.evaluation.k,
        min_recall_at_k=settings.evaluation.min_recall_at_k,
        min_mrr=settings.evaluation.min_mrr,
        min_ndcg_at_k=settings.evaluation.min_ndcg_at_k,
        weak_evidence_score=settings.retrieval.weak_evidence_score,
        strong_evidence_score=settings.retrieval.strong_evidence_score,
        fail_below_threshold=settings.evaluation.fail_pipeline_below_threshold,
        strict_verification=strict_verification,
    )
    return {"pipeline": "policy_ingestion", "collection": settings.vector_store_collection}