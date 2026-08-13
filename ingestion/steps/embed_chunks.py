"""Pipeline step 5 -- embed chunks.

The embedding provider is configuration, not code  the step
emits an embedding report alongside the vectors so a later run can be compared
against an earlier one -
"""

# NOTE: no `from __future__ import annotations` in this module, deliberately.
# It would turn the return annotation below into a string, and ZenML resolves
# the signature without evaluating strings on some versions (0.92 does not,
# 0.96 does). The result is that a step declaring two artifacts silently
# registers one called "output", and the pipeline fails much later with an
# unrelated-looking StepInterfaceError. Keep the annotations as real objects.

from typing import Annotated, Any, Dict, List, Tuple

from shared.embeddings import embedder_state_of, fit_embedder
from shared.schema import PolicyChunk


def embed_chunks(
    chunks: List[Dict[str, Any]],
    provider: str = "hashing",
    model_name: str = "",
    dimension: int = 16384,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Fit the embedder to this corpus, then embed every chunk.

    Fitting happens here rather than at query time because IDF weights are a
    property of the corpus, and the runtime must not compute them -- it would
    have to read the corpus to do so, which is precisely the coupling the
    ZenML/LangGraph split exists to prevent. The fitted state travels with the
    index instead (see the index step).
    """
    parsed = [PolicyChunk.from_dict(raw) for raw in chunks]
    texts = [chunk.text for chunk in parsed]

    embedder = fit_embedder(provider, texts, model_name, dimension)
    vectors = embedder.embed_documents(texts)

    if len(vectors) != len(parsed):
        raise ValueError("embedding provider returned a different number of vectors than chunks")

    state = embedder_state_of(embedder)
    report = {
        "provider": provider,
        "model": embedder.name,
        "dimension": embedder.dimension,
        "chunks_embedded": len(vectors),
        "fitted": bool(state and state.fitted),
        # The state is carried in the report so the index step can publish it
        # without re-fitting, and so a pipeline run's artifacts fully describe
        # the embedding space that run produced.
        "embedder_state": state.to_dict() if state else {},
        # Module-level embedding health (RAGOps Table 1). An all-zero vector
        # means a chunk that can never be retrieved, so it is worth catching at
        # ingestion time rather than as a mysterious retrieval miss later.
        "zero_vectors": sum(1 for v in vectors if not any(v)),
    }
    if report["zero_vectors"]:
        raise ValueError(
            f"{report['zero_vectors']} chunk(s) embedded to the zero vector and would be "
            "unretrievable; check the chunker output"
        )
    return vectors, report


# ---------------------------------------------------------------- ZenML step
# Module level for the same reason as verify_documents.py: ZenML parses the
# source of this function, and a nested definition has to be dedented first.


def embed_chunks_step(
    chunks: List[Dict[str, Any]],
    provider: str = "hashing",
    model_name: str = "",
    dimension: int = 16384,
) -> Tuple[
    Annotated[List[List[float]], "chunk_embeddings"],
    Annotated[Dict[str, Any], "embedding_report"],
]:
    vectors, report = embed_chunks(chunks, provider, model_name, dimension)
    return vectors, report


try:  # pragma: no cover
    from zenml import step

    embed_chunks_step = step(enable_cache=True)(embed_chunks_step)
except ImportError:  # pragma: no cover
    embed_chunks_step = None  # type: ignore[assignment]
