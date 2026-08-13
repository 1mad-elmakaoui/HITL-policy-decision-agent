"""Pipeline step 5 -- embed chunks.

The embedding provider is configuration, not code  the step
emits an embedding report alongside the vectors so a later run can be compared
against an earlier one -
"""

from __future__ import annotations

from typing import Any

from shared.embeddings import embedder_state_of, fit_embedder
from shared.schema import PolicyChunk


def embed_chunks(
    chunks: list[dict[str, Any]],
    provider: str = "hashing",
    model_name: str = "",
    dimension: int = 16384,
) -> tuple[list[list[float]], dict[str, Any]]:
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


try:  # pragma: no cover
    from typing import Annotated

    from zenml import step

    @step(enable_cache=True)
    def embed_chunks_step(
        chunks: list[dict[str, Any]],
        provider: str = "hashing",
        model_name: str = "",
        dimension: int = 16384,
    ) -> tuple[
        Annotated[list[list[float]], "chunk_embeddings"],
        Annotated[dict[str, Any], "embedding_report"],
    ]:
        # Explicit tuple literal -- see the note in verify_documents.py: ZenML
        # detects multiple outputs from the AST of the `return` statement.
        vectors, report = embed_chunks(chunks, provider, model_name, dimension)
        return vectors, report

except ImportError:  # pragma: no cover
    embed_chunks_step = None  # type: ignore[assignment]