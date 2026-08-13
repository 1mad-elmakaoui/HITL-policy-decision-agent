"""Pipeline step 6 -- index chunks into the shared vector store.

This is the only write to the vector store in the whole system.

After this step returns, the runtime can read the index. The runtime never
writes to it.
"""

# NOTE: no `from __future__ import annotations` in this module, deliberately.
# It turns every annotation into a string, and ZenML resolves step signatures
# without evaluating strings on some versions (0.92 does not, 0.96 does). The
# symptoms are remote from the cause: a two-artifact step silently registers a
# single output called "output", and single-output steps fail inside the
# materializer registry with "'str' object has no attribute '__mro__'".

from typing import Any, Dict, List, Optional

from shared.embeddings import build_embedder, coerce_embedder_state, save_embedder_state
from shared.schema import PolicyChunk
from shared.vector_store import build_vector_store


def index_chunks(
    chunks: List[Dict[str, Any]],
    vectors: List[List[float]],
    backend: str = "chroma",
    persist_directory: str = ".policy_index",
    collection: str = "policy_chunks",
    embedding_model: str = "",
    embedder_state: Optional[Dict[str, Any]] = None,
    reconcile: bool = True,
) -> Dict[str, Any]:
    parsed = [PolicyChunk.from_dict(raw) for raw in chunks]
    dimension = len(vectors[0]) if vectors else 0
    state = coerce_embedder_state(embedder_state)
    # The store's similarity is dictated by the embedder that produced the
    # vectors, never chosen independently.
    similarity = getattr(build_embedder(embedding_model or "hashing"), "similarity", "cosine")
    store = build_vector_store(
        backend, persist_directory, collection, embedding_model, dimension, similarity
    )

    removed = 0
    if reconcile:
        # Remove every chunk of each document being re-ingested, then write the
        # current version. Upsert alone would leave orphans behind whenever a
        # policy edit removes or renumbers a section.
        for document_id in sorted({chunk.metadata.document_id for chunk in parsed}):
            try:
                removed += max(0, store.delete_document(document_id))
            except Exception:  # noqa: BLE001 - an empty/new collection is fine
                pass

    written = store.upsert(parsed, vectors)

    # Publish the fitted embedder alongside the vectors. The index and the
    # embedder that produced it are one artifact: a query embedded by a
    # different fit would be searching a different space.
    state_path = save_embedder_state(persist_directory, state)

    stats = store.stats()

    return {
        "chunks_written": written,
        "stale_chunks_removed": removed,
        "embedder_state_path": str(state_path) if state_path else "",
        "embedder_fitted": bool(state and state.fitted),
        "documents": sorted({c.metadata.document_id for c in parsed}),
        "document_versions": {
            c.metadata.document_id: c.metadata.document_version for c in parsed
        },
        **stats.to_dict(),
    }


try:  # pragma: no cover
    from zenml import step

    @step(enable_cache=False)
    def index_chunks_step(
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        backend: str = "chroma",
        persist_directory: str = ".policy_index",
        collection: str = "policy_chunks",
        embedding_model: str = "",
        embedder_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Caching is disabled here on purpose: this step has a side effect
        outside the ZenML artifact store, so a cache hit would report success
        while leaving the actual index untouched."""
        return index_chunks(
            chunks, vectors, backend, persist_directory, collection, embedding_model,
            embedder_state,
        )

except ImportError:  # pragma: no cover
    index_chunks_step = None  # type: ignore[assignment]