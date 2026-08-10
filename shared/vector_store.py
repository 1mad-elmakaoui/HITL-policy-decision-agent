"""The vector store: the single interface between the offline ZenML ingestion
layer and the online LangGraph runtime.

Two backends implement the same protocol:

* ``ChromaPolicyVectorStore`` -- persistent on-disk store, the default.
* ``InMemoryPolicyVectorStore`` -- exact brute-force search, used by tests and
  by the ingestion pipeline's own evaluation step
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from shared.embeddings import SIMILARITY_COSINE, SIMILARITY_INNER_PRODUCT, cosine_similarity, inner_product
from shared.schema import ChunkMetadata, PolicyChunk, RetrievedChunk


class VectorStoreUnavailable(RuntimeError):
    """Raised when the index cannot be reached or has not been populated.

    The runtime treats this as a controlled failure state rather than an excuse
    to answer from parametric knowledge (specification section 15).
    """


@dataclass
class IndexStats:
    backend: str
    collection: str
    chunk_count: int
    embedding_model: str
    dimension: int
    similarity: str = SIMILARITY_COSINE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "collection": self.collection,
            "chunk_count": self.chunk_count,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "similarity": self.similarity,
        }


@runtime_checkable
class PolicyVectorStore(Protocol):
    """Contract shared by ingestion (write) and runtime (read)."""

    def upsert(self, chunks: Sequence[PolicyChunk], vectors: Sequence[Sequence[float]]) -> int: ...

    def query(
        self, vector: Sequence[float], k: int, where: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]: ...

    def count(self) -> int: ...

    def stats(self) -> IndexStats: ...

    def delete_document(self, document_id: str) -> int: ...


class InMemoryPolicyVectorStore:
    """Exact-search reference implementation.

    Useful wherever reproducibility matters more than scale: unit tests, and the
    ingestion pipeline's own retrieval evaluation step.
    """

    def __init__(
        self,
        collection: str = "policy_chunks",
        embedding_model: str = "",
        dimension: int = 0,
        similarity: str = SIMILARITY_COSINE,
    ) -> None:
        self._collection = collection
        self._embedding_model = embedding_model
        self._dimension = dimension
        self._similarity = similarity
        self._chunks: Dict[str, PolicyChunk] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def upsert(self, chunks: Sequence[PolicyChunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        with self._lock:
            for chunk, vector in zip(chunks, vectors):
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = list(map(float, vector))
                if not self._dimension:
                    self._dimension = len(vector)
        return len(chunks)

    def query(
        self, vector: Sequence[float], k: int, where: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        if not self._chunks:
            raise VectorStoreUnavailable(
                f"Collection {self._collection!r} is empty. Run the ingestion pipeline first."
            )
        score_fn = inner_product if self._similarity == SIMILARITY_INNER_PRODUCT else cosine_similarity
        scored = []
        for chunk_id, stored in self._vectors.items():
            chunk = self._chunks[chunk_id]
            if where and not _matches(chunk.metadata, where):
                continue
            scored.append((score_fn(vector, stored), chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:k], start=1)
        ]

    def count(self) -> int:
        return len(self._chunks)

    def stats(self) -> IndexStats:
        return IndexStats(
            backend="in-memory",
            collection=self._collection,
            chunk_count=self.count(),
            embedding_model=self._embedding_model,
            dimension=self._dimension,
            similarity=self._similarity,
        )

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            doomed = [
                cid for cid, chunk in self._chunks.items() if chunk.metadata.document_id == document_id
            ]
            for chunk_id in doomed:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk_id, None)
        return len(doomed)


class ChromaPolicyVectorStore:
    """Persistent Chroma-backed store.

    RAGOps section 7.1.1 works through this exact choice for a regulation corpus
    and settles on a plain vector database over a knowledge graph, on the grounds
    that the corpus has no complex cross-document dependencies and a graph would
    be "unnecessarily heavy and not cost-efficient". A company policy manual has
    the same shape, so the same conclusion is applied here.
    """

    def __init__(
        self,
        persist_directory: str,
        collection: str = "policy_chunks",
        embedding_model: str = "",
        dimension: int = 0,
        similarity: str = SIMILARITY_COSINE,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise VectorStoreUnavailable("chromadb is not installed") from exc

        self._collection_name = collection
        self._embedding_model = embedding_model
        self._dimension = dimension
        self._similarity = similarity
        try:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            # The index space must match the embedder: a length-aware lexical
            # model publishes unnormalised vectors and is compared by inner
            # product, while a sentence-transformer publishes normalised ones
            # and is compared by cosine.
            self._collection = self._client.get_or_create_collection(
                name=collection, metadata={"hnsw:space": similarity}
            )
        except Exception as exc:  # noqa: BLE001 - any client failure is "unavailable"
            raise VectorStoreUnavailable(f"Could not open Chroma at {persist_directory}: {exc}") from exc

    def upsert(self, chunks: Sequence[PolicyChunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(map(float, v)) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata.to_dict() for c in chunks],
        )
        if not self._dimension:
            self._dimension = len(vectors[0])
        return len(chunks)

    def query(
        self, vector: Sequence[float], k: int, where: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        try:
            total = self._collection.count()
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"Chroma collection unreachable: {exc}") from exc
        if total == 0:
            raise VectorStoreUnavailable(
                f"Collection {self._collection_name!r} is empty. Run the ingestion pipeline first."
            )
        try:
            result = self._collection.query(
                query_embeddings=[list(map(float, vector))],
                n_results=min(k, total),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"Chroma query failed: {exc}") from exc

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: List[RetrievedChunk] = []
        for rank, (text, meta, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            chunk = PolicyChunk(text=text, metadata=ChunkMetadata.from_dict(dict(meta)))
            # Chroma returns a *distance*; both the cosine and inner-product
            # spaces define it as 1 - similarity, so invert to recover the score
            # the evidence gate is calibrated against.
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance), rank=rank))
        return retrieved

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(f"Chroma collection unreachable: {exc}") from exc

    def stats(self) -> IndexStats:
        return IndexStats(
            backend="chroma",
            collection=self._collection_name,
            chunk_count=self.count(),
            embedding_model=self._embedding_model,
            dimension=self._dimension,
            similarity=self._similarity,
        )

    def delete_document(self, document_id: str) -> int:
        before = self.count()
        self._collection.delete(where={"document_id": document_id})
        return before - self.count()


def _matches(metadata: ChunkMetadata, where: Dict[str, Any]) -> bool:
    for key, expected in where.items():
        if getattr(metadata, key, None) != expected:
            return False
    return True


def build_vector_store(
    backend: str,
    persist_directory: str,
    collection: str,
    embedding_model: str = "",
    dimension: int = 0,
    similarity: str = SIMILARITY_COSINE,
) -> PolicyVectorStore:
    backend = (backend or "chroma").lower()
    if backend == "chroma":
        return ChromaPolicyVectorStore(
            persist_directory, collection, embedding_model, dimension, similarity
        )
    if backend in {"memory", "in-memory", "inmemory"}:
        return InMemoryPolicyVectorStore(collection, embedding_model, dimension, similarity)
    raise ValueError(f"Unknown vector store backend: {backend!r}")