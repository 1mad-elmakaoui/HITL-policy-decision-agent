"""Runtime policy retrieval.

The runtime is a *reader* of the vector store the ZenML pipeline populated. It
never parses a document, never embeds the corpus, and never writes to the index
-- the only thing it embeds is the user's question, which it must, in order to
search.

Retrieval is deliberately callable on its own (``PolicyRetriever.retrieve``)
without constructing the graph, so retrieval quality can be evaluated
independently of the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.state import EVIDENCE_NONE, EVIDENCE_STRONG, EVIDENCE_WEAK
from config.settings import RetrievalSettings, Settings
from shared.embeddings import EmbeddingModel, build_embedder, load_embedder_state
from shared.schema import RetrievedChunk
from shared.vector_store import PolicyVectorStore, VectorStoreUnavailable, build_vector_store


@dataclass
class RetrievalResult:
    """Retrieved evidence plus the grade that will drive routing."""

    passages: List[RetrievedChunk]
    evidence_grade: str
    top_score: float
    error: str = ""

    def to_state_passages(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.passages]

    @property
    def ok(self) -> bool:
        return not self.error


class PolicyRetriever:
    """Reads policy evidence from the shared vector store."""

    def __init__(
        self,
        store: PolicyVectorStore,
        embedder: EmbeddingModel,
        settings: Optional[RetrievalSettings] = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings or RetrievalSettings()

    @classmethod
    def from_settings(cls, settings: Settings) -> "PolicyRetriever":
        # Load the embedder state the ingestion pipeline published with the
        # index, so queries are embedded in the same space as the chunks. If it
        # is absent the embedder falls back to unweighted term frequency, which
        # still ranks but separates poorly -- so its absence is worth noticing.
        state = load_embedder_state(settings.vector_store_path)
        embedder = build_embedder(
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_dimension,
            state=state,
        )
        store = build_vector_store(
            settings.vector_store_backend,
            str(settings.vector_store_path),
            settings.vector_store_collection,
            embedding_model=settings.embedding_provider,
            dimension=embedder.dimension,
            similarity=getattr(embedder, "similarity", "cosine"),
        )
        return cls(store, embedder, settings.retrieval)

    def retrieve(self, question: str, k: Optional[int] = None) -> RetrievalResult:
        """Retrieve evidence for a question.

        Store failures are returned as a result carrying an error, not raised:
        an unreachable index is a workflow state the graph must route on, not an
        exception that escapes the node (specification section 15).
        """
        k = k or self._settings.top_k
        if not question.strip():
            return RetrievalResult([], EVIDENCE_NONE, 0.0, error="empty question")

        try:
            vector = self._embedder.embed_query(question)
            passages = self._store.query(vector, k=k)
        except VectorStoreUnavailable as exc:
            return RetrievalResult([], EVIDENCE_NONE, 0.0, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - any retrieval fault is a failure state
            return RetrievalResult([], EVIDENCE_NONE, 0.0, error=f"retrieval failed: {exc}")

        kept = [p for p in passages if p.score >= self._settings.weak_evidence_score]
        top_score = kept[0].score if kept else (passages[0].score if passages else 0.0)
        return RetrievalResult(kept, self.grade(kept), top_score)

    def grade(self, passages: List[RetrievedChunk]) -> str:
        """Grade evidence quality.

        The LangGraph paper's agentic RAG recipe makes evidence quality a state
        field that routes the workflow rather than "a hidden prompt instruction
        such as 'only answer if supported'" (section 5.2). The grade is computed
        here, from scores, so it is inspectable and testable on its own.
        """
        if not passages:
            return EVIDENCE_NONE
        top = passages[0].score
        if top < self._settings.weak_evidence_score:
            return EVIDENCE_NONE
        supporting = sum(1 for p in passages if p.score >= self._settings.weak_evidence_score)
        if top >= self._settings.strong_evidence_score and supporting >= self._settings.min_supporting_chunks:
            return EVIDENCE_STRONG
        return EVIDENCE_WEAK

    def index_stats(self) -> Dict[str, Any]:
        return self._store.stats().to_dict()