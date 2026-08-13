"""Chunk and provenance schema shared by the ZenML ingestion layer and the
LangGraph runtime layer.

This module is deliberately the only place where the shape of a policy chunk is
defined. Ingestion writes chunks in this shape; the runtime reads them back in
this shape. That makes the vector store a real contract between the two layers
rather than an incidental implementation detail.

"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

#: Metadata keys every indexed chunk is required to carry. Verification and
#: indexing both enforce this list, so a chunk can never reach the runtime
#: without provenance.
REQUIRED_METADATA_FIELDS = (
    "document_id",
    "document_version",
    "policy_name",
    "section",
    "source",
    "chunk_id",
)


@dataclass(frozen=True)
class ChunkMetadata:
    """Provenance for a single indexed policy chunk."""

    document_id: str
    document_version: str
    policy_name: str
    section: str
    source: str
    chunk_id: str
    effective_date: str = ""
    owner: str = ""
    ingested_at: str = ""
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ChunkMetadata:
        known = {f: raw.get(f, "") for f in cls.__dataclass_fields__}
        return cls(**known)

    def missing_fields(self) -> list[str]:
        return [f for f in REQUIRED_METADATA_FIELDS if not getattr(self, f, "")]


@dataclass
class PolicyChunk:
    """A retrievable unit of policy text plus its provenance."""

    text: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata.to_dict()}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyChunk:
        return cls(text=raw["text"], metadata=ChunkMetadata.from_dict(raw["metadata"]))

    def citation(self) -> str:
        """Human-readable citation used in grounded answers."""
        return (
            f"{self.metadata.policy_name} "
            f"v{self.metadata.document_version}, {self.metadata.section}"
        )


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, with its score and rank.

    Score is a similarity in [0, 1]; rank is 1-based. Both are kept in graph
    state so that evidence gating decisions can be re-read after the fact
    without re-running retrieval (LangGraph paper, section 6, "Audit traces").
    """

    chunk: PolicyChunk
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.chunk.text,
            "metadata": self.chunk.metadata.to_dict(),
            "score": round(float(self.score), 6),
            "rank": int(self.rank),
            "citation": self.chunk.citation(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RetrievedChunk:
        return cls(
            chunk=PolicyChunk.from_dict(raw),
            score=float(raw.get("score", 0.0)),
            rank=int(raw.get("rank", 0)),
        )


@dataclass
class PolicyDocument:
    """A parsed policy document before chunking."""

    document_id: str
    policy_name: str
    document_version: str
    source: str
    effective_date: str = ""
    owner: str = ""
    sections: list[dict[str, str]] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyDocument:
        known = {f: raw[f] for f in cls.__dataclass_fields__ if f in raw}
        return cls(**known)


def content_hash(text: str) -> str:
    """Stable hash used for RAGOps 4.2.2 uniqueness (deduplication) checks."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def make_chunk_id(document_id: str, document_version: str, section: str, ordinal: int) -> str:
    """Deterministic chunk id.

    Determinism matters twice over: re-running ingestion on an unchanged corpus
    must produce the same ids (so indexing is idempotent rather than duplicating
    the corpus), and a decision record citing a chunk id must stay resolvable.
    """
    slug = "".join(c if c.isalnum() else "-" for c in section.lower()).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)[:60]
    return f"{document_id}@{document_version}#{ordinal:03d}-{slug}"


def optional_str(value: Any | None) -> str:
    return "" if value is None else str(value)