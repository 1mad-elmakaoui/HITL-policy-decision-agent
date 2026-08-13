"""Pipeline step 4 -- chunk verified documents"""

# NOTE: no `from __future__ import annotations` in this module, deliberately.
# It turns every annotation into a string, and ZenML resolves step signatures
# without evaluating strings on some versions (0.92 does not, 0.96 does). The
# symptoms are remote from the cause: a two-artifact step silently registers a
# single output called "output", and single-output steps fail inside the
# materializer registry with "'str' object has no attribute '__mro__'".

import datetime as _dt
from typing import Any, Dict, List

from ingestion.chunking.section_chunker import chunk_document
from shared.schema import REQUIRED_METADATA_FIELDS, PolicyChunk, PolicyDocument


def chunk_documents(
    documents: List[Dict[str, Any]],
    max_chunk_chars: int = 1400,
    overlap_chars: int = 160,
    ingested_at: str = "",
) -> List[Dict[str, Any]]:
    ingested_at = ingested_at or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    chunks: List[PolicyChunk] = []
    for raw in documents:
        chunks.extend(
            chunk_document(
                PolicyDocument.from_dict(raw),
                max_chunk_chars=max_chunk_chars,
                overlap_chars=overlap_chars,
                ingested_at=ingested_at,
            )
        )

    _assert_provenance(chunks)
    _assert_unique_ids(chunks)
    return [chunk.to_dict() for chunk in chunks]


def _assert_provenance(chunks: List[PolicyChunk]) -> None:
    """No chunk reaches the index without complete provenance.

    This is the ingestion-side half of the guarantee the runtime relies on when
    it cites a passage; enforcing it here means the runtime never has to handle
    an un-attributable chunk.
    """
    for chunk in chunks:
        missing = chunk.metadata.missing_fields()
        if missing:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} is missing required metadata {missing}; "
                f"required: {list(REQUIRED_METADATA_FIELDS)}"
            )


def _assert_unique_ids(chunks: List[PolicyChunk]) -> None:
    seen: Dict[str, int] = {}
    for chunk in chunks:
        seen[chunk.chunk_id] = seen.get(chunk.chunk_id, 0) + 1
    collisions = sorted(cid for cid, count in seen.items() if count > 1)
    if collisions:
        raise ValueError(f"duplicate chunk ids generated: {collisions[:5]}")


try:  # pragma: no cover
    from zenml import step

    @step(enable_cache=True)
    def chunk_documents_step(
        documents: List[Dict[str, Any]],
        max_chunk_chars: int = 1400,
        overlap_chars: int = 160,
    ) -> List[Dict[str, Any]]:
        # `ingested_at` is deliberately left to default inside the step body
        # rather than being a pipeline parameter, so that ZenML caching keys on
        # the document content alone and a re-run with unchanged policies is a
        # cache hit rather than a fresh timestamp.
        return chunk_documents(documents, max_chunk_chars, overlap_chars)

except ImportError:  # pragma: no cover
    chunk_documents_step = None  # type: ignore[assignment]