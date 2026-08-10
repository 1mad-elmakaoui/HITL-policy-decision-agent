
from __future__ import annotations

import datetime as _dt
from typing import List

from shared.schema import ChunkMetadata, PolicyChunk, PolicyDocument, content_hash, make_chunk_id


def chunk_document(
    document: PolicyDocument,
    max_chunk_chars: int = 1400,
    overlap_chars: int = 160,
    ingested_at: str = "",
) -> List[PolicyChunk]:
    """Turn one parsed document into retrievable chunks."""
    ingested_at = ingested_at or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    chunks: List[PolicyChunk] = []
    ordinal = 0

    for section in document.sections:
        heading = section.get("heading", "").strip()
        body = section.get("body", "").strip()
        if not body:
            continue

        for part in _split_section(body, max_chunk_chars, overlap_chars):
            ordinal += 1
            # The heading is prepended to the indexed text so that the section
            # title participates in retrieval; a question phrased in the words of
            # a heading should find that section.
            text = f"{heading}\n\n{part}" if heading else part
            metadata = ChunkMetadata(
                document_id=document.document_id,
                document_version=document.document_version,
                policy_name=document.policy_name,
                section=heading or "Untitled section",
                source=document.source,
                chunk_id=make_chunk_id(
                    document.document_id, document.document_version, heading, ordinal
                ),
                effective_date=document.effective_date,
                owner=document.owner,
                ingested_at=ingested_at,
                content_hash=content_hash(text),
            )
            chunks.append(PolicyChunk(text=text, metadata=metadata))

    return chunks


def _split_section(body: str, max_chars: int, overlap: int) -> List[str]:
    """Split an over-long section on paragraph boundaries, with overlap."""
    if len(body) <= max_chars:
        return [body]

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    parts: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        parts.append(current)
        tail = current[-overlap:] if overlap > 0 else ""
        current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if current:
        parts.append(current)

    # A single paragraph longer than the maximum still has to be broken up.
    final: List[str] = []
    for part in parts:
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            final.append(part[:cut].strip())
            part = part[max(0, cut - overlap) :].strip()
        if part:
            final.append(part)
    return final