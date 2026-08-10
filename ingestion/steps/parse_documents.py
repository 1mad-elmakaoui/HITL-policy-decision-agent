"""Pipeline step 2 -- parse raw documents into structured policy documents."""

from __future__ import annotations

import re
from typing import Any, Dict, List

import yaml

from shared.schema import PolicyDocument

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^##\s+(.*?)\s*$", re.MULTILINE)


class MalformedPolicyDocument(ValueError):
    """Raised when a document cannot be parsed into an identifiable policy.

    Surfaced as a controlled ingestion failure rather than being indexed as an
    anonymous blob -- an un-attributable policy passage is worse than no passage,
    because the runtime cannot cite it.
    """


def parse_documents(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [parse_document(record).to_dict() for record in records]


def parse_document(record: Dict[str, Any]) -> PolicyDocument:
    raw_text = record.get("raw_text", "")
    source = record.get("source", record.get("filename", "unknown"))

    match = _FRONT_MATTER_RE.match(raw_text)
    if not match:
        raise MalformedPolicyDocument(
            f"{source}: missing YAML front matter with document_id / policy_name / document_version"
        )

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise MalformedPolicyDocument(f"{source}: front matter is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise MalformedPolicyDocument(f"{source}: front matter must be a mapping")

    body = raw_text[match.end() :]

    document = PolicyDocument(
        document_id=str(meta.get("document_id", "")).strip(),
        policy_name=str(meta.get("policy_name", "")).strip(),
        # Quoted in YAML so that "4.2" does not arrive as a float and silently
        # become version "4.2" != "4.20" across runs.
        document_version=str(meta.get("document_version", "")).strip(),
        source=str(meta.get("source", source)).strip(),
        effective_date=str(meta.get("effective_date", "")).strip(),
        owner=str(meta.get("owner", "")).strip(),
        sections=_split_sections(body),
        raw_text=body.strip(),
    )

    if not document.document_id or not document.policy_name or not document.document_version:
        raise MalformedPolicyDocument(
            f"{source}: front matter must set document_id, policy_name and document_version"
        )
    if not document.sections:
        raise MalformedPolicyDocument(f"{source}: no '## Section ...' headings found")

    return document


def _split_sections(body: str) -> List[Dict[str, str]]:
    headings = list(_HEADING_RE.finditer(body))
    sections: List[Dict[str, str]] = []

    preamble = body[: headings[0].start()].strip() if headings else body.strip()
    if preamble:
        sections.append({"heading": "Preamble", "body": preamble})

    for index, heading_match in enumerate(headings):
        start = heading_match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append({"heading": heading_match.group(1).strip(), "body": text})

    return sections


try:  # pragma: no cover
    from zenml import step

    @step(enable_cache=True)
    def parse_documents_step(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return parse_documents(records)

except ImportError:  # pragma: no cover
    parse_documents_step = None  # type: ignore[assignment]