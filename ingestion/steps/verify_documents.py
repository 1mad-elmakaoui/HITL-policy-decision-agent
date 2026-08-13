"""Pipeline step 3 -- verify parsed documents.

five checks that ingested data must pass before it reaches the retrieval sources:

* **Quality** -- "well-structured and not contain corrupted or malformed content"
* **Completeness** -- "All expected fields, including metadata ... should be
  present and contain values. Texts should not be truncated."
* **Recency** -- "Older versions should not be retained, but archived and
  replaced by more recent versions", evaluated "using timestamps from the
  metadata"
* **Consistency** -- contradictory information detected, with conflicts resolved
  by source trustworthiness "and/or human intervention can be solicited"
* **Uniqueness** -- "Duplicates should be eliminated, for example with hash-based
  deduplication"
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Tuple

from shared.schema import PolicyDocument, content_hash


class PolicyVerificationError(ValueError):
    """Raised when the corpus cannot be safely indexed."""


def verify_documents(
    parsed: List[Dict[str, Any]], strict: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    documents = [PolicyDocument.from_dict(raw) for raw in parsed]

    report: Dict[str, Any] = {
        "documents_in": len(documents),
        "quality": [],
        "completeness": [],
        "recency": [],
        "consistency": [],
        "uniqueness": [],
        "errors": [],
    }

    kept = _check_recency(documents, report)
    _check_quality(kept, report)
    _check_completeness(kept, report)
    _check_uniqueness(kept, report)
    _check_consistency(kept, report)

    report["documents_out"] = len(kept)
    report["passed"] = not report["errors"]

    if strict and report["errors"]:
        raise PolicyVerificationError(
            "Policy corpus failed verification:\n  - " + "\n  - ".join(report["errors"])
        )

    return [doc.to_dict() for doc in kept], report


def _check_recency(documents: List[PolicyDocument], report: Dict[str, Any]) -> List[PolicyDocument]:
    """Keep only the newest version of each document_id; archive the rest.

    Retaining two versions of the same policy is the failure mode that matters
    most here: retrieval would surface a superseded clause with full provenance,
    and it would look entirely legitimate.
    """
    newest: Dict[str, PolicyDocument] = {}
    for document in documents:
        incumbent = newest.get(document.document_id)
        if incumbent is None or _version_key(document.document_version) > _version_key(
            incumbent.document_version
        ):
            if incumbent is not None:
                report["recency"].append(
                    f"{document.document_id}: superseded v{incumbent.document_version} "
                    f"with v{document.document_version}"
                )
            newest[document.document_id] = document
        else:
            report["recency"].append(
                f"{document.document_id}: dropped stale v{document.document_version} "
                f"(index holds v{incumbent.document_version})"
            )
    return list(newest.values())


def _check_quality(documents: List[PolicyDocument], report: Dict[str, Any]) -> None:
    for document in documents:
        empty = [s.get("heading", "?") for s in document.sections if not s.get("body", "").strip()]
        if empty:
            report["errors"].append(f"{document.document_id}: empty sections {empty}")
        for section in document.sections:
            body = section.get("body", "").strip()
            # A body ending mid-sentence is the signature of a truncated source.
            if body and body[-1] not in ".:;!?)\"'`" and not body.endswith("-"):
                report["quality"].append(
                    f"{document.document_id}/{section.get('heading')}: section does not end "
                    "with terminal punctuation; possible truncation"
                )


def _check_completeness(documents: List[PolicyDocument], report: Dict[str, Any]) -> None:
    required = ("document_id", "policy_name", "document_version", "source")
    for document in documents:
        missing = [f for f in required if not getattr(document, f, "")]
        if missing:
            report["errors"].append(f"{document.source}: missing required fields {missing}")
        for optional in ("effective_date", "owner"):
            if not getattr(document, optional, ""):
                report["completeness"].append(f"{document.document_id}: no {optional} recorded")


def _check_uniqueness(documents: List[PolicyDocument], report: Dict[str, Any]) -> None:
    """Hash-based deduplication, per RAGOps 4.2.2."""
    seen_documents: Dict[str, str] = {}
    for document in documents:
        digest = content_hash(document.raw_text)
        if digest in seen_documents:
            report["errors"].append(
                f"{document.document_id} duplicates {seen_documents[digest]} (identical body)"
            )
        else:
            seen_documents[digest] = document.document_id

        seen_sections: Dict[str, str] = {}
        for section in document.sections:
            section_digest = content_hash(section.get("body", ""))
            heading = section.get("heading", "?")
            if section_digest in seen_sections:
                report["uniqueness"].append(
                    f"{document.document_id}: '{heading}' duplicates "
                    f"'{seen_sections[section_digest]}'"
                )
            else:
                seen_sections[section_digest] = heading


def _check_consistency(documents: List[PolicyDocument], report: Dict[str, Any]) -> None:
    """Flag section headings reused across documents.

    RAGOps proposes semantic-similarity conflict detection here. This
    implementation reports structural collisions -- two policies both owning a
    section with the same title -- and records them for human review rather than
    resolving them automatically, following RAGOps' guidance that conflicts may
    require that "human intervention can be solicited".
    """
    headings: Dict[str, List[str]] = {}
    for document in documents:
        for section in document.sections:
            heading = section.get("heading", "").strip().lower()
            if heading and heading not in {"preamble"}:
                headings.setdefault(heading, []).append(document.document_id)

    for heading, owners in headings.items():
        if len(set(owners)) > 1:
            report["consistency"].append(
                f"heading '{heading}' appears in multiple policies: {sorted(set(owners))}"
            )


def _version_key(version: str) -> Tuple[int, ...]:
    parts = []
    for piece in str(version).split("."):
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


# ---------------------------------------------------------------- ZenML step
#
# Defined at module level, not nested inside the `try` below. ZenML decides how
# many artifacts a step produces by calling `inspect.getsource()` on it and
# parsing the result. A function nested inside a `try:` block comes back
# indented and has to survive a dedent round-trip first, which is fragile
# across environments. At column zero there is nothing to undo.


def verify_documents_step(
    parsed: List[Dict[str, Any]], strict: bool = True
) -> Tuple[
    Annotated[List[Dict[str, Any]], "verified_documents"],
    Annotated[Dict[str, Any], "verification_report"],
]:
    # Explicit tuple literal: ZenML reads the AST of this `return`, so
    # `return verify_documents(...)` would publish one anonymous artifact.
    verified, report = verify_documents(parsed, strict=strict)
    return verified, report


try:  # pragma: no cover
    from zenml import step

    verify_documents_step = step(enable_cache=True)(verify_documents_step)
except ImportError:  # pragma: no cover
    verify_documents_step = None  # type: ignore[assignment]
