"""Node: finalise and write the decision record.

Two jobs.

**The governance guard.**"the system must
make it impossible for the normal high-risk execution path to silently bypass
human review". The router already sends every high-risk request to the
interrupt, but a router is one edge in one builder, and edges get edited. So
finalisation independently re-checks the invariant and raises
:class:`HumanReviewBypassError` if a high-risk request reaches it without a
recorded reviewer decision. Under the shipped graph this is unreachable; it is
there so that a future edit which makes it reachable fails loudly and
immediately instead of quietly emitting unapproved answers. The LangGraph paper
describes exactly that regression: removing the review gate "can leave
status=completed while the decision record no longer matches what a reviewer
would have approved".

**The decision record.** The paper lists ``decision_record`` among the fields
that make a workflow auditable. It is written as structured data -- who asked,
what was retrieved and from which document version, how risk was classified,
who signed off -- so the record survives without the prose.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Callable, Dict, List

from agent.observability import EventLog, traced_node
from agent.state import (
    DECISION_APPROVED,
    EVIDENCE_NONE,
    RISK_HIGH,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_REJECTED,
    PolicyReviewState,
)

NODE_NAME = "finalize_decision"


class HumanReviewBypassError(RuntimeError):
    """Raised if a high-risk request reaches finalisation unreviewed."""


def make_finalize_decision_node(
    log_factory: Callable[[], EventLog],
) -> Callable[[PolicyReviewState], Dict[str, Any]]:
    @traced_node(NODE_NAME, log_factory)
    def finalize_decision(state: PolicyReviewState) -> Dict[str, Any]:
        _assert_human_review_not_bypassed(state)

        review_required = bool(state.get("review_required"))
        decision = state.get("review_decision", "")
        status = _status_for(state, review_required, decision)

        final_answer = _compose_answer(state, review_required, decision)
        record = _build_record(state, status, review_required)

        log_factory().emit(
            "decision.finalized",
            str(state.get("request_id", "")),
            status=status,
            risk_level=state.get("risk_level"),
            human_review_required=review_required,
            review_decision=decision or None,
            record_id=record["record_id"],
        )

        return {"final_answer": final_answer, "decision_record": record, "status": status}

    return finalize_decision


def _assert_human_review_not_bypassed(state: PolicyReviewState) -> None:
    high_risk = state.get("risk_level") == RISK_HIGH or bool(state.get("force_human_review"))
    if high_risk and not state.get("review_decision"):
        raise HumanReviewBypassError(
            f"request {state.get('request_id', '?')!r} is classified "
            f"{state.get('risk_level', 'unknown')!r} and requires human sign-off, but reached "
            "finalisation with no recorded reviewer decision. Refusing to produce a final answer."
        )


def _status_for(state: PolicyReviewState, review_required: bool, decision: str) -> str:
    if state.get("retrieval_error"):
        return STATUS_FAILED
    if review_required:
        return STATUS_COMPLETED if decision == DECISION_APPROVED else STATUS_REJECTED
    if state.get("evidence_grade") == EVIDENCE_NONE:
        # No policy basis was found. The workflow ran correctly and produced no
        # answer, which is a failed *request*, not a failed system.
        return STATUS_FAILED
    return STATUS_COMPLETED


def _compose_answer(state: PolicyReviewState, review_required: bool, decision: str) -> str:
    parts: List[str] = [state.get("draft_answer", "").strip()]

    conditions = [c for c in (state.get("draft_conditions") or []) if str(c).strip()]
    if conditions:
        parts.append("Conditions and exceptions:\n" + "\n".join(f"- {c}" for c in conditions))

    basis = [b for b in (state.get("draft_basis") or []) if str(b).strip()]
    if basis:
        parts.append("Policy basis:\n" + "\n".join(f"- {b}" for b in basis))

    uncertainty = (state.get("draft_uncertainty") or "").strip()
    if uncertainty:
        parts.append(f"Uncertainty: {uncertainty}")

    # Whether a human approved is part of the answer, not metadata beside it.
    if review_required:
        reviewer = state.get("reviewer_id") or "unidentified reviewer"
        if decision == DECISION_APPROVED:
            parts.append(
                f"Human approval: required and granted by {reviewer} on "
                f"{state.get('reviewed_at', 'unknown date')}."
            )
        else:
            parts.append(
                f"Human approval: required and NOT granted. {reviewer} recorded "
                f"'{decision}' on {state.get('reviewed_at', 'unknown date')}."
            )
    else:
        parts.append(
            "Human approval: not required. This request was classified "
            f"{state.get('risk_level', 'low')} risk and answered automatically."
        )

    return "\n\n".join(p for p in parts if p)


def _build_record(state: PolicyReviewState, status: str, review_required: bool) -> Dict[str, Any]:
    passages = state.get("policy_passages") or []
    return {
        "record_id": f"decision-{state.get('request_id', 'unknown')}",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "request": {
            "request_id": state.get("request_id", ""),
            "question": state.get("question", ""),
            "requested_by": state.get("requested_by", ""),
            "force_human_review": bool(state.get("force_human_review")),
        },
        "evidence": {
            "grade": state.get("evidence_grade", ""),
            "retrieval_error": state.get("retrieval_error", ""),
            # Provenance at the version level: a record must remain interpretable
            # after the underlying policy is revised.
            "passages": [
                {
                    "chunk_id": (p.get("metadata") or {}).get("chunk_id", ""),
                    "document_id": (p.get("metadata") or {}).get("document_id", ""),
                    "document_version": (p.get("metadata") or {}).get("document_version", ""),
                    "policy_name": (p.get("metadata") or {}).get("policy_name", ""),
                    "section": (p.get("metadata") or {}).get("section", ""),
                    "score": p.get("score", 0.0),
                }
                for p in passages
            ],
        },
        "risk": {
            "level": state.get("risk_level", ""),
            "reason": state.get("risk_reason", ""),
            "signals": list(state.get("risk_signals") or []),
            "classifier": state.get("risk_classifier", ""),
        },
        "human_review": {
            "required": review_required,
            "decision": state.get("review_decision", ""),
            "approved": bool(state.get("approved")),
            "reviewer_id": state.get("reviewer_id", ""),
            "feedback": state.get("reviewer_feedback", ""),
            "reviewed_at": state.get("reviewed_at", ""),
        },
        "assessment": {
            "answer": state.get("draft_answer", ""),
            "basis": list(state.get("draft_basis") or []),
            "conditions": list(state.get("draft_conditions") or []),
            "uncertainty": state.get("draft_uncertainty", ""),
        },
        "status": status,
        "route_history": list(state.get("route_history") or []),
        "errors": list(state.get("errors") or []),
    }