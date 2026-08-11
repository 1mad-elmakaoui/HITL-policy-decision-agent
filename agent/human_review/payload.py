"""The payload handed to a human reviewer at the interrupt.

HITL recipe  exposes ``draft_answer``,
``risk_level`` and ``policy_passages`` to the reviewer. The specification adds
the original request, the reason for the high-risk classification, and the
preliminary assessment. All of it is assembled here, from state, so that the
interrupt node stays a boundary rather than a place where content is invented.

The payload deliberately states that the preliminary assessment is *not*
authorisation. It is the model's reading of the evidence, and the reviewer is
being asked to supply the decision that the model cannot.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.state import PolicyReviewState


def build_review_payload(state: PolicyReviewState) -> Dict[str, Any]:
    """Assemble everything a reviewer needs to decide."""
    passages: List[Dict[str, Any]] = list(state.get("policy_passages") or [])

    return {
        "kind": "policy_review.human_sign_off_required",
        "request_id": state.get("request_id", ""),
        "question": state.get("question", ""),
        "requested_by": state.get("requested_by", ""),
        # why this stopped 
        "risk_level": state.get("risk_level", ""),
        "risk_reason": state.get("risk_reason", ""),
        "risk_signals": list(state.get("risk_signals") or []),
        "risk_classifier": state.get("risk_classifier", ""),
        "forced_by_requester": bool(state.get("force_human_review")),
        # what the evidence says
        "evidence_grade": state.get("evidence_grade", ""),
        "policy_passages": [
            {
                "rank": p.get("rank"),
                "citation": p.get("citation"),
                "score": p.get("score"),
                "chunk_id": (p.get("metadata") or {}).get("chunk_id"),
                "document_version": (p.get("metadata") or {}).get("document_version"),
                "text": p.get("text"),
            }
            for p in passages
        ],
        #what the assistant thinks (not a decision)
        "preliminary_assessment": {
            "draft_answer": state.get("draft_answer", ""),
            "basis": list(state.get("draft_basis") or []),
            "conditions": list(state.get("draft_conditions") or []),
            "uncertainty": state.get("draft_uncertainty", ""),
        },
        "notice": (
            "This is a preliminary assessment produced from retrieved policy text. It is "
            "not an authorisation. No answer is released for this request until a reviewer "
            "records a decision below."
        ),
        "response_schema": {
            "decision": "approved | rejected | changes_requested",
            "reviewer_id": "identifier of the person signing off (required)",
            "feedback": "reviewer's reasoning or required amendments (required unless approved)",
        },
    }