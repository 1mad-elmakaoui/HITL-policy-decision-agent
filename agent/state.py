"""The typed graph state."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

#risk levels 
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_ORDER = {RISK_LOW: 0, RISK_MEDIUM: 1, RISK_HIGH: 2}

#evidence grades (LangGraph , agentic RAG recipe)
EVIDENCE_STRONG = "strong"
EVIDENCE_WEAK = "weak"
EVIDENCE_NONE = "none"

#terminal and intermediate statuses
STATUS_PENDING_REVIEW = "pending_human_review"
STATUS_COMPLETED = "completed"
STATUS_REJECTED = "rejected_by_reviewer"
STATUS_FAILED = "failed"

#human review decisions
DECISION_APPROVED = "approved"
DECISION_REJECTED = "rejected"
DECISION_CHANGES_REQUESTED = "changes_requested"
VALID_REVIEW_DECISIONS = (DECISION_APPROVED, DECISION_REJECTED, DECISION_CHANGES_REQUESTED)


class PolicyReviewState(TypedDict, total=False):
    """Durable state for one policy-review request."""

    #request
    request_id: str
    question: str
    requested_by: str
    force_human_review: bool

    #retrieval (populated by retrieve_policy)
    policy_passages: list[dict[str, Any]]
    evidence_grade: str
    retrieval_error: str

    #preliminary assessment (populated by draft_assessment) 
    draft_answer: str
    draft_basis: list[str]
    draft_conditions: list[str]
    draft_uncertainty: str

    #risk classification (populated by classify_risk)
    risk_level: str
    risk_reason: str
    risk_signals: list[str]
    risk_classifier: str

    #human review (populated by interrupt_for_review / apply_feedback)
    review_required: bool
    review_decision: str
    approved: bool
    reviewer_id: str
    reviewer_feedback: str
    reviewed_at: str

    #outcome (populated by finalize_decision / record_failure)
    final_answer: str
    decision_record: dict[str, Any]
    status: str

    #observability
    # `operator.add` makes this an append-only log across nodes, so the route a
    # request actually took survives in state and can be asserted in tests.
    route_history: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def new_request_state(
    question: str,
    request_id: str,
    requested_by: str = "",
    force_human_review: bool = False,
) -> PolicyReviewState:
    """Build the initial state for a request."""
    return PolicyReviewState(
        request_id=request_id,
        question=question,
        requested_by=requested_by,
        force_human_review=bool(force_human_review),
        policy_passages=[],
        evidence_grade=EVIDENCE_NONE,
        retrieval_error="",
        draft_answer="",
        draft_basis=[],
        draft_conditions=[],
        draft_uncertainty="",
        risk_level="",
        risk_reason="",
        risk_signals=[],
        risk_classifier="",
        review_required=False,
        review_decision="",
        approved=False,
        reviewer_id="",
        reviewer_feedback="",
        reviewed_at="",
        final_answer="",
        decision_record={},
        status="",
        route_history=[],
        errors=[],
    )


def max_risk(*levels: str) -> str:
    """Highest of the given risk levels; unknown values are treated as high.

    Used by the escalate-only classifier composition, and deliberately
    fail-closed: an unrecognised label must not be able to reduce a risk level.
    """
    best = RISK_LOW
    for level in levels:
        if not level:
            continue
        if level not in RISK_ORDER:
            return RISK_HIGH
        if RISK_ORDER[level] > RISK_ORDER[best]:
            best = level
    return best