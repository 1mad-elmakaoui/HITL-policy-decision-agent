"""Node: apply reviewer feedback.

``apply_feedback`` , sitting between the interrupt and finalisation. It reconciles the reviewer's decision with the
preliminary assessment.

The reviewer's decision governs. Where a reviewer rejects or requires changes,
their feedback replaces the assistant's conclusion rather than being appended as
a caveat -- an assistant that keeps its own answer and notes a disagreement
underneath has not been reviewed, it has been commented on.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from agent.observability import EventLog, traced_node
from agent.state import (
    DECISION_APPROVED,
    DECISION_CHANGES_REQUESTED,
    DECISION_REJECTED,
    PolicyReviewState,
)

NODE_NAME = "apply_feedback"


def make_apply_feedback_node(
    log_factory: Callable[[], EventLog],
) -> Callable[[PolicyReviewState], Dict[str, Any]]:
    @traced_node(NODE_NAME, log_factory)
    def apply_feedback(state: PolicyReviewState) -> Dict[str, Any]:
        decision = state.get("review_decision", "")
        feedback = (state.get("reviewer_feedback") or "").strip()
        draft = state.get("draft_answer", "")

        if decision == DECISION_APPROVED:
            answer = draft
            if feedback:
                answer = f"{draft}\n\nReviewer note: {feedback}"
            return {"draft_answer": answer}

        if decision == DECISION_REJECTED:
            return {
                "draft_answer": (
                    "The reviewer did not approve this request. The assistant's preliminary "
                    f"assessment is superseded.\n\nReviewer decision: {feedback}"
                )
            }

        if decision == DECISION_CHANGES_REQUESTED:
            return {
                "draft_answer": (
                    "The reviewer required changes before this request can proceed. The action "
                    "as described is not approved in its current form.\n\n"
                    f"Required changes: {feedback}"
                )
            }

        # Unreachable through the graph: the interrupt node only emits validated
        # decisions. Handled anyway, closed rather than open.
        return {
            "errors": [f"{NODE_NAME}: unrecognised review decision {decision!r}"],
            "draft_answer": (
                "The review outcome for this request could not be interpreted, so no answer is "
                "released. The request requires re-review."
            ),
        }

    return apply_feedback