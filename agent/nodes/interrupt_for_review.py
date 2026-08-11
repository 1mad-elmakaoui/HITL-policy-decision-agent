"""Node: pause for human sign-off.

not a prompt asking the model to consult someone. "The interrupt is not just a callback; it is a durable workflow boundary.
The checkpointer preserves the thread so the graph can resume after review
without reconstructing state from logs or a ticketing system."

Mechanically: ``interrupt()`` raises out of the node, LangGraph writes the
checkpoint, and ``invoke`` returns with the payload under ``__interrupt__``. The
process can exit here. When ``Command(resume=...)`` is later supplied for the
same ``thread_id``, this node re-executes and ``interrupt()`` returns the
reviewer's value instead of raising.

That is why nothing above this line in the node has side effects, and why the
reviewer's input is validated the moment it arrives.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from langgraph.types import interrupt

from agent.human_review.decision import InvalidReviewInput, parse_review_input
from agent.human_review.payload import build_review_payload
from agent.observability import EventLog, traced_node
from agent.state import STATUS_PENDING_REVIEW, PolicyReviewState

NODE_NAME = "interrupt_for_review"


def make_interrupt_for_review_node(
    log_factory: Callable[[], EventLog],
) -> Callable[[PolicyReviewState], Dict[str, Any]]:
    @traced_node(NODE_NAME, log_factory)
    def interrupt_for_review(state: PolicyReviewState) -> Dict[str, Any]:
        payload = build_review_payload(state)

        log = log_factory()
        log.emit(
            "human_review.requested",
            str(state.get("request_id", "")),
            risk_level=state.get("risk_level"),
            risk_reason=state.get("risk_reason"),
            evidence_grade=state.get("evidence_grade"),
        )

        # Execution stops here. Everything below runs only on resume.
        reviewer_input = interrupt(payload)

        try:
            decision = parse_review_input(reviewer_input)
        except InvalidReviewInput as exc:
            # Bad input must not fall through as an approval. Re-interrupt with
            # the error attached; the thread stays parked, pending review, until
            # a well-formed decision arrives.
            log.emit(
                "human_review.invalid_input",
                str(state.get("request_id", "")),
                error=str(exc),
            )
            reviewer_input = interrupt({**payload, "error": str(exc), "previous_input": reviewer_input})
            decision = parse_review_input(reviewer_input)

        log.emit(
            "human_review.recorded",
            str(state.get("request_id", "")),
            decision=decision.decision,
            reviewer_id=decision.reviewer_id,
            approved=decision.approved,
        )

        return {
            "review_decision": decision.decision,
            "approved": decision.approved,
            "reviewer_id": decision.reviewer_id,
            "reviewer_feedback": decision.feedback,
            "reviewed_at": decision.reviewed_at,
            "status": STATUS_PENDING_REVIEW,
        }

    return interrupt_for_review