"""Conditional edge functions.

route functions
should read state and return small labels ... Keeping routes small is what makes
the graph inspectable. The model may draft content, but application state decides
whether the workflow retries, escalates, fails closed, or finalizes."

So these are pure functions of state returning a label. No model calls, no I/O,
no side effects -- which also makes them directly testable, which is what
``tests/graph/test_routes.py`` does.

``route_after_risk`` follows the paper's own definition (section 5.4):

    def route_after_risk(state):
        if state["risk_level"] == "high" or state.get("force_human_review"):
            return "interrupt_for_review"
        return "finalize_decision"

with one addition: a missing or unrecognised risk level also routes to review.
An absent classification is not evidence of low risk.
"""

from __future__ import annotations

from agent.state import RISK_HIGH, RISK_ORDER, PolicyReviewState

# Route labels. Named constants so the builder's route map and the tests refer
# to the same strings.
ROUTE_DRAFT = "draft"
ROUTE_FAIL = "fail"
ROUTE_INTERRUPT = "interrupt_for_review"
ROUTE_FINALIZE = "finalize_decision"


def route_after_retrieval(state: PolicyReviewState) -> str:
    """Fail closed when the knowledge layer is unavailable.

    Note what does *not* route to failure: a working index that simply has no
    relevant policy. That case continues to assessment and risk classification,
    so a high-risk question with no policy behind it still reaches a human
    rather than being closed out automatically.
    """
    if state.get("retrieval_error"):
        return ROUTE_FAIL
    return ROUTE_DRAFT


def route_after_risk(state: PolicyReviewState) -> str:
    """Send high-risk work to the human gate; finalise everything else."""
    risk_level = state.get("risk_level", "")

    if risk_level == RISK_HIGH:
        return ROUTE_INTERRUPT
    if state.get("force_human_review"):
        return ROUTE_INTERRUPT
    if risk_level not in RISK_ORDER:
        # Unclassified or unknown label: escalate rather than assume.
        return ROUTE_INTERRUPT
    return ROUTE_FINALIZE