"""Node: classify risk.

An explicit graph node, per the specification and the LangGraph paper's
``risk_score``. All it does is call the classifier and write the structured
result into state; the classification logic itself lives in ``agent/risk/`` and
is tested there without a graph.

Keeping this node thin is the point. The risk level in state is what the router
reads, so the route decision is inspectable after the fact, rather than being an
inference the model made inside a longer prompt.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from agent.observability import EventLog, traced_node
from agent.risk.classifier import RiskClassifier
from agent.state import RISK_HIGH, PolicyReviewState

NODE_NAME = "classify_risk"


def make_classify_risk_node(
    classifier: RiskClassifier, log_factory: Callable[[], EventLog]
) -> Callable[[PolicyReviewState], Dict[str, Any]]:
    @traced_node(NODE_NAME, log_factory)
    def classify_risk(state: PolicyReviewState) -> Dict[str, Any]:
        assessment = classifier.classify(
            state.get("question", ""), state.get("policy_passages") or []
        )

        review_required = assessment.requires_human_review or bool(state.get("force_human_review"))
        update: Dict[str, Any] = {
            "risk_level": assessment.level,
            "risk_reason": assessment.reason,
            "risk_signals": assessment.signals,
            "risk_classifier": assessment.classifier,
            "review_required": review_required,
        }

        if review_required and assessment.level != RISK_HIGH:
            # The requester asked for review on a request the classifier did not
            # rate high. Recorded, so the interrupt is explicable later.
            update["risk_reason"] = (
                f"{assessment.reason} Human review was additionally requested by the submitter."
            )
        return update

    return classify_risk