"""Node: retrieve policy evidence.

Thin adapter over :class:`agent.retrieval.retriever.PolicyRetriever`. The node
turns a retrieval outcome into state; the retriever does the work. That split is
what lets retrieval be evaluated on its own without building a graph.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.observability import EventLog, traced_node
from agent.retrieval.retriever import PolicyRetriever
from agent.state import EVIDENCE_NONE, PolicyReviewState

NODE_NAME = "retrieve_policy"


def make_retrieve_policy_node(
    retriever: PolicyRetriever, log_factory: Callable[[], EventLog]
) -> Callable[[PolicyReviewState], dict[str, Any]]:
    @traced_node(NODE_NAME, log_factory)
    def retrieve_policy(state: PolicyReviewState) -> dict[str, Any]:
        result = retriever.retrieve(state.get("question", ""))

        if not result.ok:
            # An unreachable or empty index is a routed failure, not an
            # exception and certainly not a reason to answer unaided.
            return {
                "policy_passages": [],
                "evidence_grade": EVIDENCE_NONE,
                "retrieval_error": result.error,
                "errors": [f"{NODE_NAME}: {result.error}"],
            }

        return {
            "policy_passages": result.to_state_passages(),
            "evidence_grade": result.evidence_grade,
            "retrieval_error": "",
        }

    return retrieve_policy