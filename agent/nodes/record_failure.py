"""Node: terminate a request that cannot be resolved.

The counterpart of the SQL recipe's ``fail`` node
and of the agentic RAG recipe's terminal failed status: a run that cannot be
supported ends in a recorded, controlled state instead of an answer.

Reached only when retrieval itself failed -- an unreachable or unpopulated index.
A request with a working index but no relevant policy is *not* routed here: it
carries on to risk classification and, if high risk, to a human, because "we
found no policy on this" is exactly the kind of answer that should not be handed
back automatically for a consequential action.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from typing import Any

from agent.observability import EventLog, traced_node
from agent.state import STATUS_FAILED, PolicyReviewState

NODE_NAME = "record_failure"


def make_record_failure_node(
    log_factory: Callable[[], EventLog],
) -> Callable[[PolicyReviewState], dict[str, Any]]:
    @traced_node(NODE_NAME, log_factory)
    def record_failure(state: PolicyReviewState) -> dict[str, Any]:
        reason = state.get("retrieval_error") or "the request could not be processed"

        log_factory().emit(
            "decision.failed", str(state.get("request_id", "")), reason=reason
        )

        return {
            "final_answer": (
                "This policy question cannot be resolved right now: "
                f"{reason}. No assessment has been produced, and nothing here should be read "
                "as permission to proceed. Resubmit once the policy index is available."
            ),
            "status": STATUS_FAILED,
            "decision_record": {
                "record_id": f"decision-{state.get('request_id', 'unknown')}",
                "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "request": {
                    "request_id": state.get("request_id", ""),
                    "question": state.get("question", ""),
                    "requested_by": state.get("requested_by", ""),
                },
                "status": STATUS_FAILED,
                "failure_reason": reason,
                "evidence": {"grade": state.get("evidence_grade", ""), "passages": []},
                "risk": {},
                "human_review": {"required": False, "decision": "", "approved": False},
                "route_history": list(state.get("route_history") or []),
                "errors": list(state.get("errors") or []),
            },
        }

    return record_failure