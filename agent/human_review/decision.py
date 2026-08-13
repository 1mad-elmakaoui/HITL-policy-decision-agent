"""Validation of the reviewer's input.

The value passed to ``Command(resume=...)`` arrives from outside the system --
an HTTP handler, a queue consumer, a CLI. 
"invalid human-review input" as a failure to handle, and the failure mode that
matters is a malformed payload being read as an approval.

So validation is strict and closed: an unrecognised decision is rejected rather
than coerced, and ``approved`` is true only for an explicit ``"approved"``
decision from an identified reviewer.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent.state import (
    DECISION_APPROVED,
    DECISION_CHANGES_REQUESTED,
    DECISION_REJECTED,
    VALID_REVIEW_DECISIONS,
)


class InvalidReviewInput(ValueError):
    """Raised when reviewer input cannot be accepted as a decision."""


@dataclass(frozen=True)
class ReviewDecision:
    decision: str
    reviewer_id: str
    feedback: str
    reviewed_at: str

    @property
    def approved(self) -> bool:
        # Explicitly, not `decision != rejected`. Only an approval approves.
        return self.decision == DECISION_APPROVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reviewer_id": self.reviewer_id,
            "feedback": self.feedback,
            "reviewed_at": self.reviewed_at,
            "approved": self.approved,
        }


#: Accepted spellings for each decision. Anything outside this map is refused --
#: a free-text reviewer note is not silently read as consent.
_ALIASES = {
    "approved": DECISION_APPROVED,
    "approve": DECISION_APPROVED,
    "approved_with_conditions": DECISION_APPROVED,
    "rejected": DECISION_REJECTED,
    "reject": DECISION_REJECTED,
    "denied": DECISION_REJECTED,
    "deny": DECISION_REJECTED,
    "changes_requested": DECISION_CHANGES_REQUESTED,
    "request_changes": DECISION_CHANGES_REQUESTED,
    "needs_changes": DECISION_CHANGES_REQUESTED,
    "amend": DECISION_CHANGES_REQUESTED,
}


def parse_review_input(raw: Any) -> ReviewDecision:
    """Validate reviewer input into a :class:`ReviewDecision`."""
    if not isinstance(raw, Mapping):
        raise InvalidReviewInput(
            f"reviewer input must be a mapping with a 'decision' key, got {type(raw).__name__}"
        )

    decision_raw = str(raw.get("decision", "")).strip().lower()
    if not decision_raw:
        raise InvalidReviewInput(
            f"reviewer input must set 'decision' to one of {list(VALID_REVIEW_DECISIONS)}"
        )
    decision = _ALIASES.get(decision_raw)
    if decision is None:
        raise InvalidReviewInput(
            f"unrecognised decision {decision_raw!r}; expected one of {list(VALID_REVIEW_DECISIONS)}"
        )

    reviewer_id = str(raw.get("reviewer_id", "")).strip()
    if not reviewer_id:
        raise InvalidReviewInput(
            "reviewer input must identify the reviewer via 'reviewer_id'; an anonymous "
            "sign-off is not a sign-off"
        )

    feedback = str(raw.get("feedback", "")).strip()
    if decision != DECISION_APPROVED and not feedback:
        raise InvalidReviewInput(
            f"a '{decision}' decision must include 'feedback' explaining what is required"
        )

    reviewed_at = str(raw.get("reviewed_at", "")).strip() or _dt.datetime.now(
        _dt.timezone.utc
    ).isoformat(timespec="seconds")

    return ReviewDecision(
        decision=decision, reviewer_id=reviewer_id, feedback=feedback, reviewed_at=reviewed_at
    )