"""Route-function tests.

"Recipe tests should assert route
behavior and state transitions, not answer quality alone. For example, a contract
test might assert ``route_after_validation(state) == "retry"`` when ``error`` is
set and ``attempts`` remain."

These are the contract tests for this workflow's routers. They are pure-function
tests -- no graph, no checkpointer, no model -- which is exactly the point of
keeping route functions small.
"""

from __future__ import annotations

import pytest

from agent.routes import (
    ROUTE_DRAFT,
    ROUTE_FAIL,
    ROUTE_FINALIZE,
    ROUTE_INTERRUPT,
    route_after_retrieval,
    route_after_risk,
)
from agent.state import RISK_HIGH, RISK_LOW, RISK_MEDIUM


# --------------------------------------------------------------------------
# route_after_retrieval
# --------------------------------------------------------------------------
def test_retrieval_failure_routes_to_fail() -> None:
    assert route_after_retrieval({"retrieval_error": "index unreachable"}) == ROUTE_FAIL


def test_successful_retrieval_routes_to_draft() -> None:
    assert route_after_retrieval({"retrieval_error": "", "policy_passages": [{}]}) == ROUTE_DRAFT


def test_no_relevant_policy_still_continues_to_assessment() -> None:
    """A working-but-unhelpful index is not a retrieval failure.

    The request must carry on to risk classification, so that a high-risk
    question with no policy behind it still reaches a human rather than being
    closed out automatically.
    """
    state = {"retrieval_error": "", "policy_passages": [], "evidence_grade": "none"}
    assert route_after_retrieval(state) == ROUTE_DRAFT


# --------------------------------------------------------------------------
# route_after_risk -- the governance gate
# --------------------------------------------------------------------------
def test_high_risk_routes_to_the_interrupt() -> None:
    assert route_after_risk({"risk_level": RISK_HIGH}) == ROUTE_INTERRUPT


@pytest.mark.parametrize("level", [RISK_LOW, RISK_MEDIUM])
def test_sub_high_risk_finalizes_without_a_human(level: str) -> None:
    assert route_after_risk({"risk_level": level}) == ROUTE_FINALIZE


def test_forced_review_overrides_a_low_classification() -> None:
    """The requester can always demand a human, per the paper's route function."""
    assert route_after_risk({"risk_level": RISK_LOW, "force_human_review": True}) == ROUTE_INTERRUPT


@pytest.mark.parametrize("level", ["", "unknown", "HIGH", None, "critical"])
def test_unrecognised_risk_level_escalates(level) -> None:
    """Fail closed: an absent or unparseable classification is not low risk."""
    assert route_after_risk({"risk_level": level}) == ROUTE_INTERRUPT


def test_empty_state_escalates() -> None:
    assert route_after_risk({}) == ROUTE_INTERRUPT


# --------------------------------------------------------------------------
# Routers are pure
# --------------------------------------------------------------------------
def test_routers_do_not_mutate_state() -> None:
    state = {"risk_level": RISK_HIGH, "force_human_review": False}
    snapshot = dict(state)
    route_after_risk(state)
    assert state == snapshot


def test_routers_return_labels_the_graph_knows() -> None:
    """A label with no entry in the route map would fail only at runtime."""
    known_risk_routes = {ROUTE_INTERRUPT, ROUTE_FINALIZE}
    known_retrieval_routes = {ROUTE_DRAFT, ROUTE_FAIL}

    for level in (RISK_LOW, RISK_MEDIUM, RISK_HIGH, "", "nonsense"):
        for forced in (True, False):
            assert route_after_risk({"risk_level": level, "force_human_review": forced}) in (
                known_risk_routes
            )

    for error in ("", "boom"):
        assert route_after_retrieval({"retrieval_error": error}) in known_retrieval_routes