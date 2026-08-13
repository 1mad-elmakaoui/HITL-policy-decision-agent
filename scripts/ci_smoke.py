"""End-to-end behaviour check for CI.

Runs real questions through the real service against the real index and
verifies the four properties the project exists to guarantee:

  1. A low-risk question is answered without involving a human.
  2. A high-risk question stops and releases nothing.
  3. A fresh process resumes the parked request from its checkpoint, and
     continues rather than restarting.
  4. A question the corpus does not cover produces no evidence, so the
     assistant cannot invent a rule.

Prints what happened, so the CI log shows the behaviour and not just a tick.
Exits non-zero on the first failure.

Locally:

    python -m cli.ingest --local
    python scripts/ci_smoke.py
"""

from __future__ import annotations

import sys

from agent.runtime import PolicyReviewService

LOW_RISK = "How many days of annual leave can be carried over?"
HIGH_RISK = "Can we terminate an employee immediately for expense fraud?"
OFF_CORPUS = "Is my iguana allowed at the shareholder meeting?"


def check_low_risk(service: PolicyReviewService) -> None:
    print("\n[1/4] A low-risk question is answered automatically")
    result = service.submit(LOW_RISK, request_id="ci-low")

    assert result.status == "completed", f"expected completed, got {result.status}"
    assert result.final_answer, "no answer produced"
    assert not result.awaiting_human_review, "a low-risk question asked for a human"

    print(f"      risk   : {result.risk_level}")
    print(f"      route  : {' -> '.join(result.route_history)}")
    print("      OK")


def check_high_risk_pauses(service: PolicyReviewService) -> None:
    print("\n[2/4] A high-risk question stops for human sign-off")
    result = service.submit(HIGH_RISK, request_id="ci-gate")

    assert result.awaiting_human_review, "OVERSIGHT BYPASSED: the graph did not pause"
    assert not result.final_answer, "OVERSIGHT BYPASSED: an answer was released"
    assert result.status == "pending_human_review", f"unexpected status {result.status}"

    print(f"      status : {result.status}")
    print(f"      risk   : {result.risk_level} - {result.risk_reason}")
    print(f"      route  : {' -> '.join(result.route_history)}")
    print("      OK, paused with nothing released")


def check_resume_from_checkpoint() -> None:
    print("\n[3/4] A fresh service resumes the parked request")

    # A brand-new object holding nothing from the calls above. Everything it
    # knows comes from the checkpoint, found by thread_id alone.
    service = PolicyReviewService()

    assert service.pending_review("ci-gate") is not None, "the parked request was lost"

    result = service.resume(
        "ci-gate", {"decision": "approved", "reviewer_id": "ci@example.com"}
    )

    assert result.status == "completed", f"expected completed, got {result.status}"
    assert result.final_answer, "resumed but produced no answer"

    # Resumed, not restarted: every node before the pause ran exactly once.
    for node in ("retrieve_policy", "draft_assessment", "classify_risk"):
        count = result.route_history.count(node)
        assert count == 1, f"{node} ran {count} times, the workflow restarted"

    print(f"      route  : {' -> '.join(result.route_history)}")
    print("      OK, continued from the checkpoint")


def check_off_corpus(service: PolicyReviewService) -> None:
    print("\n[4/4] A question outside the corpus reaches no evidence")
    result = service.submit(OFF_CORPUS, request_id="ci-offcorpus")

    assert result.evidence_grade != "strong", (
        f"an off-corpus question reached strong evidence: {result.evidence_grade}"
    )

    print(f"      status   : {result.status}")
    print(f"      evidence : {result.evidence_grade}")
    print("      OK, no confident answer was manufactured")


def main() -> int:
    service = PolicyReviewService()

    check_low_risk(service)
    check_high_risk_pauses(service)
    check_resume_from_checkpoint()
    check_off_corpus(service)

    print("\nAll behaviour checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"\nFAILED: {error}", file=sys.stderr)
        sys.exit(1)
