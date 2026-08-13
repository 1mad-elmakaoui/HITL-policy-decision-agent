"""CLI for the runtime policy-review assistant.

    python -m cli.review submit "Can we terminate an employee for expense fraud?"
    python -m cli.review pending  pr-1a2b3c4d5e6f
    python -m cli.review resume   pr-1a2b3c4d5e6f --decision approved --reviewer alex@example.com
    python -m cli.review show     pr-1a2b3c4d5e6f
    python -m cli.review trace    pr-1a2b3c4d5e6f

Each sub-command is a separate process. That is deliberate: running `submit`,
letting the shell exit, and running `resume` hours later exercises the same
recovery path a real review delay would.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agent.runtime import NotAwaitingReview, PolicyReviewService, UnknownThread
from config.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Policy Review Assistant")
    parser.add_argument("--config", default=None, help="path to a settings YAML file")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="submit a policy question")
    p_submit.add_argument("question")
    p_submit.add_argument("--request-id", default=None, help="stable thread_id for this request")
    p_submit.add_argument("--requested-by", default="", help="who is asking")
    p_submit.add_argument(
        "--force-review", action="store_true", help="require human sign-off regardless of risk"
    )

    p_pending = sub.add_parser("pending", help="show what a parked request is waiting for")
    p_pending.add_argument("thread_id")

    p_resume = sub.add_parser("resume", help="supply a reviewer decision and resume")
    p_resume.add_argument("thread_id")
    p_resume.add_argument(
        "--decision", required=True, choices=["approved", "rejected", "changes_requested"]
    )
    p_resume.add_argument("--reviewer", required=True, help="reviewer identifier")
    p_resume.add_argument("--feedback", default="", help="reviewer reasoning (required unless approved)")

    p_show = sub.add_parser("show", help="print the checkpointed state of a request")
    p_show.add_argument("thread_id")

    p_trace = sub.add_parser("trace", help="print the observability events for a request")
    p_trace.add_argument("thread_id")

    args = parser.parse_args(argv)
    service = PolicyReviewService(load_settings(args.config))

    try:
        return _dispatch(args, service)
    except UnknownThread as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except NotAwaitingReview as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4


def _dispatch(args: argparse.Namespace, service: PolicyReviewService) -> int:
    if args.command == "submit":
        result = service.submit(
            args.question,
            request_id=args.request_id,
            requested_by=args.requested_by,
            force_human_review=args.force_review,
        )
        _print_result(result)
        return 0

    if args.command == "pending":
        payload = service.pending_review(args.thread_id)
        if payload is None:
            print(f"{args.thread_id} is not awaiting human review.")
            return 1
        _print_review_payload(payload)
        return 0

    if args.command == "resume":
        result = service.resume(
            args.thread_id,
            {
                "decision": args.decision,
                "reviewer_id": args.reviewer,
                "feedback": args.feedback,
            },
        )
        _print_result(result)
        return 0

    if args.command == "show":
        print(json.dumps(_jsonable(service.get_state(args.thread_id)), indent=2, default=str))
        return 0

    if args.command == "trace":
        for event in service.history(args.thread_id):
            print(json.dumps(event, default=str))
        return 0

    return 1


def _print_result(result: Any) -> None:
    print(f"thread_id : {result.thread_id}")
    print(f"status    : {result.status}")
    print(f"risk      : {result.risk_level or '-'}  ({result.risk_reason or 'no rationale recorded'})")
    print(f"evidence  : {result.evidence_grade or '-'}")
    print(f"route     : {' -> '.join(result.route_history) or '-'}")

    if result.awaiting_human_review:
        print("\n=== PAUSED: human sign-off required ===")
        _print_review_payload(result.interrupt_payload or {})
        print(
            "\nResume with:\n"
            f"  python -m cli.review resume {result.thread_id} "
            "--decision approved --reviewer you@example.com"
        )
        return

    if result.final_answer:
        print("\n=== ANSWER ===")
        print(result.final_answer)
    for error in result.errors:
        print(f"\n[error] {error}", file=sys.stderr)


def _print_review_payload(payload: dict[str, Any]) -> None:
    print(f"\nrequest   : {payload.get('question', '')}")
    print(f"risk      : {payload.get('risk_level', '')} -- {payload.get('risk_reason', '')}")
    if payload.get("risk_signals"):
        print(f"signals   : {', '.join(payload['risk_signals'])}")
    print(f"evidence  : {payload.get('evidence_grade', '')}")

    print("\npolicy evidence:")
    for passage in payload.get("policy_passages", []):
        print(f"  [{passage.get('rank')}] {passage.get('citation')} (score={passage.get('score'):.3f})")

    preliminary = payload.get("preliminary_assessment", {})
    print("\npreliminary assessment (NOT an authorisation):")
    print(f"  {preliminary.get('draft_answer', '')}")
    for condition in preliminary.get("conditions", []):
        print(f"  - {condition}")
    if preliminary.get("uncertainty"):
        print(f"  uncertainty: {preliminary['uncertainty']}")
    if payload.get("error"):
        print(f"\n  previous reviewer input rejected: {payload['error']}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())