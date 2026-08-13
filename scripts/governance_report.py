"""Turn a CI run's artifacts into a governance evidence bundle.

This is not a rubber stamp. It reads the audit trail the run actually produced
and fails if the human-oversight gate did not fire, because a build where the
gate never engaged is not evidence that the gate works.

    python scripts/governance_report.py evidence/ --out compliance-evidence.md

`evidence/` is the directory `actions/download-artifact` populates: one
subdirectory per uploaded artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Obligation -> what in this repository discharges it. Kept here rather than in
# prose so it stays next to the check that proves each row.
AI_ACT_MAP = [
    ("Art. 9  Risk management",
     "Risk taxonomy and classifier, tested independently of the graph",
     "tests/risk"),
    ("Art. 10 Data governance",
     "Corpus verification, version reconciliation, provenance on every chunk",
     "tests/ingestion + the ingestion pipeline's verification step"),
    ("Art. 11 Technical documentation",
     "Architecture, design decisions and this bundle, generated per run",
     "docs/ + README.md"),
    ("Art. 12 Record-keeping",
     "Append-only event log of every node transition, keyed by request",
     ".policy_state/events.jsonl"),
    ("Art. 13 Transparency to deployers",
     "Draft is labelled an assessment, not an authorisation; run mode is shown",
     "agent/nodes/draft_assessment.py"),
    ("Art. 14 Human oversight",
     "Execution suspends before any high-risk answer exists; sign-off is named",
     "agent/nodes/interrupt_for_review.py + tests/graph"),
    ("Art. 15 Accuracy and robustness",
     "Retrieval evaluated against a labelled set; the run fails below threshold",
     "ingestion/evaluation + tests/retrieval"),
]


def read_events(root: Path) -> List[Dict]:
    """Every audit event, tagged with the artifact it came from.

    One artifact per runner, so the tag is what lets the report distinguish
    "the gate fired twice" from "the gate fired once on each of two runners".
    """
    events: List[Dict] = []
    for path in sorted(root.rglob("events.jsonl")):
        source = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["_source"] = source
            events.append(record)
    return events


def read_index_report(root: Path) -> Dict:
    for path in root.rglob("index-report.json"):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return {}


def verify(events: List[Dict]) -> List[str]:
    """Return a list of failures. Empty means the run is sound."""
    failures: List[str] = []

    if not events:
        return ["No audit trail was found. The orchestration job produced no record."]

    paused = [e for e in events if e.get("event") == "node.paused"]
    if not paused:
        failures.append(
            "The oversight gate never fired in this run. Either no high-risk "
            "request was exercised, or it was answered without pausing."
        )

    # A pause that is logged as a crash is a defective record even when the
    # behaviour was correct, so the trail is checked for that specifically.
    errors = [e for e in events if e.get("event") == "node.error"]
    interrupt_errors = [e for e in errors if e.get("node") == "interrupt_for_review"]
    if interrupt_errors:
        failures.append(
            f"{len(interrupt_errors)} suspension(s) were recorded as errors. "
            "The gate worked but the audit trail misreports it."
        )

    resumed = [e for e in events if e.get("event") == "request.resumed"]
    for event in resumed:
        if not event.get("reviewer_id"):
            failures.append(
                f"Thread {event.get('thread_id', '?')} was resumed without a "
                "named reviewer."
            )

    return failures


def render(events: List[Dict], index: Dict, failures: List[str]) -> str:
    paused = [e for e in events if e.get("event") == "node.paused"]
    resumed = [e for e in events if e.get("event") == "request.resumed"]
    threads = {e.get("thread_id") for e in events if e.get("thread_id")}

    lines: List[str] = []
    add = lines.append

    add("## Governance evidence")
    add("")
    add(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"from commit `{os.environ.get('GITHUB_SHA', 'local')[:12]}`.")
    add("")

    if failures:
        add("### Result: FAILED")
        add("")
        for failure in failures:
            add(f"- {failure}")
    else:
        add("### Result: passed")
        add("")
        add("The human-oversight gate engaged during this run, every resumption "
            "carried a named reviewer, and no suspension was misrecorded as a "
            "failure.")
    add("")

    add("### What this run exercised")
    add("")
    sources = sorted({e.get("_source", "?") for e in events})
    add(f"Aggregated over {len(sources)} audit trail(s): {', '.join(sources)}.")
    add("")
    add("| Audit trail | Requests | Suspended | Resumed by a named reviewer | Events |")
    add("|---|---|---|---|---|")
    for source in sources:
        rows = [e for e in events if e.get("_source") == source]
        add(
            f"| {source} "
            f"| {len({r.get('thread_id') for r in rows if r.get('thread_id')})} "
            f"| {sum(1 for r in rows if r.get('event') == 'node.paused')} "
            f"| {sum(1 for r in rows if r.get('event') == 'request.resumed')} "
            f"| {len(rows)} |"
        )
    add("")
    add("| Signal | Total |")
    add("|---|---|")
    add(f"| Distinct requests traced | {len(threads)} |")
    add(f"| Executions suspended for human review | {len(paused)} |")
    add(f"| Resumptions with a named reviewer | {len(resumed)} |")
    add(f"| Audit events recorded | {len(events)} |")
    if index:
        chunks = index.get("chunk_count", index.get("chunks", "n/a"))
        add(f"| Chunks in the evaluated index | {chunks} |")
        add(f"| Retrieval backend | {index.get('backend', 'n/a')} / {index.get('similarity', 'n/a')} |")
    add("")

    add("### Obligation coverage")
    add("")
    add("The system answers questions about employment actions including "
        "termination and demotion, which places it in Annex III point 4 of the "
        "EU AI Act. The obligations that follow, and what discharges each:")
    add("")
    add("| Obligation | Discharged by | Where |")
    add("|---|---|---|")
    for obligation, mechanism, where in AI_ACT_MAP:
        add(f"| {obligation} | {mechanism} | `{where}` |")
    add("")
    add("Retained artifacts for this run: the audit trail, the index report and "
        "this bundle.")

    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("compliance-evidence.md"))
    args = parser.parse_args(argv)

    events = read_events(args.evidence_dir)
    index = read_index_report(args.evidence_dir)
    failures = verify(events)

    report = render(events, index, failures)
    args.out.write_text(report, encoding="utf-8")
    print(report)

    if failures:
        print("Governance check failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
