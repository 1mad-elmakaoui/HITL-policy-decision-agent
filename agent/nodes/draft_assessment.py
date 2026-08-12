"""Node: draft the preliminary policy assessment.
placed before risk scoring so that a high-risk request arrives at the interrupt with an
assessment the reviewer can actually evaluate.

The draft is *always* a draft. On the low-risk path it becomes the answer after
finalisation; on the high-risk path it is evidence put in front of a reviewer.
Neither is authorisation, and the node never says otherwise.

When evidence is absent, the node produces an explicit insufficiency assessment
rather than calling the model at all.  agentic RAG recipe
takes the same line: a run that cannot support an answer "terminates with
status=failed rather than fabricating a supported answer" -- "that terminal
failure is correct pathway behavior, not a system crash".
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from agent.llm.client import LLMClient, LLMError
from agent.llm.prompts import POLICY_ASSESSMENT_SYSTEM, render_policy_assessment_prompt
from agent.observability import EventLog, traced_node
from agent.state import EVIDENCE_NONE, PolicyReviewState

NODE_NAME = "draft_assessment"

NO_EVIDENCE_ANSWER = (
    "No relevant company policy was retrieved for this question, so the assistant cannot "
    "say whether the action is permitted. This is not a statement that the action is "
    "allowed. Refer the question to the policy owner, or re-run ingestion if the relevant "
    "policy should be in the index."
)

RETRIEVAL_FAILED_ANSWER = (
    "Policy evidence could not be retrieved, so no assessment can be made. The request "
    "cannot be resolved until the policy index is available again."
)


def make_draft_assessment_node(
    llm: LLMClient, log_factory: Callable[[], EventLog]
) -> Callable[[PolicyReviewState], Dict[str, Any]]:
    @traced_node(NODE_NAME, log_factory)
    def draft_assessment(state: PolicyReviewState) -> Dict[str, Any]:
        passages: List[Dict[str, Any]] = list(state.get("policy_passages") or [])
        grade = state.get("evidence_grade", EVIDENCE_NONE)

        if state.get("retrieval_error"):
            return _insufficient(RETRIEVAL_FAILED_ANSWER, state["retrieval_error"])
        if not passages or grade == EVIDENCE_NONE:
            return _insufficient(NO_EVIDENCE_ANSWER, "No policy passage cleared the evidence threshold.")

        try:
            raw = llm.complete(
                system=POLICY_ASSESSMENT_SYSTEM,
                prompt=render_policy_assessment_prompt(state.get("question", ""), passages, grade),
            )
            payload = _parse(raw)
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            # A model failure must not become an answer. The request stays
            # unresolved and, if it is high risk, still reaches a human.
            return _insufficient(
                "The assessment step could not be completed, so no position is stated on this "
                "request. The retrieved policy passages are recorded and the request can be "
                "resumed once the fault is cleared.",
                f"assessment failed: {exc}",
            )

        answer = str(payload.get("answer", "")).strip()
        if not answer:
            return _insufficient(NO_EVIDENCE_ANSWER, "The assessment step returned no answer.")

        permitted = str(payload.get("permitted", "unknown")).strip().lower()
        basis = [str(b) for b in payload.get("basis", []) if str(b).strip()]
        # If the assessment cites nothing, fall back to the citations of the
        # passages actually supplied, so provenance is never lost.
        if not basis:
            basis = [str(p.get("citation", "")) for p in passages[:3] if p.get("citation")]

        return {
            "draft_answer": _prefix(permitted, answer),
            "draft_basis": basis,
            "draft_conditions": [str(c) for c in payload.get("conditions", []) if str(c).strip()],
            "draft_uncertainty": str(payload.get("uncertainty", "")).strip()
            or ("Evidence for this question was graded weak." if grade != "strong" else ""),
        }

    return draft_assessment


_PERMISSION_PREFIX = {
    "yes": "Permitted under the retrieved policy.",
    "no": "Not permitted under the retrieved policy.",
    "conditional": "Permitted only subject to conditions.",
    "unknown": "The retrieved policy does not settle this.",
}


def _prefix(permitted: str, answer: str) -> str:
    lead = _PERMISSION_PREFIX.get(permitted, _PERMISSION_PREFIX["unknown"])
    return f"{lead} {answer}".strip()


def _insufficient(answer: str, reason: str) -> Dict[str, Any]:
    return {
        "draft_answer": answer,
        "draft_basis": [],
        "draft_conditions": [],
        "draft_uncertainty": reason,
        "errors": [f"{NODE_NAME}: {reason}"],
    }


def _parse(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("assessment response was not JSON")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("assessment response was not a JSON object")
    return parsed