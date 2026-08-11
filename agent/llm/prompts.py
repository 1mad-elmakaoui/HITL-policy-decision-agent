"""Prompt templates.

Two rules shape everything here:

* Risk classification and policy assessment are **separate** prompts issued by
  **separate** nodes. The specification forbids hiding risk classification
  inside the answer prompt,  -- these are route decisions, not instructions buried in
  prose.
* The assessment prompt is told to answer *from the supplied evidence only*, and
  is given an explicit "insufficient evidence" output. Declining is a supported
  result, not a failure the model has to talk its way out of.

Neither prompt is asked to decide whether a human should review the request.
That decision belongs to the graph.
"""

from __future__ import annotations

from typing import Any, Dict, List

RISK_CLASSIFIER_SYSTEM = """You classify the risk level of a workplace policy question \
for a governance workflow. You do not answer the question and you do not decide \
who approves anything.

Return a single JSON object and nothing else:
{"level": "low" | "medium" | "high", "reason": "<one sentence>", "signals": ["<short tag>", ...]}

Guidance:
- "high": the request concerns a consequential employment action (termination, \
demotion, disciplinary action, suspension, pay reduction or deduction), \
investigating or monitoring a named individual, or any action touching a \
complaint, protected disclosure or protected characteristic.
- "medium": the request seeks an exception to written policy, changes an \
individual's system access, or discloses information externally.
- "low": the request asks how an ordinary, routine process works.

When the request is ambiguous about whether a person is adversely affected, \
choose the higher level."""


POLICY_ASSESSMENT_SYSTEM = """You are a policy analyst. You answer strictly from the \
policy evidence supplied to you.

Hard rules:
- Use ONLY the supplied evidence. If the evidence does not settle the question, \
say so. Never supply a rule from general knowledge.
- Never state or imply that an action is approved. You produce an assessment; \
authorisation is a separate step performed by a human.
- Quote or closely paraphrase the governing wording, and cite the passage it \
came from.

Return a single JSON object and nothing else:
{
  "permitted": "yes" | "no" | "conditional" | "unknown",
  "answer": "<the assessment, in plain language>",
  "basis": ["<citation>", ...],
  "conditions": ["<condition or exception>", ...],
  "uncertainty": "<what the evidence does not settle, or an empty string>"
}"""


def render_risk_classifier_prompt(question: str, policy_passages: List[Dict[str, Any]]) -> str:
    return f"""RISK_CLASSIFICATION_TASK

<QUESTION>
{question.strip()}
</QUESTION>

<POLICY_EVIDENCE>
{render_evidence(policy_passages)}
</POLICY_EVIDENCE>

Classify the risk level of the requested action."""


def render_policy_assessment_prompt(
    question: str, policy_passages: List[Dict[str, Any]], evidence_grade: str
) -> str:
    return f"""POLICY_ASSESSMENT_TASK

<QUESTION>
{question.strip()}
</QUESTION>

<EVIDENCE_GRADE>{evidence_grade}</EVIDENCE_GRADE>

<POLICY_EVIDENCE>
{render_evidence(policy_passages)}
</POLICY_EVIDENCE>

Assess the question against the evidence above. If the evidence grade is \
"weak" or "none", set "permitted" to "unknown" unless the retrieved wording \
plainly settles the question, and state what is missing in "uncertainty"."""


def render_evidence(policy_passages: List[Dict[str, Any]]) -> str:
    """Render passages with rank, citation and score.

    The score is included so the model can see how well-supported each passage
    is, and so the rendered prompt can be reconstructed from state during an
    audit (RAGOps section 5.1, Traceability).
    """
    if not policy_passages:
        return "(no relevant policy passages were retrieved)"
    blocks = []
    for passage in policy_passages:
        blocks.append(
            f"[{passage.get('rank', 0)}] {passage.get('citation', 'unknown source')} "
            f"(score={float(passage.get('score', 0.0)):.3f})\n{passage.get('text', '').strip()}"
        )
    return "\n\n".join(blocks)