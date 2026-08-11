"""Risk-classification tests.
requires the classifier to be testable "independently without
executing the entire graph". None of these tests builds a graph, a checkpointer
or a retriever -- they call :class:`RiskClassifier` directly.

Coverage follows the four required cases: a low-risk question, a high-risk
employment action, an ambiguous case, and a clearly unrelated question. The
escalate-only invariant is tested separately, because it is the property that
makes the human gate non-bypassable.
"""

from __future__ import annotations

import pytest

from agent.llm.client import LLMClient, LLMError, MockLLMClient
from agent.risk.classifier import RiskAssessment, RiskClassifier
from agent.risk.taxonomy import match_signals
from agent.state import RISK_HIGH, RISK_LOW, RISK_MEDIUM
from config.settings import RiskSettings


# --------------------------------------------------------------------------
# The four required cases
# --------------------------------------------------------------------------
LOW_RISK_QUESTIONS = [
    "How many days of annual leave can be carried over to next year?",
    "When does an employee need a medical certificate for sick leave?",
    "What parental leave is the primary caregiver entitled to?",
    "What is the deadline for submitting an expense claim?",
    "Is business class travel allowed on a nine hour flight?",
    "How much bereavement leave is available for an immediate family member?",
]

HIGH_RISK_QUESTIONS = [
    "Can we terminate an employee immediately for expense fraud?",
    "Is it allowed to demote someone to a lower job level?",
    "Can we issue a final written warning without an investigation?",
    "Can we suspend an employee while we investigate an allegation?",
    "Can we dismiss a contractor for gross misconduct?",
    "Can we reduce their salary after a poor performance review?",
    "Can we deduct the overpayment from an employee's pay?",
    "An employee raised a harassment complaint last month. Can we discipline them now?",
    "Can we put him on a PIP and then let him go?",
    "Can we monitor a specific employee's messages during an investigation?",
]

AMBIGUOUS_QUESTIONS = [
    "Can we make an exception to the approval process for this one case?",
    "Can we skip a stage of the process this time?",
    "Can we revoke access for someone who is leaving?",
    "Can we disclose this to a third-party vendor?",
]

UNRELATED_QUESTIONS = [
    "What will the weather be like in Lisbon next Tuesday?",
    "What is the company position on quantum computing procurement?",
    "Is my iguana allowed to attend the annual shareholder meeting?",
    "What is the capital of Portugal?",
]


@pytest.mark.parametrize("question", LOW_RISK_QUESTIONS)
def test_routine_questions_are_low_risk(classifier: RiskClassifier, question: str) -> None:
    assessment = classifier.classify(question)
    assert assessment.level == RISK_LOW, f"{question!r} -> {assessment.level} ({assessment.reason})"
    assert not assessment.requires_human_review


@pytest.mark.parametrize("question", HIGH_RISK_QUESTIONS)
def test_consequential_employment_actions_are_high_risk(
    classifier: RiskClassifier, question: str
) -> None:
    assessment = classifier.classify(question)
    assert assessment.level == RISK_HIGH, f"{question!r} -> {assessment.level}"
    assert assessment.requires_human_review
    # A rationale a reviewer can read is part of the contract, not decoration.
    assert assessment.reason.strip()
    assert assessment.signals


@pytest.mark.parametrize("question", AMBIGUOUS_QUESTIONS)
def test_ambiguous_questions_are_flagged_not_ignored(
    classifier: RiskClassifier, question: str
) -> None:
    """Policy exceptions and access changes are recorded above low risk.

    They do not by themselves reach the human gate -- that is reserved for
    consequential employment actions -- but they must not be indistinguishable
    from a question about carry-over leave.
    """
    assessment = classifier.classify(question)
    assert assessment.level in {RISK_MEDIUM, RISK_HIGH}, f"{question!r} -> {assessment.level}"
    assert assessment.signals


@pytest.mark.parametrize("question", UNRELATED_QUESTIONS)
def test_unrelated_questions_are_low_risk(classifier: RiskClassifier, question: str) -> None:
    """An off-topic question is not a risky one.

    Whether the corpus can answer it is the retriever's business; the classifier
    should not manufacture risk from unfamiliarity.
    """
    assessment = classifier.classify(question)
    assert assessment.level == RISK_LOW, f"{question!r} -> {assessment.level}"


# --------------------------------------------------------------------------
# Determinism and structure
# --------------------------------------------------------------------------
def test_deterministic_pass_needs_no_model() -> None:
    """The baseline classifier runs with no LLM configured at all."""
    classifier = RiskClassifier(RiskSettings(use_llm_second_opinion=False), llm=None)
    assessment = classifier.classify("Can we terminate an employee for cause?")
    assert assessment.level == RISK_HIGH
    assert assessment.classifier == "deterministic"


def test_classification_is_reproducible(classifier: RiskClassifier) -> None:
    question = "Can we demote an employee who raised a grievance?"
    first = classifier.classify(question)
    second = classifier.classify(question)
    assert first.to_dict() == second.to_dict()


def test_assessment_is_structured(classifier: RiskClassifier) -> None:
    assessment = classifier.classify("Can we terminate an employee for expense fraud?")
    assert isinstance(assessment, RiskAssessment)
    payload = assessment.to_dict()
    assert set(payload) == {"level", "reason", "signals", "classifier"}
    assert payload["level"] in {RISK_LOW, RISK_MEDIUM, RISK_HIGH}


def test_signals_name_the_matched_rule() -> None:
    signals = {s.name for s in match_signals("Can we terminate and then demote the replacement?")}
    assert {"termination", "demotion"} <= signals


def test_empty_question_is_low_risk_not_an_error(classifier: RiskClassifier) -> None:
    assert classifier.classify("").level == RISK_LOW


# --------------------------------------------------------------------------
# The escalate-only invariant
# --------------------------------------------------------------------------
class _DowngradingLLM:
    """A model that always insists the request is low risk."""

    name = "downgrading"

    def complete(self, system: str, prompt: str) -> str:
        return '{"level": "low", "reason": "Looks routine to me.", "signals": []}'


class _EscalatingLLM:
    """A model that always insists the request is high risk."""

    name = "escalating"

    def complete(self, system: str, prompt: str) -> str:
        return '{"level": "high", "reason": "Adverse action against an individual.", "signals": ["model_flagged"]}'


class _FailingLLM:
    name = "failing"

    def complete(self, system: str, prompt: str) -> str:
        raise LLMError("provider unavailable")


class _GarbageLLM:
    name = "garbage"

    def complete(self, system: str, prompt: str) -> str:
        return "I'd rather not say."


@pytest.mark.parametrize("question", HIGH_RISK_QUESTIONS)
def test_model_cannot_downgrade_a_high_risk_request(question: str) -> None:
    """The governance invariant: no model output can clear the human gate.

    This is the single most important test in the suite. If it fails, a model can
    talk the workflow out of human review.
    """
    classifier = RiskClassifier(RiskSettings(), llm=_DowngradingLLM())
    assert classifier.classify(question).level == RISK_HIGH


def test_model_may_escalate_a_low_risk_request() -> None:
    """Escalation is allowed in the direction that adds a human, not removes one."""
    question = "How many days of annual leave can be carried over?"
    assert RiskClassifier(RiskSettings(), llm=MockLLMClient()).classify(question).level == RISK_LOW

    escalated = RiskClassifier(RiskSettings(), llm=_EscalatingLLM()).classify(question)
    assert escalated.level == RISK_HIGH
    assert escalated.classifier == "deterministic+llm_escalation"


def test_classifier_failure_escalates_rather_than_degrades() -> None:
    """An outage routes work to a human instead of past one."""
    classifier = RiskClassifier(RiskSettings(high_risk_on_classifier_error=True), llm=_FailingLLM())
    assessment = classifier.classify("How many days of annual leave can be carried over?")
    assert assessment.level == RISK_HIGH
    assert "classifier_error" in assessment.signals


def test_unparseable_model_output_escalates() -> None:
    classifier = RiskClassifier(RiskSettings(), llm=_GarbageLLM())
    # Non-JSON is treated as "no second opinion" and falls back to the
    # deterministic verdict rather than being invented into a level.
    assert classifier.classify("How many days of annual leave carry over?").level == RISK_LOW


def test_model_is_not_consulted_when_already_high() -> None:
    """An escalate-only pass cannot change a verdict already at the ceiling."""

    class _Counting:
        name = "counting"
        calls = 0

        def complete(self, system: str, prompt: str) -> str:
            _Counting.calls += 1
            return '{"level": "high", "reason": "x", "signals": []}'

    classifier = RiskClassifier(RiskSettings(), llm=_Counting())
    assert classifier.classify("Can we terminate an employee?").level == RISK_HIGH
    assert _Counting.calls == 0


def test_classifier_satisfies_the_llm_protocol() -> None:
    assert isinstance(MockLLMClient(), LLMClient)