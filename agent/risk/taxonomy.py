"""The high-risk action taxonomy.This module is
that domain logic, kept as data rather than prose so it is reviewable by the
people who own the policy, and testable without a model.

A signal is a named pattern with a risk level and a rationale. Keeping the
rationale beside the pattern is what lets the classifier explain itself: the
reason a request was escalated is looked up, not generated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from agent.state import RISK_HIGH, RISK_LOW, RISK_MEDIUM


@dataclass(frozen=True)
class RiskSignal:
    name: str
    level: str
    rationale: str
    patterns: Tuple[str, ...]

    def matches(self, text: str) -> bool:
        return any(re.search(p, text, flags=re.IGNORECASE) for p in self.patterns)


# Consequential employment actions. These are the cases the specification names
# explicitly (termination, demotion, disciplinary action) plus the actions that
# carry the same employment consequence under the corpus's own policies.
HIGH_RISK_SIGNALS: Tuple[RiskSignal, ...] = (
    RiskSignal(
        name="termination",
        level=RISK_HIGH,
        rationale="The request concerns ending someone's employment.",
        patterns=(
            r"\bterminat(e|ing|ion|ed)\b",
            r"\bdismiss(al|ing|ed)?\b",
            r"\bfir(e|ing|ed)\b(?!\s*(drill|alarm|safety|extinguisher))",
            r"\blet\s+(them|him|her|someone|an?\s+employee)\s+go\b",
            r"\bmade?\s+redundan(t|cy)\b",
            r"\bredundanc(y|ies)\b",
            r"\bsever\s+(the\s+)?employment\b",
            r"\boffboard(ing)?\s+(them|for\s+cause)\b",
        ),
    ),
    RiskSignal(
        name="demotion",
        level=RISK_HIGH,
        rationale="The request concerns reducing someone's role, level or pay.",
        patterns=(
            r"\bdemot(e|ing|ion|ed)\b",
            r"\bdowngrad(e|ing|ed)\s+(their|his|her|the)\s+(role|level|title|grade)\b",
            r"\breduc(e|ing|tion)\s+(their|his|her|the)\s+(pay|salary|compensation|level|grade|title)\b",
            r"\bcut\s+(their|his|her)\s+(pay|salary)\b",
            r"\bstrip\s+(them|him|her)\s+of\b",
        ),
    ),
    RiskSignal(
        name="disciplinary_action",
        level=RISK_HIGH,
        rationale="The request concerns formal disciplinary action against an employee.",
        patterns=(
            r"\bdisciplin(e|ary|ing|ed)\b",
            r"\bwritten\s+warning\b",
            r"\bfinal\s+warning\b",
            r"\bformal\s+warning\b",
            r"\bwrite\s+(them|him|her)\s+up\b",
            r"\bperformance\s+improvement\s+plan\b",
            r"\bput\s+(them|him|her)\s+on\s+a\s+pip\b",
            r"\bsanction(s|ing|ed)?\b",
        ),
    ),
    RiskSignal(
        name="suspension",
        level=RISK_HIGH,
        rationale="The request concerns suspending an employee, which affects their status and pay.",
        patterns=(
            r"\bsuspend(ing|ed|sion)?\b",
            r"\bplace\s+(them|him|her)\s+on\s+(unpaid\s+)?leave\b",
            r"\bgarden(ing)?\s+leave\b",
        ),
    ),
    RiskSignal(
        name="investigation_of_individual",
        level=RISK_HIGH,
        rationale="The request concerns investigating or monitoring a named individual.",
        patterns=(
            r"\binvestigat(e|ing|ion)\s+(an?\s+)?(employee|individual|person|staff|worker|them|him|her)\b",
            r"\bmonitor(ing)?\s+(an?\s+)?(employee|individual|specific|named|their)\b",
            r"\bcovert(ly)?\s+(monitor|record|observ)",
            r"\bsurveil(lance|ling)?\b",
            r"\bsearch\s+(their|his|her)\s+(email|messages|device|laptop)\b",
        ),
    ),
    RiskSignal(
        name="compensation_deduction",
        level=RISK_HIGH,
        rationale="The request concerns withholding or deducting money from an employee's pay.",
        patterns=(
            r"\bdeduct(ing|ion)?\b.{0,40}\b(pay|salary|wages)\b",
            r"\bwithhold(ing)?\b.{0,40}\b(pay|salary|wages|bonus)\b",
            r"\bclaw\s*back\b",
            r"\bdock\s+(their|his|her)\s+(pay|wages)\b",
        ),
    ),
    RiskSignal(
        name="protected_activity",
        level=RISK_HIGH,
        rationale=(
            "The request touches protected activity (a complaint, disclosure or protected "
            "characteristic), where an adverse action carries retaliation exposure."
        ),
        patterns=(
            r"\bwhistleblow(er|ing)?\b",
            r"\bprotected\s+(disclosure|characteristic|activity)\b",
            r"\bretaliat(e|ion|ory)\b",
            r"\braised?\s+a\s+(complaint|grievance|concern)\b",
            r"\bfiled?\s+a\s+(complaint|grievance)\b",
            r"\bdiscriminat(e|ion|ory)\b",
            r"\bharassment\b",
            r"\bunion\s+(rep|representative|organi[sz]ing|activity)\b",
        ),
    ),
    RiskSignal(
        name="contract_or_legal_exposure",
        level=RISK_HIGH,
        rationale="The request concerns a change to contractual terms or carries direct legal exposure.",
        patterns=(
            r"\bbreach\s+of\s+contract\b",
            r"\bchange\s+(their|his|her|the)\s+(contract|terms|notice\s+period)\b",
            r"\bwithout\s+notice\b",
            r"\bimmediate(ly)?\b.{0,40}\b(terminat|dismiss|fir(e|ing)|remov)",
            r"\b(terminat|dismiss|fir(e|ing)|remov)\w*\b.{0,40}\bimmediate(ly)?\b",
            r"\bsettlement\s+agreement\b",
            r"\bnon[\s-]?compete\b",
        ),
    ),
)

# Actions with real consequence that nonetheless sit below the human-review gate
# on their own. They raise the level to medium, which is recorded and visible but
# does not by itself interrupt.
MEDIUM_RISK_SIGNALS: Tuple[RiskSignal, ...] = (
    RiskSignal(
        name="access_revocation",
        level=RISK_MEDIUM,
        rationale="The request concerns changing or revoking an individual's system access.",
        patterns=(
            r"\brevok(e|ing)\b.{0,30}\baccess\b",
            r"\bremov(e|ing)\b.{0,30}\b(access|permissions|account)\b",
            r"\bdisabl(e|ing)\b.{0,30}\b(account|access)\b",
        ),
    ),
    RiskSignal(
        name="exception_request",
        level=RISK_MEDIUM,
        rationale="The request asks to depart from the written policy, which needs an owner's decision.",
        patterns=(
            r"\bexception\s+to\b",
            r"\bwaiv(e|er|ing)\b",
            r"\boverrid(e|ing)\b.{0,30}\bpolicy\b",
            r"\bskip\b.{0,30}\b(step|stage|approval|process)\b",
            r"\bbypass(ing)?\b",
            r"\bwithout\s+(going\s+through|following)\b",
        ),
    ),
    RiskSignal(
        name="third_party_disclosure",
        level=RISK_MEDIUM,
        rationale="The request concerns disclosing information outside the company.",
        patterns=(
            r"\bshar(e|ing)\b.{0,30}\b(with|to)\s+(a\s+)?(third[\s-]?party|vendor|external)\b",
            r"\bdisclos(e|ing|ure)\b",
            r"\bexport\b.{0,30}\bdata\b",
        ),
    ),
)

ALL_SIGNALS: Tuple[RiskSignal, ...] = HIGH_RISK_SIGNALS + MEDIUM_RISK_SIGNALS


def match_signals(text: str) -> List[RiskSignal]:
    """Every signal whose pattern appears in ``text``, high-risk first."""
    matched = [signal for signal in ALL_SIGNALS if signal.matches(text)]
    return sorted(matched, key=lambda s: 0 if s.level == RISK_HIGH else 1)


def baseline_level(matched: List[RiskSignal]) -> str:
    if any(s.level == RISK_HIGH for s in matched):
        return RISK_HIGH
    if any(s.level == RISK_MEDIUM for s in matched):
        return RISK_MEDIUM
    return RISK_LOW