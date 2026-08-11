"""Risk classification.

1. A **deterministic** pass over the taxonomy produces a baseline level and a
   rationale drawn from the matched signals. It needs no model, so it is
   reproducible and unit-testable on its own.
2. An **optional model pass** may then *raise* the level. It can never lower one.
   That asymmetry is the governance property: a model cannot talk the workflow
   out of human review, only into it.
3. Any failure in step 2 escalates rather than degrades (``high_risk_on_
   classifier_error``), so an outage routes work to a human instead of past one.

The classifier is a plain object with a plain method. The graph node in
``agent/nodes/classify_risk.py`` is a thin adapter over it, which is what makes
"risk classification must be independently testable without executing the entire
graph" true by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.llm.client import LLMClient, LLMError
from agent.llm.prompts import RISK_CLASSIFIER_SYSTEM, render_risk_classifier_prompt
from agent.risk.taxonomy import RiskSignal, baseline_level, match_signals
from agent.state import RISK_HIGH, RISK_LOW, RISK_ORDER, max_risk
from config.settings import RiskSettings


@dataclass
class RiskAssessment:
    """Structured result of risk classification."""

    level: str
    reason: str
    signals: List[str] = field(default_factory=list)
    classifier: str = "deterministic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "reason": self.reason,
            "signals": list(self.signals),
            "classifier": self.classifier,
        }

    @property
    def requires_human_review(self) -> bool:
        return self.level == RISK_HIGH


class RiskClassifier:
    """Deterministic-first risk classifier with escalate-only model backup."""

    def __init__(
        self,
        settings: Optional[RiskSettings] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self._settings = settings or RiskSettings()
        self._llm = llm

    # ------------------------------------------------------------------ API
    def classify(self, question: str, policy_passages: Optional[List[Dict[str, Any]]] = None) -> RiskAssessment:
        deterministic = self.classify_deterministic(question)

        if not (self._settings.use_llm_second_opinion and self._llm is not None):
            return deterministic
        if deterministic.level == RISK_HIGH:
            # Already at the ceiling; an escalate-only pass cannot change it, so
            # skip the call entirely.
            return deterministic

        try:
            escalated = self._classify_with_llm(question, policy_passages or [])
        except (LLMError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if self._settings.high_risk_on_classifier_error:
                return RiskAssessment(
                    level=RISK_HIGH,
                    reason=(
                        "Risk classification could not be completed, so the request is treated "
                        f"as high risk and routed to human review. Cause: {exc}"
                    ),
                    signals=deterministic.signals + ["classifier_error"],
                    classifier="deterministic+error_escalation",
                )
            return deterministic

        if escalated is None:
            return deterministic

        if self._settings.escalate_only_llm:
            level = max_risk(deterministic.level, escalated.level)
        else:  # pragma: no cover - not reachable under the shipped configuration
            level = escalated.level

        if RISK_ORDER.get(level, 2) > RISK_ORDER.get(deterministic.level, 0):
            return RiskAssessment(
                level=level,
                reason=escalated.reason or deterministic.reason,
                signals=sorted(set(deterministic.signals + escalated.signals)),
                classifier="deterministic+llm_escalation",
            )
        return deterministic

    def classify_deterministic(self, question: str) -> RiskAssessment:
        """The model-free pass. Fully reproducible."""
        matched = match_signals(question or "")
        level = baseline_level(matched)
        return RiskAssessment(
            level=level,
            reason=_explain(level, matched),
            signals=[s.name for s in matched],
            classifier="deterministic",
        )

    # ------------------------------------------------------------- internals
    def _classify_with_llm(
        self, question: str, policy_passages: List[Dict[str, Any]]
    ) -> Optional[RiskAssessment]:
        assert self._llm is not None
        raw = self._llm.complete(
            system=RISK_CLASSIFIER_SYSTEM,
            prompt=render_risk_classifier_prompt(question, policy_passages),
        )
        payload = _extract_json(raw)
        if payload is None:
            return None

        level = str(payload.get("level", "")).strip().lower()
        if level not in RISK_ORDER:
            # An unparseable level is a classifier error, not a low-risk answer.
            raise ValueError(f"model returned an unknown risk level: {level!r}")

        return RiskAssessment(
            level=level,
            reason=str(payload.get("reason", "")).strip(),
            signals=[str(s) for s in payload.get("signals", []) if str(s).strip()],
            classifier="llm",
        )


def _explain(level: str, matched: List[RiskSignal]) -> str:
    if level == RISK_LOW or not matched:
        return (
            "No consequential employment action, policy exception or individual "
            "investigation was identified in the request."
        )
    leading = [s for s in matched if s.level == level]
    rationales = [s.rationale for s in leading] or [s.rationale for s in matched]
    # De-duplicate while preserving order.
    seen: List[str] = []
    for rationale in rationales:
        if rationale not in seen:
            seen.append(rationale)
    return " ".join(seen)


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    parsed = json.loads(text[start : end + 1])
    return parsed if isinstance(parsed, dict) else None