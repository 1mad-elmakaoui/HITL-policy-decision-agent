"""Model clients.

 The examples can run with mock model responses for
repeatable local development or with live providers for integration testing."

``MockLLMClient`` is not a stub that returns a fixed string. It composes its
output from the retrieved passages it is given, so the groundedness property the
workflow depends on -- an answer that cites the evidence and declines when the
evidence is absent -- holds in mock mode too, and can be asserted in tests
without a network call.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol, runtime_checkable

from config.settings import Settings


class LLMError(RuntimeError):
    """Raised when a model call fails.

    Nodes translate this into a controlled workflow state; it never becomes a
    fabricated policy answer.
    """


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class MockLLMClient:
    """Deterministic, evidence-grounded model substitute.

    It recognises the two prompts the workflow issues -- risk classification and
    policy assessment -- from the structured blocks the prompt templates emit,
    and answers each from the supplied evidence alone.
    """

    name = "mock"

    def complete(self, system: str, prompt: str) -> str:
        if "RISK_CLASSIFICATION_TASK" in prompt:
            return self._classify(prompt)
        if "POLICY_ASSESSMENT_TASK" in prompt:
            return self._assess(prompt)
        return json.dumps({"note": "mock client received an unrecognised task"})

    #risk 
    def _classify(self, prompt: str) -> str:
        """Mirror the deterministic taxonomy.

        The escalate-only composition means the mock can only agree with or
        raise the deterministic verdict, so reproducing that verdict keeps mock
        runs faithful to the shipped routing behaviour.
        """
        from agent.risk.taxonomy import baseline_level, match_signals

        question = _extract_block(prompt, "QUESTION")
        matched = match_signals(question)
        return json.dumps(
            {
                "level": baseline_level(matched),
                "reason": (
                    "; ".join(sorted({s.rationale for s in matched}))
                    or "No consequential action identified in the request."
                ),
                "signals": [s.name for s in matched],
            }
        )

    #assessment
    def _assess(self, prompt: str) -> str:
        evidence = _extract_block(prompt, "POLICY_EVIDENCE")
        question = _extract_block(prompt, "QUESTION")

        passages = _parse_evidence(evidence)
        if not passages:
            return json.dumps(
                {
                    "permitted": "unknown",
                    "answer": (
                        "The policy index returned no passage relevant to this question, so no "
                        "position can be stated on it. Escalate to the policy owner rather than "
                        "treating this as permission."
                    ),
                    "basis": [],
                    "conditions": [],
                    "uncertainty": "No relevant policy evidence was retrieved.",
                }
            )

        # Rank the evidence sentences by how much they overlap the question, and
        # keep the best few. Selecting from the passages rather than composing
        # freely is what makes the mock grounded: every sentence it returns is a
        # sentence that was actually retrieved.
        keywords = {w for w in re.findall(r"[a-z]{4,}", question.lower())}
        scored: list[tuple] = []
        for order, passage in enumerate(passages):
            for position, sentence in enumerate(_sentences(passage["text"])):
                overlap = len({w for w in re.findall(r"[a-z]{4,}", sentence.lower())} & keywords)
                if overlap:
                    # Rank by overlap, then by the passage's retrieval rank, then
                    # by position, so the selection is fully deterministic.
                    scored.append((-overlap, order, position, sentence))

        scored.sort()
        picked = [item[3] for item in scored[:3]]
        if not picked:
            picked = _sentences(passages[0]["text"])[:2]

        permitted = _infer_permission(picked)
        conditions = [s for s in picked if re.search(r"requir|approv|must|only|before", s, re.I)][:3]

        return json.dumps(
            {
                "permitted": permitted,
                "answer": " ".join(picked[:3]),
                "basis": [p["citation"] for p in passages[:3]],
                "conditions": conditions,
                "uncertainty": (
                    ""
                    if len(passages) >= 2
                    else "Only one policy passage was retrieved; confirm with the policy owner."
                ),
            }
        )


def _sentences(text: str) -> list[str]:
    """Split passage text into whitespace-normalised sentences.

    The chunker prepends the section heading to the indexed text so the title
    participates in retrieval; it is dropped here so a composed assessment reads
    as prose rather than starting mid-title.
    """
    body = text.split("\n\n", 1)[-1] if "\n\n" in text else text
    body = " ".join(body.split())
    return [s.strip() for s in re.split(r"(?<=[.:])\s+", body) if len(s.strip()) >= 25]


def _infer_permission(sentences: list[str]) -> str:
    joined = " ".join(sentences).lower()
    if re.search(r"\b(is not permitted|are not permitted|is prohibited|may not|are prohibited|not reimbursed|does not permit)\b", joined):
        return "conditional" if re.search(r"\b(only|unless|except|requires)\b", joined) else "no"
    if re.search(r"\b(requires?|only|must|unless|except|subject to|approval)\b", joined):
        return "conditional"
    if re.search(r"\b(is permitted|are permitted|may|entitled|is allowed)\b", joined):
        return "yes"
    return "conditional"


class AnthropicLLMClient:
    """Live provider."""

    def __init__(self, model: str, max_tokens: int = 1200, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - live-only path
            raise LLMError(
                "anthropic is not installed. Install the 'live' extra, or run with "
                "POLICY_REVIEW_RUN_MODE=mock."
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set; cannot run in live mode.")
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self._max_tokens = max_tokens
        self.name = model

    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover - live-only path
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"model call failed: {exc}") from exc
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def build_llm_client(settings: Settings) -> LLMClient:
    provider = (settings.llm_provider or "mock").lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "anthropic":
        return AnthropicLLMClient(settings.llm_model, settings.llm_max_tokens)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def _extract_block(prompt: str, name: str) -> str:
    match = re.search(rf"<{name}>(.*?)</{name}>", prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_evidence(block: str) -> list[dict[str, Any]]:
    """Parse the numbered evidence block the prompt template renders."""
    passages: list[dict[str, Any]] = []
    for raw in re.split(r"\n(?=\[\d+\])", block):
        raw = raw.strip()
        if not raw:
            continue
        header = re.match(r"\[(\d+)\]\s*(.*?)\s*\(score=([\d.]+)\)\s*\n(.*)", raw, flags=re.DOTALL)
        if header:
            passages.append(
                {
                    "rank": int(header.group(1)),
                    "citation": header.group(2).strip(),
                    "score": float(header.group(3)),
                    "text": header.group(4).strip(),
                }
            )
    return passages