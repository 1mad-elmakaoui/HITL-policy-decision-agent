"""The runtime service.

The API a business process calls: submit a question, discover that it is parked
awaiting sign-off, and later resume it.

The central design point is that every call opens its own checkpointer, builds
its own graph, and closes them again. Nothing about a request lives in this
process between calls -- ``submit`` and ``resume`` can run in different
processes, on different machines, days apart, and the only thing that connects
them is the ``thread_id`` and the checkpoint database. 
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from langgraph.types import Command

from agent.checkpointing import checkpointer_scope, thread_config
from agent.graph import compile_policy_review_graph
from agent.human_review.decision import InvalidReviewInput
from agent.llm.client import LLMClient
from agent.observability import EventLog, build_event_log
from agent.retrieval.retriever import PolicyRetriever
from agent.risk.classifier import RiskClassifier
from agent.state import (
    STATUS_FAILED,
    STATUS_PENDING_REVIEW,
    PolicyReviewState,
    new_request_state,
)
from config.settings import Settings, load_settings


class UnknownThread(KeyError):
    """Raised when a thread_id has no checkpointed state."""


class NotAwaitingReview(RuntimeError):
    """Raised when a resume is attempted on a thread that is not paused."""


@dataclass
class PolicyReviewResult:
    """The outcome of an invocation."""

    thread_id: str
    status: str
    final_answer: str = ""
    interrupt_payload: Optional[Dict[str, Any]] = None
    risk_level: str = ""
    risk_reason: str = ""
    evidence_grade: str = ""
    decision_record: Dict[str, Any] = field(default_factory=dict)
    route_history: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def awaiting_human_review(self) -> bool:
        return self.interrupt_payload is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "status": self.status,
            "awaiting_human_review": self.awaiting_human_review,
            "risk_level": self.risk_level,
            "risk_reason": self.risk_reason,
            "evidence_grade": self.evidence_grade,
            "final_answer": self.final_answer,
            "interrupt_payload": self.interrupt_payload,
            "decision_record": self.decision_record,
            "route_history": self.route_history,
            "errors": self.errors,
        }


class PolicyReviewService:
    """Submit, inspect and resume policy-review requests."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retriever: Optional[PolicyRetriever] = None,
        classifier: Optional[RiskClassifier] = None,
        llm: Optional[LLMClient] = None,
        log: Optional[EventLog] = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._retriever = retriever
        self._classifier = classifier
        self._llm = llm
        self._log = log if log is not None else build_event_log(self._settings.observability_path)

    # ---------------------------------------------------------------- submit
    def submit(
        self,
        question: str,
        request_id: Optional[str] = None,
        requested_by: str = "",
        force_human_review: bool = False,
    ) -> PolicyReviewResult:
        """Start a review. Returns either a final answer or a pending interrupt."""
        thread_id = request_id or f"pr-{uuid.uuid4().hex[:12]}"
        initial = new_request_state(
            question=question,
            request_id=thread_id,
            requested_by=requested_by,
            force_human_review=force_human_review,
        )

        self._log.emit(
            "request.submitted",
            thread_id,
            question=question,
            requested_by=requested_by,
            force_human_review=force_human_review,
        )

        with checkpointer_scope(self._settings.checkpoint_path) as checkpointer:
            graph = self._compile(checkpointer)
            config = thread_config(thread_id)
            output = graph.invoke(initial, config)
            return self._result(thread_id, output, graph, config)

    # ---------------------------------------------------------------- resume
    def resume(self, thread_id: str, reviewer_input: Dict[str, Any]) -> PolicyReviewResult:
        """Resume a parked request with the reviewer's decision.

        Requires nothing from the submitting process: the graph is rebuilt here
        and the state comes from the checkpoint.
        """
        with checkpointer_scope(self._settings.checkpoint_path) as checkpointer:
            graph = self._compile(checkpointer)
            config = thread_config(thread_id)

            snapshot = graph.get_state(config)
            if not snapshot.created_at:
                raise UnknownThread(
                    f"no checkpoint found for thread_id {thread_id!r}; nothing to resume"
                )
            if not snapshot.interrupts and not snapshot.next:
                raise NotAwaitingReview(
                    f"thread {thread_id!r} is not awaiting review (status="
                    f"{snapshot.values.get('status', 'unknown')!r})"
                )

            # Structural pre-check. LangGraph treats a falsy `resume` value as
            # "no value supplied" and re-runs the interrupt without entering the
            # node body -- so an empty payload would leave the thread parked with
            # no explanation of why. Rejecting it here turns a silent no-op into
            # an error the caller can act on. The thread is untouched either way;
            # the authoritative validation still happens in the interrupt node.
            if not isinstance(reviewer_input, Mapping) or not reviewer_input:
                raise InvalidReviewInput(
                    "reviewer input must be a non-empty mapping containing 'decision' and "
                    f"'reviewer_id'; got {type(reviewer_input).__name__}"
                )

            self._log.emit(
                "request.resumed",
                thread_id,
                reviewer_id=reviewer_input.get("reviewer_id", ""),
                decision=reviewer_input.get("decision", ""),
            )

            output = graph.invoke(Command(resume=reviewer_input), config)
            return self._result(thread_id, output, graph, config)

    # ------------------------------------------------------------ inspection
    def get_state(self, thread_id: str) -> PolicyReviewState:
        with checkpointer_scope(self._settings.checkpoint_path) as checkpointer:
            graph = self._compile(checkpointer)
            snapshot = graph.get_state(thread_config(thread_id))
            if not snapshot.created_at:
                raise UnknownThread(f"no checkpoint found for thread_id {thread_id!r}")
            return snapshot.values

    def pending_review(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """The interrupt payload for a parked thread, or ``None``.

        This is what an operator UI or queue consumer polls. It reads the
        checkpoint; it does not re-run any part of the workflow.
        """
        with checkpointer_scope(self._settings.checkpoint_path) as checkpointer:
            graph = self._compile(checkpointer)
            snapshot = graph.get_state(thread_config(thread_id))
            if not snapshot.created_at:
                raise UnknownThread(f"no checkpoint found for thread_id {thread_id!r}")
            return _interrupt_payload_from_snapshot(snapshot)

    def history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Observability events recorded for this thread."""
        return self._log.read(thread_id)

    def checkpoint_count(self, thread_id: str) -> int:
        """Number of persisted checkpoints. Used by the persistence tests."""
        with checkpointer_scope(self._settings.checkpoint_path) as checkpointer:
            graph = self._compile(checkpointer)
            return sum(1 for _ in graph.get_state_history(thread_config(thread_id)))

    # ------------------------------------------------------------- internals
    def _compile(self, checkpointer: Any) -> Any:
        return compile_policy_review_graph(
            checkpointer=checkpointer,
            settings=self._settings,
            retriever=self._retriever,
            classifier=self._classifier,
            llm=self._llm,
            log=self._log,
        )

    def _result(
        self, thread_id: str, output: Dict[str, Any], graph: Any, config: Dict[str, Any]
    ) -> PolicyReviewResult:
        payload = _interrupt_payload_from_output(output)
        if payload is None:
            payload = _interrupt_payload_from_snapshot(graph.get_state(config))

        status = output.get("status") or (STATUS_PENDING_REVIEW if payload else STATUS_FAILED)

        return PolicyReviewResult(
            thread_id=thread_id,
            status=status,
            final_answer=output.get("final_answer", ""),
            interrupt_payload=payload,
            risk_level=output.get("risk_level", ""),
            risk_reason=output.get("risk_reason", ""),
            evidence_grade=output.get("evidence_grade", ""),
            decision_record=output.get("decision_record", {}) or {},
            route_history=list(output.get("route_history") or []),
            errors=list(output.get("errors") or []),
        )


def _interrupt_payload_from_output(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the interrupt payload from an ``invoke`` return value."""
    interrupts = output.get("__interrupt__") if isinstance(output, dict) else None
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"value": value}


def _interrupt_payload_from_snapshot(snapshot: Any) -> Optional[Dict[str, Any]]:
    """Extract the pending interrupt payload from a state snapshot.

    Read from the checkpoint rather than from an invocation result, so a
    different process can discover what a parked thread is waiting for.
    """
    interrupts = getattr(snapshot, "interrupts", None) or ()
    for item in interrupts:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            return value

    # Older/newer LangGraph releases surface pending interrupts on the tasks.
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                return value
    return None