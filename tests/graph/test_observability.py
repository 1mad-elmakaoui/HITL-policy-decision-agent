"""Observability tests"""

from __future__ import annotations

from agent.runtime import PolicyReviewService
from agent.state import RISK_HIGH


def test_every_node_transition_is_logged(
    service: PolicyReviewService, low_risk_question: str
) -> None:
    service.submit(low_risk_question, request_id="obs-1")
    events = service.history("obs-1")

    nodes_started = {e["node"] for e in events if e["event"] == "node.start"}
    nodes_ended = {e["node"] for e in events if e["event"] == "node.end"}

    assert nodes_started == nodes_ended
    assert {"retrieve_policy", "draft_assessment", "classify_risk", "finalize_decision"} <= (
        nodes_started
    )


def test_retrieved_chunks_are_traceable_from_the_log(
    service: PolicyReviewService, low_risk_question: str
) -> None:
    """RAGOps 5.1 traceability: link a query to the retrieved chunk ids."""
    service.submit(low_risk_question, request_id="obs-2")
    events = service.history("obs-2")

    retrieval = next(e for e in events if e.get("node") == "retrieve_policy" and "retrieved" in e)
    assert retrieval["retrieved"]
    for entry in retrieval["retrieved"]:
        assert entry["chunk_id"]
        assert entry["citation"]
        assert entry["score"] is not None


def test_risk_classification_and_rationale_are_logged(
    service: PolicyReviewService, high_risk_question: str
) -> None:
    service.submit(high_risk_question, request_id="obs-3")
    events = service.history("obs-3")

    classification = next(e for e in events if e.get("node") == "classify_risk" and "risk_level" in e)
    assert classification["risk_level"] == RISK_HIGH
    assert classification["risk_signals"]

    requested = next(e for e in events if e["event"] == "human_review.requested")
    assert requested["risk_reason"]


def test_the_human_review_lifecycle_is_logged(
    service: PolicyReviewService, high_risk_question: str
) -> None:
    service.submit(high_risk_question, request_id="obs-4")
    service.resume(
        "obs-4", {"decision": "approved", "reviewer_id": "hr@example.com", "feedback": "ok"}
    )
    events = {e["event"] for e in service.history("obs-4")}

    assert {
        "request.submitted",
        "human_review.requested",
        "request.resumed",
        "human_review.recorded",
        "decision.finalized",
    } <= events


def test_every_event_carries_the_thread_id(
    service: PolicyReviewService, high_risk_question: str
) -> None:
    service.submit(high_risk_question, request_id="obs-5")
    events = service.history("obs-5")
    assert events
    assert all(e["thread_id"] == "obs-5" for e in events)


def test_events_record_the_process_that_produced_them(
    service: PolicyReviewService, low_risk_question: str
) -> None:
    """A workflow spanning restarts needs to say which process did what."""
    service.submit(low_risk_question, request_id="obs-6")
    assert all("pid" in e and "ts" in e for e in service.history("obs-6"))


def test_the_decision_record_explains_why_the_answer_was_produced(
    service: PolicyReviewService, high_risk_question: str
) -> None:
    """Specification section 16: it must be possible to understand the outcome."""
    service.submit(high_risk_question, request_id="obs-7")
    result = service.resume(
        "obs-7",
        {"decision": "approved", "reviewer_id": "hr@example.com", "feedback": "Proceed."},
    )
    record = result.decision_record

    assert record["request"]["question"] == high_risk_question
    assert record["risk"]["level"] == RISK_HIGH
    assert record["risk"]["reason"]
    assert record["risk"]["classifier"]
    assert record["evidence"]["passages"]
    assert record["human_review"]["reviewer_id"] == "hr@example.com"
    assert record["route_history"]
    assert record["status"]


def test_retrieval_risk_review_and_answer_are_separate_steps(
    service: PolicyReviewService, high_risk_question: str
) -> None:
    """The anti-opacity check.

    Specification section 16 warns against "opaque 'single agent' behavior where
    retrieval, risk assessment, human review, and final answering are hidden
    inside one LLM call". Each stage must leave its own trace with its own
    inspectable output.
    """
    service.submit(high_risk_question, request_id="obs-8")
    service.resume("obs-8", {"decision": "approved", "reviewer_id": "hr@example.com"})
    events = service.history("obs-8")

    ends = {e["node"]: e for e in events if e["event"] == "node.end"}

    # Retrieval produced evidence, and nothing else.
    assert "retrieved" in ends["retrieve_policy"]
    assert "risk_level" not in ends["retrieve_policy"]

    # Classification produced a risk level, and nothing else.
    assert ends["classify_risk"]["risk_level"] == RISK_HIGH
    assert "retrieved" not in ends["classify_risk"]

    # Human review produced a decision.
    assert ends["interrupt_for_review"]["review_decision"] == "approved"

    # Finalisation produced the status and the record.
    assert ends["finalize_decision"]["status"]
    assert ends["finalize_decision"]["decision_record_id"]


def test_failed_requests_are_logged_as_failures(isolated_settings, tmp_path) -> None:
    """A request that cannot be resolved leaves a record saying so."""
    from agent.observability import build_event_log
    from agent.retrieval.retriever import PolicyRetriever
    from shared.embeddings import build_embedder
    from shared.vector_store import InMemoryPolicyVectorStore

    isolated_settings.checkpoint_db = str(tmp_path / "fail.sqlite")
    isolated_settings.observability_log = str(tmp_path / "fail.jsonl")

    broken = PolicyRetriever(
        InMemoryPolicyVectorStore(collection="empty"),
        build_embedder(isolated_settings.embedding_provider),
        isolated_settings.retrieval,
    )
    service = PolicyReviewService(
        isolated_settings,
        retriever=broken,
        log=build_event_log(isolated_settings.observability_path),
    )

    result = service.submit("Can we terminate an employee?", request_id="obs-fail-1")

    assert result.status == "failed"
    assert not result.awaiting_human_review
    # And crucially: no policy rule was invented to fill the gap.
    assert "cannot be resolved" in result.final_answer
    assert "permission" in result.final_answer

    events = {e["event"] for e in service.history("obs-fail-1")}
    assert "decision.failed" in events


def test_pausing_for_review_is_not_logged_as_an_error(
    service: PolicyReviewService, high_risk_question: str
) -> None:
    """A paused request is normal control flow, not an incident.

    `interrupt()` suspends a node by raising, so a naive trace decorator records
    every high-risk pause as `node.error` -- which in a governance audit trail
    reads as a system failure on exactly the requests that matter most.
    """
    service.submit(high_risk_question, request_id="obs-pause-1")
    events = service.history("obs-pause-1")

    assert not [e for e in events if e["event"] == "node.error"], (
        "pausing for human review was logged as an error"
    )
    paused = [e for e in events if e["event"] == "node.paused"]
    assert len(paused) == 1
    assert paused[0]["node"] == "interrupt_for_review"


def test_genuine_node_failures_are_still_logged_as_errors(
    service: PolicyReviewService, high_risk_question: str
) -> None:
    """The pause carve-out must not swallow real faults."""
    import pytest

    from agent.nodes.finalize_decision import HumanReviewBypassError, make_finalize_decision_node
    from agent.observability import build_event_log
    from agent.state import RISK_HIGH

    log = build_event_log(service._settings.observability_path)  # noqa: SLF001
    finalize = make_finalize_decision_node(lambda: log)

    with pytest.raises(HumanReviewBypassError):
        finalize({"request_id": "obs-err-1", "risk_level": RISK_HIGH, "review_decision": ""})

    errors = [e for e in log.read("obs-err-1") if e["event"] == "node.error"]
    assert errors and "HumanReviewBypassError" in errors[0]["error"]