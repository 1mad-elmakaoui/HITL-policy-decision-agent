"""Checkpoint persistence and recovery tests.



    graph interrupted -> process/session ends -> graph is recreated
    -> same thread_id -> resume -> execution continues correctly

and, crucially, that "recovery does not restart the entire workflow".

Three levels of evidence, weakest to strongest:

1. Rebuilding the graph object in the same process and resuming.
2. Counting node executions to prove earlier nodes did not re-run.
3. Resuming from a genuinely separate OS process that never saw the submit --
   the only test that actually exercises what the LangGraph paper means by
   "thread_id state survives restarts and delayed review".

``MemorySaver`` appears nowhere. Every graph here compiles against the real
SQLite checkpointer.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent.graph import NODE_CLASSIFY, NODE_DRAFT, NODE_INTERRUPT, NODE_RETRIEVE
from agent.observability import build_event_log
from agent.retrieval.retriever import PolicyRetriever
from agent.runtime import PolicyReviewService
from agent.state import STATUS_COMPLETED
from config.settings import PROJECT_ROOT, Settings


# --------------------------------------------------------------------------
# The checkpoint is durable
# --------------------------------------------------------------------------
def test_interrupt_writes_a_durable_checkpoint_file(
    isolated_settings: Settings, high_risk_question: str
) -> None:
    service = PolicyReviewService(isolated_settings)
    service.submit(high_risk_question, request_id="persist-1")

    checkpoint_file = isolated_settings.checkpoint_path
    assert checkpoint_file.exists(), "no checkpoint database was written"
    assert checkpoint_file.stat().st_size > 0


def test_state_survives_discarding_every_object(
    isolated_settings: Settings, high_risk_question: str
) -> None:
    """Drop the service, the graph and the checkpointer; the state remains."""
    service = PolicyReviewService(isolated_settings)
    submitted = service.submit(high_risk_question, request_id="persist-2")
    assert submitted.awaiting_human_review
    del service

    # A brand-new service, sharing nothing but the settings.
    recovered = PolicyReviewService(isolated_settings)
    payload = recovered.pending_review("persist-2")

    assert payload is not None
    assert payload["question"] == high_risk_question
    assert payload["risk_level"] == "high"
    assert payload["preliminary_assessment"]["draft_answer"]


def test_resume_after_rebuilding_the_graph(
    isolated_settings: Settings, high_risk_question: str
) -> None:
    PolicyReviewService(isolated_settings).submit(high_risk_question, request_id="persist-3")

    result = PolicyReviewService(isolated_settings).resume(
        "persist-3", {"decision": "approved", "reviewer_id": "hr@example.com"}
    )

    assert result.status == STATUS_COMPLETED
    assert result.final_answer


# --------------------------------------------------------------------------
# Recovery resumes; it does not restart
# --------------------------------------------------------------------------
class _CountingRetriever(PolicyRetriever):
    """A retriever that records how many times it was asked to retrieve."""

    def __init__(self, inner: PolicyRetriever) -> None:
        self._inner = inner
        self.calls = 0

    def retrieve(self, question: str, k=None):  # type: ignore[override]
        self.calls += 1
        return self._inner.retrieve(question, k)

    def index_stats(self):  # type: ignore[override]
        return self._inner.index_stats()


def test_resume_does_not_re_run_completed_nodes(
    isolated_settings: Settings, high_risk_question: str
) -> None:
    """The point of a checkpoint: work already done is not done again.

    Retrieval is the observable proxy -- it is the first node and the only one
    with an injectable collaborator, so a second call would mean the graph
    restarted from START rather than resuming at the interrupt.
    """
    counting = _CountingRetriever(PolicyRetriever.from_settings(isolated_settings))
    log = build_event_log(isolated_settings.observability_path)

    submit_service = PolicyReviewService(isolated_settings, retriever=counting, log=log)
    submit_service.submit(high_risk_question, request_id="persist-4")
    assert counting.calls == 1

    resume_service = PolicyReviewService(isolated_settings, retriever=counting, log=log)
    result = resume_service.resume(
        "persist-4", {"decision": "approved", "reviewer_id": "hr@example.com"}
    )

    assert result.status == STATUS_COMPLETED
    assert counting.calls == 1, (
        f"retrieval ran {counting.calls} times; the workflow restarted instead of resuming"
    )


def test_route_history_shows_one_pass_through_the_early_nodes(
    isolated_settings: Settings, high_risk_question: str
) -> None:
    """The route record itself proves the early nodes ran exactly once."""
    PolicyReviewService(isolated_settings).submit(high_risk_question, request_id="persist-5")
    result = PolicyReviewService(isolated_settings).resume(
        "persist-5", {"decision": "approved", "reviewer_id": "hr@example.com"}
    )

    for node in (NODE_RETRIEVE, NODE_DRAFT, NODE_CLASSIFY):
        assert result.route_history.count(node) == 1, (
            f"{node} appears {result.route_history.count(node)} times in {result.route_history}"
        )


def test_work_done_before_the_interrupt_is_preserved_verbatim(
    isolated_settings: Settings, high_risk_question: str
) -> None:
    """The LangGraph paper's contract test: a resumed thread preserves
    ``draft_answer`` while updating ``reviewer_feedback``."""
    paused = PolicyReviewService(isolated_settings).submit(
        high_risk_question, request_id="persist-6"
    )
    draft_before = paused.interrupt_payload["preliminary_assessment"]["draft_answer"]
    passages_before = paused.interrupt_payload["policy_passages"]

    resumed_service = PolicyReviewService(isolated_settings)
    resumed_service.resume(
        "persist-6",
        {"decision": "approved", "reviewer_id": "hr@example.com", "feedback": "Fine as drafted."},
    )
    state = resumed_service.get_state("persist-6")

    assert draft_before in state["draft_answer"]
    assert state["reviewer_feedback"] == "Fine as drafted."
    assert len(state["policy_passages"]) == len(passages_before)
    assert [p["citation"] for p in state["policy_passages"]] == [
        p["citation"] for p in passages_before
    ]


def test_checkpoints_accumulate_across_the_lifecycle(
    isolated_settings: Settings, high_risk_question: str
) -> None:
    service = PolicyReviewService(isolated_settings)
    service.submit(high_risk_question, request_id="persist-7")
    after_submit = service.checkpoint_count("persist-7")
    assert after_submit > 1, "the workflow did not checkpoint between nodes"

    service.resume("persist-7", {"decision": "approved", "reviewer_id": "hr@example.com"})
    assert service.checkpoint_count("persist-7") > after_submit


# --------------------------------------------------------------------------
# thread_id stability
# --------------------------------------------------------------------------
def test_threads_are_isolated_from_one_another(
    isolated_settings: Settings, high_risk_question: str, low_risk_question: str
) -> None:
    service = PolicyReviewService(isolated_settings)
    service.submit(high_risk_question, request_id="thread-a")
    service.submit(low_risk_question, request_id="thread-b")

    assert service.pending_review("thread-a") is not None
    assert service.pending_review("thread-b") is None

    service.resume("thread-a", {"decision": "approved", "reviewer_id": "hr@example.com"})
    assert service.get_state("thread-a")["status"] == STATUS_COMPLETED


def test_submitted_request_id_becomes_the_thread_id(
    isolated_settings: Settings, high_risk_question: str
) -> None:
    result = PolicyReviewService(isolated_settings).submit(
        high_risk_question, request_id="stable-id-1"
    )
    assert result.thread_id == "stable-id-1"
    assert PolicyReviewService(isolated_settings).get_state("stable-id-1")["request_id"] == (
        "stable-id-1"
    )


def test_a_missing_thread_id_is_refused() -> None:
    from agent.checkpointing import thread_config

    with pytest.raises(ValueError):
        thread_config("")


# --------------------------------------------------------------------------
# The real thing: recovery across an OS process boundary
# --------------------------------------------------------------------------
_RESUME_SCRIPT = """
import json, sys
sys.path.insert(0, {project_root!r})

from agent.runtime import PolicyReviewService
from config.settings import load_settings

settings = load_settings()
settings.vector_store_dir = {vector_store!r}
settings.checkpoint_db = {checkpoint_db!r}
settings.observability_log = {event_log!r}

service = PolicyReviewService(settings)

# This process has never seen the submit. Everything it knows comes from the
# checkpoint, addressed by thread_id alone.
pending = service.pending_review({thread_id!r})
result = service.resume(
    {thread_id!r},
    {{"decision": "approved", "reviewer_id": "reviewer-in-another-process"}},
)

print(json.dumps({{
    "saw_pending_review": pending is not None,
    "pending_question": (pending or {{}}).get("question", ""),
    "status": result.status,
    "final_answer": result.final_answer,
    "route_history": result.route_history,
    "decision_record": result.decision_record,
}}))
"""


@pytest.mark.subprocess
def test_a_separate_process_resumes_the_interrupted_request(
    isolated_settings: Settings, high_risk_question: str, tmp_path: Path
) -> None:
    """The strongest form of the persistence requirement.

    The submitting interpreter exits. A new one starts, holding no objects, no
    graph and no memory of the request -- only the thread id and the path to the
    checkpoint database. It must find the parked request, resume it, and finish
    without re-running the earlier nodes.
    """
    submitted = PolicyReviewService(isolated_settings).submit(
        high_risk_question, request_id="cross-process-1", requested_by="manager@example.com"
    )
    assert submitted.awaiting_human_review

    script = tmp_path / "resume_in_new_process.py"
    script.write_text(
        _RESUME_SCRIPT.format(
            project_root=str(PROJECT_ROOT),
            vector_store=isolated_settings.vector_store_dir,
            checkpoint_db=isolated_settings.checkpoint_db,
            event_log=isolated_settings.observability_log,
            thread_id="cross-process-1",
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(PROJECT_ROOT),
        env={"PATH": "/usr/bin:/bin", "POLICY_REVIEW_RUN_MODE": "mock", "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, f"resume process failed:\n{completed.stderr}"

    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    # The separate process found the parked request from the checkpoint alone.
    assert payload["saw_pending_review"] is True
    assert payload["pending_question"] == high_risk_question

    # ...resumed it to completion...
    assert payload["status"] == STATUS_COMPLETED
    assert payload["final_answer"]
    assert "reviewer-in-another-process" in payload["final_answer"]

    # ...continuing from the interrupt rather than restarting: each pre-interrupt
    # node appears exactly once across the whole run.
    for node in (NODE_RETRIEVE, NODE_DRAFT, NODE_CLASSIFY, NODE_INTERRUPT):
        assert payload["route_history"].count(node) == 1, payload["route_history"]

    # ...and the evidence retrieved by the *first* process is in the record
    # written by the second.
    assert payload["decision_record"]["evidence"]["passages"]
    assert payload["decision_record"]["human_review"]["approved"] is True


@pytest.mark.subprocess
def test_the_original_process_sees_the_completion(
    isolated_settings: Settings, high_risk_question: str, tmp_path: Path
) -> None:
    """State written by another process is visible here -- one shared thread."""
    service = PolicyReviewService(isolated_settings)
    service.submit(high_risk_question, request_id="cross-process-2")

    script = tmp_path / "resume2.py"
    script.write_text(
        _RESUME_SCRIPT.format(
            project_root=str(PROJECT_ROOT),
            vector_store=isolated_settings.vector_store_dir,
            checkpoint_db=isolated_settings.checkpoint_db,
            event_log=isolated_settings.observability_log,
            thread_id="cross-process-2",
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(PROJECT_ROOT),
        env={"PATH": "/usr/bin:/bin", "POLICY_REVIEW_RUN_MODE": "mock", "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stderr

    state = service.get_state("cross-process-2")
    assert state["status"] == STATUS_COMPLETED
    assert state["reviewer_id"] == "reviewer-in-another-process"
    assert service.pending_review("cross-process-2") is None


# --------------------------------------------------------------------------
# The checkpointer is durable by construction
# --------------------------------------------------------------------------
def test_no_in_memory_checkpointer_anywhere_in_the_source() -> None:
    """A guard against the regression the LangGraph paper warns about.

    "Removing the checkpointer breaks durable pause/resume: a high-risk case
    cannot complete after interruption because there is no persisted thread
    state." An in-memory saver introduced for convenience -- in a test, a
    fixture, or a script -- would silently reintroduce that, so the whole tree is
    checked rather than only the application code.
    """
    banned = {"MemorySaver", "InMemorySaver"}
    offenders = []

    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in {".venv", "__pycache__", ".git", "build", "dist"} for part in path.parts):
            continue
        if path == Path(__file__):  # this file names them in order to ban them
            continue

        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:  # pragma: no cover - not our source
            continue

        # AST rather than text matching, so that prose explaining why these are
        # not used does not register as a use of them.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name in banned:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} calls {name}()")
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in banned:
                        offenders.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} imports {alias.name}"
                        )

    assert not offenders, "in-memory checkpointer found:\n" + "\n".join(offenders)


def test_compiling_the_graph_requires_a_checkpointer() -> None:
    """The API offers no way to compile without persistence."""
    import inspect

    from agent.graph import compile_policy_review_graph

    signature = inspect.signature(compile_policy_review_graph)
    checkpointer = signature.parameters["checkpointer"]
    assert checkpointer.default is inspect.Parameter.empty, (
        "checkpointer must be a required argument, so a graph cannot be compiled "
        "without durable persistence"
    )