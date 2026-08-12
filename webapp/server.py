"""Web interface for the Policy Review Assistant.

A thin HTTP layer over PolicyReviewService. It adds no logic of its own: the
routing, the risk gate, the interrupt and the checkpointing all happen exactly
as they do from the CLI, so what the demo shows is the real workflow.

Run it:

    python -m webapp                 # http://127.0.0.1:8000

Live mode (real model behind the drafting step):

    set ANTHROPIC_API_KEY=sk-ant-...
    set POLICY_REVIEW_RUN_MODE=live
    python -m webapp
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.human_review.decision import InvalidReviewInput
from agent.runtime import NotAwaitingReview, PolicyReviewService, UnknownThread
from config.settings import load_settings

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Policy Review Assistant", docs_url="/api/docs")


def service() -> PolicyReviewService:
    """A fresh service per request.

    Deliberate: it mirrors how the workflow is meant to be used. Each call opens
    its own checkpointer, so resuming holds nothing in memory from the submit -
    which is what makes a paused request survive a restart.
    """
    return PolicyReviewService(load_settings())


class SubmitRequest(BaseModel):
    question: str
    requested_by: str = ""
    force_review: bool = False


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str
    reviewer_id: str
    feedback: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status() -> Dict[str, Any]:
    """Run mode and index health, shown in the header."""
    settings = load_settings()
    payload: Dict[str, Any] = {
        "run_mode": settings.run_mode.value,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model if settings.llm_provider != "mock" else "deterministic mock",
        "index": None,
        "index_error": None,
    }
    try:
        from agent.retrieval.retriever import PolicyRetriever

        payload["index"] = PolicyRetriever.from_settings(settings).index_stats()
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI, not fatal
        payload["index_error"] = str(exc)
    return payload


@app.get("/api/samples")
def samples() -> List[Dict[str, str]]:
    """Questions for the demo, chosen to show both paths."""
    return [
        {"label": "Annual leave carry-over", "risk": "low",
         "question": "How many days of annual leave can be carried over to next year?"},
        {"label": "Sick note", "risk": "low",
         "question": "When does an employee need a medical certificate for sick leave?"},
        {"label": "Expense approval", "risk": "low",
         "question": "Who approves an expense claim of 900 currency units?"},
        {"label": "Immediate termination", "risk": "high",
         "question": "Can we terminate an employee immediately for expense fraud?"},
        {"label": "Demotion", "risk": "high",
         "question": "Can we demote an employee to a lower job level?"},
        {"label": "Discipline after a complaint", "risk": "high",
         "question": "An employee raised a harassment complaint last month. Can we discipline them now?"},
        {"label": "Not in the policy", "risk": "none",
         "question": "Is my iguana allowed at the annual shareholder meeting?"},
    ]


@app.post("/api/submit")
def submit(request: SubmitRequest) -> Dict[str, Any]:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A question is required.")
    result = service().submit(
        request.question.strip(),
        requested_by=request.requested_by.strip() or "web-demo",
        force_human_review=request.force_review,
    )
    return result.to_dict()


@app.post("/api/resume")
def resume(request: ResumeRequest) -> Dict[str, Any]:
    try:
        result = service().resume(
            request.thread_id,
            {
                "decision": request.decision,
                "reviewer_id": request.reviewer_id,
                "feedback": request.feedback,
            },
        )
    except InvalidReviewInput as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnknownThread as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotAwaitingReview as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return result.to_dict()


@app.get("/api/pending/{thread_id}")
def pending(thread_id: str) -> Dict[str, Any]:
    try:
        payload = service().pending_review(thread_id)
    except UnknownThread as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"thread_id": thread_id, "awaiting_human_review": payload is not None, "payload": payload}


@app.get("/api/trace/{thread_id}")
def trace(thread_id: str) -> Dict[str, Any]:
    return {"thread_id": thread_id, "events": service().history(thread_id)}


def main() -> None:
    import uvicorn

    host = os.environ.get("POLICY_REVIEW_HOST", "127.0.0.1")
    port = int(os.environ.get("POLICY_REVIEW_PORT", "8000"))
    print(f"\n  Policy Review Assistant  ->  http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()