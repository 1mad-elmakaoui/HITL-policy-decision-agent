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
from fastapi.responses import FileResponse, HTMLResponse
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
def index():
    """Serve the page, or explain clearly why it cannot be served.

    An empty index.html returns HTTP 200 with a zero-byte body, which a browser
    renders as a blank white page with no error anywhere - the least debuggable
    failure this app can have. Checked here so the page says what is wrong.
    """
    page = STATIC / "index.html"

    if not page.exists():
        return _setup_error(f"{page} does not exist.")
    if page.stat().st_size == 0:
        return _setup_error(f"{page} is empty (0 bytes) - the file was created but never filled in.")

    return FileResponse(page)


def _setup_error(problem: str) -> HTMLResponse:
    return HTMLResponse(
        status_code=500,
        content=f"""<!doctype html><meta charset="utf-8">
<title>Setup problem</title>
<body style="background:#0b0f1a;color:#e8edf7;font:15px/1.6 system-ui;padding:48px;max-width:720px;margin:auto">
<h1 style="color:#ff5f6d;font-size:20px">The interface cannot start</h1>
<p style="color:#8b98b0">{problem}</p>
<p style="color:#8b98b0">Copy the full contents of <code>webapp/static/index.html</code> into that
file, save it, and reload this page. The server does not need restarting - the
file is read on every request.</p>
<p style="color:#5d6b85;font-size:13px">The API itself is running:
<a style="color:#5b8cff" href="/api/status">/api/status</a> should return JSON.</p>
</body>""",
    )


@app.get("/api/status")
def status() -> Dict[str, Any]:
    """Run mode and index health, shown in the header."""
    settings = load_settings()
    key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))

    payload: Dict[str, Any] = {
        "run_mode": settings.run_mode.value,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model if settings.llm_provider != "mock" else "deterministic mock",
        "api_key_present": key_present,
        # An API key sitting unused because the run mode was never switched is
        # invisible otherwise: the app works, it just is not using the model.
        "hint": (
            "ANTHROPIC_API_KEY is set but the run mode is mock, so the key is unused. "
            "Restart with: python -m webapp --live"
            if key_present and settings.run_mode.is_mock else ""
        ),
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
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Policy Review Assistant web interface")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use the real model (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument("--host", default=os.environ.get("POLICY_REVIEW_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POLICY_REVIEW_PORT", "8000")))
    args = parser.parse_args()

    if args.live:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            parser.error(
                "--live needs ANTHROPIC_API_KEY.\n"
                "  PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
                "  cmd:         set ANTHROPIC_API_KEY=sk-ant-...\n"
                "  bash:        export ANTHROPIC_API_KEY=sk-ant-..."
            )
        # Settings are read per request, so setting this before the server
        # starts is enough - every request sees live mode.
        os.environ["POLICY_REVIEW_RUN_MODE"] = "live"

    settings = load_settings()
    mode = "LIVE" if not settings.run_mode.is_mock else "MOCK"
    model = settings.llm_model if not settings.run_mode.is_mock else "deterministic mock"

    print(f"\n  Policy Review Assistant  ->  http://{args.host}:{args.port}")
    print(f"  mode: {mode}  ({model})")
    if settings.run_mode.is_mock and os.environ.get("ANTHROPIC_API_KEY"):
        print("  note: ANTHROPIC_API_KEY is set but unused - restart with --live to use it")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()