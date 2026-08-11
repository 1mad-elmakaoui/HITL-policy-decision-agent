"""Observability for the runtime workflow.

this requires "an observability infrastructure layer ... across all
components" that "logs all queries, responses, and the inputs and outputs of each
component", and treats traceability -- linking a query to the retrieved
documents, embedding ids, and the reconstructed generation prompt -- as a
first-class quality.

Both are satisfied the same way: every node emits a structured event carrying
the thread id, the node name, and the route-relevant fields it changed. Events
go to an append-only JSONL file that can be read without re-running anything.

What is *not* here: retrieval and reasoning are never collapsed into one opaque
call, so there is always something meaningful to log at each boundary.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langgraph.errors import GraphBubbleUp

_LOCK = threading.Lock()


class EventLog:
    """Append-only JSONL event log."""

    def __init__(self, path: Optional[Path], enabled: bool = True) -> None:
        self._path = Path(path) if path else None
        self._enabled = enabled and self._path is not None

    def emit(self, event: str, thread_id: str, **fields: Any) -> Dict[str, Any]:
        record = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "thread_id": thread_id,
            "pid": os.getpid(),
            **fields,
        }
        if self._enabled and self._path is not None:
            with _LOCK:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, default=str) + "\n")
        return record

    def read(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read events back, optionally for a single thread."""
        if self._path is None or not self._path.exists():
            return []
        events: List[Dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if thread_id is None or record.get("thread_id") == thread_id:
                    events.append(record)
        return events


_NULL_LOG = EventLog(None, enabled=False)


def traced_node(name: str, log_factory: Callable[[], EventLog] = lambda: _NULL_LOG):
    """Wrap a node so its entry, exit and failure are recorded.

    The wrapper also appends the node name to ``route_history``, so the path a
    request actually took survives in the checkpoint as well as in the log --
    the log can be rotated away, the checkpoint cannot.
    """

    def decorator(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        def wrapper(state: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
            log = log_factory()
            thread_id = str(state.get("request_id", ""))
            log.emit("node.start", thread_id, node=name)
            try:
                update = fn(state, *args, **kwargs) or {}
            except GraphBubbleUp:
                # `interrupt()` suspends a node by raising, and LangGraph uses the
                # same mechanism for other control flow. That is not a failure,
                # and recording it as one would make every paused high-risk
                # request look like an incident in the audit trail.
                log.emit("node.paused", thread_id, node=name, reason="awaiting external input")
                raise
            except Exception as exc:  # noqa: BLE001 - re-raised after logging
                log.emit("node.error", thread_id, node=name, error=f"{type(exc).__name__}: {exc}")
                raise

            # `route_history` carries an `operator.add` reducer, so a node
            # returns only its own contribution and LangGraph appends it.
            update["route_history"] = list(update.get("route_history") or []) + [name]

            log.emit("node.end", thread_id, node=name, **_summarize(update))
            return update

        wrapper.__name__ = getattr(fn, "__name__", name)
        wrapper.__doc__ = fn.__doc__
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    return decorator


#: Fields worth recording on every node transition. Kept small on purpose: the
#: log is for reconstructing decisions, not for mirroring the whole state.
_TRACED_FIELDS = (
    "evidence_grade",
    "risk_level",
    "risk_classifier",
    "review_required",
    "review_decision",
    "approved",
    "status",
    "retrieval_error",
)


def _summarize(update: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {k: update[k] for k in _TRACED_FIELDS if k in update}
    if "policy_passages" in update:
        passages = update.get("policy_passages") or []
        # Chunk ids and scores, not chunk text: enough to re-fetch the exact
        # passages that were used, without duplicating the corpus into the log.
        summary["retrieved"] = [
            {
                "chunk_id": p.get("metadata", {}).get("chunk_id", ""),
                "citation": p.get("citation", ""),
                "score": p.get("score", 0.0),
            }
            for p in passages
        ]
    if "risk_signals" in update:
        summary["risk_signals"] = update["risk_signals"]
    if "decision_record" in update:
        summary["decision_record_id"] = (update["decision_record"] or {}).get("record_id", "")
    return summary


def build_event_log(path: Optional[Path], enabled: bool = True) -> EventLog:
    return EventLog(path, enabled=enabled)