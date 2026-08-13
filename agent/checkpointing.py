"""Persistent checkpointing.

``InMemorySaver()`` "is
appropriate for local tests; production HITL should use a durable checkpointer
(SQLite, Postgres, or equivalent) so thread_id state survives process restarts
and delayed review", and removing the checkpointer entirely "breaks durable
pause/resume: a high-risk case cannot complete after interruption because there
is no persisted thread state".

This application is embedded in a business process, so SQLite is used and
``MemorySaver`` is not available anywhere in the codebase, including tests --
tests point the SQLite file at a temporary path instead. A test suite that
proved recovery against an in-memory saver would prove nothing about the
property that matters.

The checkpointer is a context manager because the SQLite connection is a real
resource. :func:`checkpointer_scope` is the supported way to obtain one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def checkpointer_scope(db_path: Path | str) -> Iterator[SqliteSaver]:
    """Open a durable SQLite checkpointer at ``db_path``.

    The file is created if absent, so the first invocation of a fresh deployment
    works without a provisioning step.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # `check_same_thread=False` because a graph compiled in one thread may be
    # invoked from another (a request handler, a worker); LangGraph serialises
    # its own writes.
    connection = sqlite3.connect(str(path), check_same_thread=False)
    try:
        # WAL keeps a reader (an operator listing pending reviews) from blocking
        # a writer (a resume landing at the same moment).
        connection.execute("PRAGMA journal_mode=WAL")
        yield SqliteSaver(connection)
    finally:
        connection.close()


def thread_config(thread_id: str) -> dict:
    """The invocation config for a request.

    The ``thread_id`` *is* the request id and never changes over the life of the
    request: submit, interrupt, restart and resume all address the same thread.
    That stability is what makes recovery resume rather than restart.
    """
    if not thread_id:
        raise ValueError("thread_id is required; checkpoint recovery depends on a stable id")
    return {"configurable": {"thread_id": thread_id}}