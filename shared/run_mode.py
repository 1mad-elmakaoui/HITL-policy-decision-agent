"""Mock / live run mode.

The LangGraph paper's reference implementation ships exactly this switch:
"The examples can run with mock model responses for repeatable local development
or with live providers for integration testing" (section 3.3), driven by
``LANGGRAPH_STUDY_MODE=mock|live`` (Appendix A). The same idea is adopted here
under ``POLICY_REVIEW_RUN_MODE`` so the whole workflow -- graph, interrupts,
checkpoint recovery and the test suite -- is exercisable without provider
credentials, while the identical graph runs against a real model in live mode.
"""

from __future__ import annotations

import os
from enum import Enum

ENV_VAR = "POLICY_REVIEW_RUN_MODE"


class RunMode(str, Enum):
    MOCK = "mock"
    LIVE = "live"

    @property
    def is_mock(self) -> bool:
        return self is RunMode.MOCK


def current_run_mode(default: RunMode = RunMode.MOCK) -> RunMode:
    raw = os.environ.get(ENV_VAR, "").strip().lower()
    if not raw:
        return default
    try:
        return RunMode(raw)
    except ValueError:
        raise ValueError(f"{ENV_VAR} must be 'mock' or 'live', got {raw!r}") from None
