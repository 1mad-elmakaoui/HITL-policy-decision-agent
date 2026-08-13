"""Shared fixtures.

Two principles here.

*The index is built once.* A session-scoped fixture runs the real ingestion steps
over the real policy corpus into a temporary directory. Tests therefore exercise
the same chunker, embedder and store the application uses, rather than a
hand-made fixture index that could drift from it.

*Persistence is never faked.* Every fixture that needs a checkpointer points the
real SQLite checkpointer at a temporary file. ``MemorySaver`` appears nowhere,
including here -- proving recovery against an in-memory saver would prove nothing
about the property the tests exist to check.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("POLICY_REVIEW_RUN_MODE", "mock")

from agent.llm.client import MockLLMClient  # noqa: E402
from agent.observability import build_event_log  # noqa: E402
from agent.retrieval.retriever import PolicyRetriever  # noqa: E402
from agent.risk.classifier import RiskClassifier  # noqa: E402
from agent.runtime import PolicyReviewService  # noqa: E402
from config.settings import PROJECT_ROOT, Settings, load_settings  # noqa: E402
from ingestion.local_runner import run_ingestion_locally  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def indexed_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings with a freshly ingested policy index in a temporary directory."""
    root = tmp_path_factory.mktemp("policy-index")

    settings = load_settings()
    settings.vector_store_dir = str(root / "index")
    settings.checkpoint_db = str(root / "checkpoints.sqlite")
    settings.observability_log = str(root / "events.jsonl")

    report = run_ingestion_locally(settings)
    assert report["index"]["chunks_written"] > 0, "ingestion produced no chunks"
    return settings


@pytest.fixture(scope="session")
def ingestion_report(indexed_settings: Settings) -> dict:
    """The report from the session's ingestion run, re-run against the same index."""
    return run_ingestion_locally(indexed_settings)


@pytest.fixture
def retriever(indexed_settings: Settings) -> PolicyRetriever:
    return PolicyRetriever.from_settings(indexed_settings)


@pytest.fixture
def classifier() -> RiskClassifier:
    """Classifier with the shipped configuration, including the model pass."""
    return RiskClassifier(load_settings().risk, llm=MockLLMClient())


@pytest.fixture
def isolated_settings(indexed_settings: Settings, tmp_path: Path) -> Settings:
    """Per-test settings: shared index, but a private checkpoint DB and log.

    Each test gets its own durable checkpointer file so threads cannot collide,
    while still sharing the (read-only) index.
    """
    settings = load_settings()
    settings.vector_store_dir = indexed_settings.vector_store_dir
    settings.checkpoint_db = str(tmp_path / "checkpoints.sqlite")
    settings.observability_log = str(tmp_path / "events.jsonl")
    return settings


@pytest.fixture
def service(isolated_settings: Settings) -> Iterator[PolicyReviewService]:
    yield PolicyReviewService(
        isolated_settings, log=build_event_log(isolated_settings.observability_path)
    )


@pytest.fixture
def low_risk_question() -> str:
    return "How many days of annual leave can be carried over to next year?"


@pytest.fixture
def high_risk_question() -> str:
    return "Can we terminate an employee immediately for expense fraud?"