"""ZenML pipeline tests.
 pytest -m zenml
"""

from __future__ import annotations

import pytest

from config.settings import Settings

zenml = pytest.importorskip("zenml", reason="ZenML is an optional extra: pip install -e '.[ingestion]'")


# --------------------------------------------------------------------------
# The steps are real ZenML steps, wired in the documented order
# --------------------------------------------------------------------------
def test_every_step_is_a_zenml_step() -> None:
    from zenml.steps import BaseStep

    from ingestion.steps.chunk_documents import chunk_documents_step
    from ingestion.steps.embed_chunks import embed_chunks_step
    from ingestion.steps.evaluate_retrieval import evaluate_retrieval_step
    from ingestion.steps.index_chunks import index_chunks_step
    from ingestion.steps.load_documents import load_documents_step
    from ingestion.steps.parse_documents import parse_documents_step
    from ingestion.steps.verify_documents import verify_documents_step

    steps = [
        load_documents_step,
        parse_documents_step,
        verify_documents_step,
        chunk_documents_step,
        embed_chunks_step,
        index_chunks_step,
        evaluate_retrieval_step,
    ]
    for step in steps:
        assert step is not None
        assert isinstance(step, BaseStep), f"{step} is not a ZenML step"


def test_pipeline_is_a_zenml_pipeline() -> None:
    from zenml.pipelines.pipeline_definition import Pipeline

    from ingestion.pipelines.policy_ingestion import policy_ingestion_pipeline

    assert isinstance(policy_ingestion_pipeline, Pipeline)
    assert policy_ingestion_pipeline.name == "policy_ingestion"


def test_side_effecting_steps_are_not_cached() -> None:
    """A cache hit on indexing would report success while leaving the index stale.

    Indexing and evaluation both reach outside the ZenML artifact store, so they
    must re-run every time; the pure transformation steps may be cached, which is
    what makes re-ingestion after a single policy edit cheap.
    """
    from ingestion.steps.evaluate_retrieval import evaluate_retrieval_step
    from ingestion.steps.index_chunks import index_chunks_step
    from ingestion.steps.load_documents import load_documents_step
    from ingestion.steps.parse_documents import parse_documents_step

    assert index_chunks_step.configuration.enable_cache is False
    assert evaluate_retrieval_step.configuration.enable_cache is False
    assert load_documents_step.configuration.enable_cache is True
    assert parse_documents_step.configuration.enable_cache is True


def test_ingestion_does_not_import_the_runtime_agent() -> None:
    """Specification section 4: ingestion must be independently executable.

    If ingestion imported the agent, the offline pipeline would carry the online
    workflow's dependencies and the two layers would no longer be separable.
    """
    import ast

    from config.settings import PROJECT_ROOT

    offenders = []
    for path in (PROJECT_ROOT / "ingestion").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("agent"):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("agent"):
                        offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert not offenders, "ingestion imports the runtime agent: " + ", ".join(offenders)


# --------------------------------------------------------------------------
# End-to-end orchestrated run
# --------------------------------------------------------------------------
@pytest.mark.zenml
def test_the_zenml_pipeline_produces_a_queryable_index(
    tmp_path, monkeypatch, indexed_settings: Settings
) -> None:
    """Run the real pipeline through the orchestrator, then query what it built."""
    from agent.retrieval.retriever import PolicyRetriever
    from config.settings import load_settings
    from ingestion.pipelines.policy_ingestion import run_policy_ingestion

    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zenml"))

    settings = load_settings()
    settings.vector_store_dir = str(tmp_path / "index")
    settings.checkpoint_db = str(tmp_path / "checkpoints.sqlite")
    settings.observability_log = str(tmp_path / "events.jsonl")

    run_policy_ingestion(settings)

    result = PolicyRetriever.from_settings(settings).retrieve(
        "How many days of annual leave can be carried over?"
    )
    assert result.ok
    assert result.passages
    assert any(p.chunk.metadata.document_id == "HR-POL-002" for p in result.passages)