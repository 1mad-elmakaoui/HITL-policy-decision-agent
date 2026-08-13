"""Each multi-output step must actually declare its outputs to ZenML.

ZenML decides how many artifacts a step produces by parsing the AST of its
`return` statement: a tuple literal means several named artifacts, anything
else means one. So `return verify_documents(...)` silently collapses two
outputs into one, and the only symptom is a StepInterfaceError raised much
later, from inside the pipeline, naming neither the cause nor the fix.

These tests read the step's registered output signature directly. When the
return statement regresses, the failure is immediate and says which step and
what to do about it.
"""

from __future__ import annotations

import pytest

zenml = pytest.importorskip("zenml", reason="ZenML is not installed")

from ingestion.steps.embed_chunks import embed_chunks_step  # noqa: E402
from ingestion.steps.verify_documents import verify_documents_step  # noqa: E402

MULTI_OUTPUT_STEPS = [
    (verify_documents_step, ("verified_documents", "verification_report")),
    (embed_chunks_step, ("chunk_embeddings", "embedding_report")),
]


@pytest.mark.parametrize(
    "step, expected",
    MULTI_OUTPUT_STEPS,
    ids=lambda value: getattr(value, "name", "") or "",
)
def test_multi_output_steps_declare_every_artifact(step, expected) -> None:
    declared = tuple(step.entrypoint_definition.outputs.keys())

    assert declared == expected, (
        f"{step.name} declares {declared} instead of {expected}.\n"
        "ZenML reads the AST of the step's `return` statement. Return an "
        "explicit tuple literal:\n"
        "    a, b = the_function(...)\n"
        "    return a, b\n"
        "rather than `return the_function(...)`."
    )


@pytest.mark.parametrize("step, _expected", MULTI_OUTPUT_STEPS, ids=lambda v: getattr(v, "name", "") or "")
def test_multi_output_steps_return_a_tuple_literal(step, _expected) -> None:
    """The underlying condition, checked directly rather than by its effect."""
    from zenml.steps.utils import has_tuple_return

    assert has_tuple_return(step.entrypoint), (
        f"{step.name} does not end in a tuple literal, so ZenML will publish "
        "its outputs as a single anonymous artifact."
    )
