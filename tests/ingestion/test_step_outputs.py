"""Each multi-output step must actually declare its outputs to ZenML.

ZenML decides how many artifacts a step produces by calling `inspect.getsource`
on the step function and parsing the AST of its `return` statement. When that
detection goes wrong the only symptom is a StepInterfaceError raised much later
from inside the pipeline, naming neither the cause nor the fix.

The failure message here carries the full diagnosis -- ZenML version, the file
the interpreter resolved, and the source ZenML actually parsed -- so a CI
failure is self-describing and does not need a second run to investigate.
"""

from __future__ import annotations

import inspect
import platform
import sys

import pytest

zenml = pytest.importorskip("zenml", reason="ZenML is not installed")

from ingestion.steps.embed_chunks import embed_chunks_step  # noqa: E402
from ingestion.steps.verify_documents import verify_documents_step  # noqa: E402

MULTI_OUTPUT_STEPS = [
    (verify_documents_step, ("verified_documents", "verification_report")),
    (embed_chunks_step, ("chunk_embeddings", "embedding_report")),
]


def _diagnosis(step) -> str:
    """Everything needed to understand the failure, gathered at failure time."""
    from zenml.steps.utils import has_tuple_return

    func = step.entrypoint
    lines = [
        "",
        "--- diagnosis " + "-" * 56,
        f"python           : {sys.version.split()[0]} ({platform.system()})",
        f"zenml            : {zenml.__version__}",
        f"step name        : {step.name}",
    ]

    try:
        lines.append(f"resolved file    : {inspect.getsourcefile(func)}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"resolved file    : UNAVAILABLE ({type(exc).__name__}: {exc})")

    try:
        lines.append(f"has_tuple_return : {has_tuple_return(func)}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"has_tuple_return : RAISED {type(exc).__name__}: {exc}")

    lines.append("--- source ZenML parsed " + "-" * 46)
    try:
        source = inspect.getsource(func)
        for number, line in enumerate(source.splitlines(), 1):
            lines.append(f"{number:3} | {line}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"inspect.getsource RAISED {type(exc).__name__}: {exc}")
    lines.append("-" * 70)

    return "\n".join(lines)


@pytest.mark.parametrize(
    "step, expected",
    MULTI_OUTPUT_STEPS,
    ids=lambda value: getattr(value, "name", "") or "",
)
def test_multi_output_steps_declare_every_artifact(step, expected) -> None:
    declared = tuple(step.entrypoint_definition.outputs.keys())

    assert declared == expected, (
        f"{step.name} declares {declared} instead of {expected}.\n"
        "ZenML reads the AST of the step's `return` statement, so it must end "
        "in an explicit tuple literal:\n"
        "    a, b = the_function(...)\n"
        "    return a, b\n"
        "If the source below already does that, the detection itself is at "
        "fault rather than the code."
        + _diagnosis(step)
    )
