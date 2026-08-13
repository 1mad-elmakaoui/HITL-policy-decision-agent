"""Print what ZenML actually sees when it inspects the multi-output steps.

Runs identically on Windows and on a CI runner, so the two outputs can be
compared line for line. Run it locally first to establish the baseline, then
run the same script in CI and diff the two.

    python scripts/diagnose_zenml_step.py
"""

from __future__ import annotations

import inspect
import os
import platform
import sys
from pathlib import Path

# Run from anywhere, with or without the package installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECTED = {
    "verify_documents_step": ["verified_documents", "verification_report"],
    "embed_chunks_step": ["chunk_embeddings", "embedding_report"],
}


def main() -> int:
    print("=" * 70)
    print(f"python   : {sys.version.split()[0]}  ({platform.system()})")
    print(f"cwd      : {os.getcwd()}")
    print(f"commit   : {os.environ.get('GITHUB_SHA', 'local working tree')}")

    try:
        import zenml
        from zenml.steps.utils import has_tuple_return
    except ImportError:
        print("zenml    : NOT INSTALLED -- install with: pip install -e \".[ingestion]\"")
        return 2

    print(f"zenml    : {zenml.__version__}")
    print("=" * 70)

    from ingestion.steps.embed_chunks import embed_chunks_step
    from ingestion.steps.verify_documents import verify_documents_step

    failures = 0
    for step in (verify_documents_step, embed_chunks_step):
        name = step.name
        expected = EXPECTED[name]
        declared = list(step.entrypoint_definition.outputs.keys())
        tuple_return = has_tuple_return(step.entrypoint)

        ok = declared == expected and tuple_return
        failures += 0 if ok else 1

        print(f"\n{name}")
        print(f"  file            : {inspect.getsourcefile(step.entrypoint)}")
        print(f"  declares        : {declared}")
        print(f"  expected        : {expected}")
        print(f"  tuple return    : {tuple_return}")
        print(f"  verdict         : {'OK' if ok else 'BROKEN'}")

        if not ok:
            print("\n  --- source as the interpreter sees it ---")
            for i, line in enumerate(inspect.getsource(step.entrypoint).splitlines(), 1):
                print(f"  {i:3} | {line}")

    print("\n" + "=" * 70)
    if failures:
        print(f"{failures} step(s) BROKEN. The source above is what ZenML parsed.")
        return 1

    print("Both steps declare their artifacts correctly in this environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
