"""Pipeline step 1 -- load raw policy documents.

"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


def load_documents(corpus_dir: str) -> list[dict[str, Any]]:
    """Read every supported document under ``corpus_dir``.

    Returns raw records -- text plus file-level provenance -- with no parsing of
    the document's own structure. That happens in the parse step.
    """
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Policy corpus directory not found: {root}")

    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stat = path.stat()
        records.append(
            {
                "source": str(path.relative_to(root.parent)) if root.parent in path.parents else str(path),
                "filename": path.name,
                "raw_text": _read_document(path),
                "size_bytes": stat.st_size,
                "modified_at": _dt.datetime.fromtimestamp(
                    stat.st_mtime, _dt.timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )

    if not records:
        raise ValueError(f"No policy documents found in {root}. Supported: {sorted(SUPPORTED_SUFFIXES)}")
    return records


def _read_document(path: Path) -> str:
    return path.read_text(encoding="utf-8")


try:  # pragma: no cover - the decorated form is exercised by the ZenML test
    from zenml import step

    @step(enable_cache=True)
    def load_documents_step(corpus_dir: str) -> list[dict[str, Any]]:
        """ZenML step wrapper. Caching is enabled: an unchanged corpus does not
        need re-reading, which is what makes a re-run cheap after a policy edit
        touches only one document."""
        return load_documents(corpus_dir)

except ImportError:  # pragma: no cover - zenml is an optional extra
    load_documents_step = None  # type: ignore[assignment]
