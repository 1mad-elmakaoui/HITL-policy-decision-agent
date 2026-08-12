"""Ingestion-step tests.

Each step is a plain function, so it is testable without an orchestrator, a
stack, or an artifact store. ``test_zenml_pipeline.py`` covers the orchestrated
form separately.

The emphasis is on the properties the runtime depends on: provenance survives
chunking, verification actually rejects bad corpora, re-ingestion reconciles
rather than duplicating, and the quality gate fails a degraded index.
"""

from __future__ import annotations

import pytest

from config.settings import Settings
from ingestion.chunking.section_chunker import chunk_document
from ingestion.steps.chunk_documents import chunk_documents
from ingestion.steps.embed_chunks import embed_chunks
from ingestion.steps.evaluate_retrieval import RetrievalQualityGate, evaluate_retrieval
from ingestion.steps.index_chunks import index_chunks
from ingestion.steps.load_documents import load_documents
from ingestion.steps.parse_documents import MalformedPolicyDocument, parse_document, parse_documents
from ingestion.steps.verify_documents import PolicyVerificationError, verify_documents
from shared.schema import REQUIRED_METADATA_FIELDS, PolicyChunk, PolicyDocument


@pytest.fixture
def raw_documents(indexed_settings: Settings):
    return load_documents(str(indexed_settings.corpus_path))


@pytest.fixture
def parsed_documents(raw_documents):
    return parse_documents(raw_documents)


# --------------------------------------------------------------------------
# Load and parse
# --------------------------------------------------------------------------
def test_load_reads_the_whole_corpus(raw_documents) -> None:
    assert len(raw_documents) == 4
    for record in raw_documents:
        assert record["raw_text"].strip()
        assert record["source"]


def test_missing_corpus_directory_is_an_error() -> None:
    with pytest.raises(FileNotFoundError):
        load_documents("/nonexistent/policy/corpus")


def test_parse_extracts_identity_and_sections(parsed_documents) -> None:
    documents = [PolicyDocument.from_dict(d) for d in parsed_documents]
    ids = {d.document_id for d in documents}
    assert ids == {"HR-POL-001", "HR-POL-002", "FIN-POL-004", "SEC-POL-002"}

    for document in documents:
        assert document.policy_name
        assert document.document_version
        assert document.sections
        for section in document.sections:
            assert section["heading"]
            assert section["body"].strip()


def test_document_version_is_a_string_not_a_float(parsed_documents) -> None:
    """"4.2" must not arrive as 4.2, or version comparison and chunk ids drift."""
    for document in parsed_documents:
        assert isinstance(document["document_version"], str)


@pytest.mark.parametrize(
    "raw_text, reason",
    [
        ("## Section 1. Thing\n\nBody text here.", "no front matter"),
        ("---\ndocument_id: X\n---\n\n## Section 1. A\n\nBody.", "incomplete front matter"),
        ("---\ndocument_id: X\npolicy_name: Y\n---\n\n## Section 1. A\n\nBody.", "no version"),
        ("---\n: : :\n---\n\n## Section 1. A\n\nBody.", "invalid YAML"),
        ("---\ndocument_id: X\npolicy_name: Y\ndocument_version: '1'\n---\n\n   \n", "no content"),
    ],
)
def test_malformed_documents_are_rejected(raw_text: str, reason: str) -> None:
    """Specification section 15: a malformed policy document is a controlled failure.

    An un-attributable passage is worse than no passage, because the runtime
    cannot cite it. Every case here would produce content that could not be
    traced to an identified, versioned policy.
    """
    with pytest.raises(MalformedPolicyDocument):
        parse_document({"raw_text": raw_text, "source": "bad.md"})


def test_a_document_without_headings_is_kept_as_a_preamble() -> None:
    """Not every policy is numbered, and unsectioned text is still citable.

    The bar for rejection is attributability, not formatting: this document can
    still be cited as "Y v1, Preamble", so it is indexed rather than dropped.
    """
    document = parse_document(
        {
            "raw_text": "---\ndocument_id: X\npolicy_name: Y\ndocument_version: '1'\n---\n\n"
            "Staff must not share accounts.",
            "source": "flat.md",
        }
    )
    assert [s["heading"] for s in document.sections] == ["Preamble"]
    assert "share accounts" in document.sections[0]["body"]


# --------------------------------------------------------------------------
# Verification (RAGOps 4.2.2)
# --------------------------------------------------------------------------
def test_shipped_corpus_passes_verification(parsed_documents) -> None:
    verified, report = verify_documents(parsed_documents)
    assert report["passed"], report["errors"]
    assert len(verified) == 4


def test_recency_check_keeps_only_the_newest_version(parsed_documents) -> None:
    """RAGOps: "Older versions should not be retained, but archived and replaced".

    Two versions of one policy in the index is the dangerous case: retrieval
    would surface a superseded clause with full provenance, looking entirely
    legitimate.
    """
    newest = dict(parsed_documents[0])
    stale = dict(parsed_documents[0])
    stale["document_version"] = "0.9"

    verified, report = verify_documents([stale, newest])

    assert len(verified) == 1
    assert verified[0]["document_version"] == newest["document_version"]
    assert any("0.9" in note for note in report["recency"])


def test_uniqueness_check_rejects_a_duplicated_document(parsed_documents) -> None:
    duplicate = dict(parsed_documents[1])
    duplicate["document_id"] = "HR-POL-999"

    with pytest.raises(PolicyVerificationError):
        verify_documents([parsed_documents[1], duplicate])


def test_completeness_check_rejects_missing_provenance(parsed_documents) -> None:
    incomplete = dict(parsed_documents[0])
    incomplete["source"] = ""

    with pytest.raises(PolicyVerificationError):
        verify_documents([incomplete])


def test_verification_can_report_without_failing(parsed_documents) -> None:
    incomplete = dict(parsed_documents[0])
    incomplete["source"] = ""
    _, report = verify_documents([incomplete], strict=False)
    assert not report["passed"]
    assert report["errors"]


def test_verification_report_is_a_first_class_artifact(parsed_documents) -> None:
    """An operator must be able to ask later why a version is or is not indexed."""
    _, report = verify_documents(parsed_documents)
    assert set(report) >= {
        "documents_in",
        "documents_out",
        "quality",
        "completeness",
        "recency",
        "consistency",
        "uniqueness",
        "errors",
        "passed",
    }


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def test_chunking_preserves_section_boundaries(parsed_documents) -> None:
    """RAGOps 7.1.1: the retrieval unit is the natural semantic unit."""
    document = PolicyDocument.from_dict(
        next(d for d in parsed_documents if d["document_id"] == "HR-POL-001")
    )
    chunks = chunk_document(document, max_chunk_chars=100_000)
    assert len(chunks) == len(document.sections)

    headings = [c.metadata.section for c in chunks]
    assert "Section 6. Summary Dismissal for Gross Misconduct" in headings


def test_every_chunk_carries_complete_provenance(parsed_documents) -> None:
    chunks = [PolicyChunk.from_dict(c) for c in chunk_documents(parsed_documents)]
    assert chunks
    for chunk in chunks:
        assert not chunk.metadata.missing_fields()
        for field in REQUIRED_METADATA_FIELDS:
            assert getattr(chunk.metadata, field)


def test_chunk_ids_are_unique_and_deterministic(parsed_documents) -> None:
    """Re-running ingestion must be idempotent, not corpus-duplicating."""
    first = chunk_documents(parsed_documents)
    second = chunk_documents(parsed_documents)

    ids = [c["metadata"]["chunk_id"] for c in first]
    assert len(ids) == len(set(ids))
    assert ids == [c["metadata"]["chunk_id"] for c in second]


def test_oversized_sections_are_split_not_truncated(parsed_documents) -> None:
    """RAGOps 4.2.2 completeness: "Texts should not be truncated"."""
    document = PolicyDocument.from_dict(parsed_documents[0])
    full_length = sum(len(s["body"]) for s in document.sections)

    chunks = chunk_document(document, max_chunk_chars=300, overlap_chars=40)
    assert len(chunks) > len(document.sections)
    # Overlap means the total grows; it must never shrink.
    assert sum(len(c.text) for c in chunks) >= full_length


def test_chunk_text_includes_the_section_heading(parsed_documents) -> None:
    """So that a question phrased in a heading's words finds that section."""
    chunks = [PolicyChunk.from_dict(c) for c in chunk_documents(parsed_documents)]
    sample = next(c for c in chunks if "Summary Dismissal" in c.metadata.section)
    assert sample.metadata.section.split(". ", 1)[-1].lower() in sample.text.lower()


# --------------------------------------------------------------------------
# Embedding and indexing
# --------------------------------------------------------------------------
def test_embedding_produces_one_fitted_vector_per_chunk(parsed_documents) -> None:
    chunks = chunk_documents(parsed_documents)
    vectors, report = embed_chunks(chunks)

    assert len(vectors) == len(chunks)
    assert report["zero_vectors"] == 0
    assert report["fitted"] is True
    assert report["embedder_state"]["n_documents"] == len(chunks)


def test_reindexing_reconciles_instead_of_duplicating(parsed_documents, tmp_path) -> None:
    """RAGOps 4.2.4: outdated information must be removed from retrieval."""
    chunks = chunk_documents(parsed_documents)
    vectors, report = embed_chunks(chunks)

    index_dir = str(tmp_path / "index")
    first = index_chunks(chunks, vectors, backend="memory", persist_directory=index_dir,
                         embedder_state=report)
    second = index_chunks(chunks, vectors, backend="memory", persist_directory=index_dir,
                          embedder_state=report)

    assert first["chunks_written"] == second["chunks_written"] == len(chunks)


def test_indexing_publishes_the_fitted_embedder_state(parsed_documents, tmp_path) -> None:
    """The index and the embedder that produced it are one artifact."""
    from shared.embeddings import load_embedder_state

    chunks = chunk_documents(parsed_documents)
    vectors, report = embed_chunks(chunks)
    index_dir = tmp_path / "index"

    result = index_chunks(chunks, vectors, backend="memory", persist_directory=str(index_dir),
                          embedder_state=report)

    assert result["embedder_fitted"] is True
    state = load_embedder_state(index_dir)
    assert state is not None and state.fitted


# --------------------------------------------------------------------------
# The retrieval quality gate
# --------------------------------------------------------------------------
def test_quality_gate_fails_a_degraded_index(parsed_documents, indexed_settings: Settings) -> None:
    """An index that cannot retrieve the right policy must not be published."""
    chunks = chunk_documents(parsed_documents)
    vectors, report = embed_chunks(chunks)

    with pytest.raises(RetrievalQualityGate):
        evaluate_retrieval(
            chunks=chunks,
            vectors=vectors,
            eval_set_path=str(indexed_settings.eval_set_full_path),
            embedder_state=report,
            min_recall_at_k=0.999,  # unreachable
            min_mrr=0.999,
            min_ndcg_at_k=0.999,
        )


def test_quality_gate_can_report_without_failing(
    parsed_documents, indexed_settings: Settings
) -> None:
    chunks = chunk_documents(parsed_documents)
    vectors, report = embed_chunks(chunks)

    evaluation = evaluate_retrieval(
        chunks=chunks,
        vectors=vectors,
        eval_set_path=str(indexed_settings.eval_set_full_path),
        embedder_state=report,
        min_recall_at_k=0.999,
        fail_below_threshold=False,
    )
    assert not evaluation["passed"]
    assert evaluation["failures"]


def test_stale_eval_labels_are_detected(parsed_documents, tmp_path, indexed_settings) -> None:
    """An eval set that has drifted from the corpus must not silently pass."""
    stale = tmp_path / "stale_eval.yaml"
    stale.write_text(
        "k: 5\nqueries:\n"
        "  - id: q1\n"
        "    question: anything\n"
        "    relevant:\n"
        "      - document_id: HR-POL-404\n"
        "        section_contains: Nonexistent\n",
        encoding="utf-8",
    )
    chunks = chunk_documents(parsed_documents)
    vectors, report = embed_chunks(chunks)

    with pytest.raises(ValueError, match="stale"):
        evaluate_retrieval(
            chunks=chunks, vectors=vectors, eval_set_path=str(stale), embedder_state=report
        )


def test_missing_eval_set_is_an_error(parsed_documents) -> None:
    chunks = chunk_documents(parsed_documents)
    vectors, report = embed_chunks(chunks)
    with pytest.raises(FileNotFoundError):
        evaluate_retrieval(
            chunks=chunks, vectors=vectors, eval_set_path="/no/such/eval.yaml", embedder_state=report
        )


# --------------------------------------------------------------------------
# Layer separation
# --------------------------------------------------------------------------
def test_the_runtime_never_writes_to_the_vector_store() -> None:
    """Specification section 13: the layers must not absorb each other's work.

    Only the ingestion layer may call the store's write methods. If the runtime
    ever did, the vector store would stop being a reproducible ZenML artifact.
    """
    import ast

    from config.settings import PROJECT_ROOT

    write_methods = {"upsert", "delete_document"}
    offenders = []

    for path in (PROJECT_ROOT / "agent").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) in write_methods:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert not offenders, "runtime code writes to the vector store: " + ", ".join(offenders)


def test_the_runtime_never_parses_policy_documents() -> None:
    """The runtime reads the index; it does not read the corpus."""
    import ast

    from config.settings import PROJECT_ROOT

    offenders = []
    for path in (PROJECT_ROOT / "agent").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ingestion"):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} -> {node.module}")

    assert not offenders, "runtime imports from the ingestion layer: " + ", ".join(offenders)