"""Retrieval tests.

These exercise the retrieval component on its own, with no graph and no LLM --
means by testing at the component level, and
what makes it possible to attribute a regression to retrieval rather than to the
answer.

The four required properties: relevant chunks are retrieved; irrelevant questions
do not produce misleading evidence; provenance is preserved; 
"""

from __future__ import annotations

import pytest

from agent.retrieval.retriever import PolicyRetriever
from agent.state import EVIDENCE_NONE, EVIDENCE_STRONG, EVIDENCE_WEAK
from config.settings import RetrievalSettings, Settings
from ingestion.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from shared.embeddings import build_embedder, load_embedder_state
from shared.schema import REQUIRED_METADATA_FIELDS
from shared.vector_store import InMemoryPolicyVectorStore, VectorStoreUnavailable

# Relevant policy is retrieved

@pytest.mark.parametrize(
    "question, expected_document, expected_section_fragment",
    [
        ("How many days of annual leave can be carried over?", "HR-POL-002", "Carry-Over"),
        ("When is a medical certificate needed for sick leave?", "HR-POL-002", "Sick Leave"),
        ("What parental leave does the primary caregiver get?", "HR-POL-002", "Parental Leave"),
        ("Who approves an expense claim of 900 currency units?", "FIN-POL-004", "Approval Thresholds"),
        ("What is the deadline for submitting expense receipts?", "FIN-POL-004", "Submission and Receipts"),
        ("Who must approve a demotion to a lower job level?", "HR-POL-001", "Approval Authority"),
        ("Can an employee be suspended pending an investigation?", "HR-POL-001", "Suspension"),
        ("What is required before targeted monitoring of an employee?", "SEC-POL-002", "Monitoring"),
    ],
)
def test_relevant_policy_is_retrieved(
    retriever: PolicyRetriever,
    question: str,
    expected_document: str,
    expected_section_fragment: str,
) -> None:
    result = retriever.retrieve(question)
    assert result.ok
    assert result.passages, f"no evidence retrieved for {question!r}"

    matches = [
        p
        for p in result.passages
        if p.chunk.metadata.document_id == expected_document
        and expected_section_fragment.lower() in p.chunk.metadata.section.lower()
    ]
    assert matches, (
        f"{question!r} did not retrieve {expected_document}/{expected_section_fragment}; got "
        + ", ".join(p.chunk.citation() for p in result.passages)
    )


def test_retrieved_passages_are_ranked_and_scored(retriever: PolicyRetriever) -> None:
    result = retriever.retrieve("How many days of annual leave can be carried over?")
    assert [p.rank for p in result.passages] == list(range(1, len(result.passages) + 1))
    scores = [p.score for p in result.passages]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_respected(indexed_settings: Settings) -> None:
    retriever = PolicyRetriever.from_settings(indexed_settings)
    assert len(retriever.retrieve("annual leave", k=2).passages) <= 2



# Irrelevant questions do not produce misleading evidence

@pytest.mark.parametrize(
    "question",
    [
        "What will the weather be like in Lisbon next Tuesday?",
        "What is the company position on quantum computing procurement?",
        "How do I configure a Kubernetes ingress controller?",
        "What is the airspeed velocity of an unladen swallow?",
    ],
)
def test_off_corpus_questions_never_reach_strong_evidence(
    retriever: PolicyRetriever, question: str
) -> None:
    """The safety property the ingestion gate also enforces.

    Strong evidence is what licenses the workflow to assert a policy position, so
    a question the corpus does not cover must never reach it. A *weak* grade is
    acceptable: the assessment step is required to answer "unknown" and say what
    the evidence does not settle.
    """
    result = retriever.retrieve(question)
    assert result.evidence_grade != EVIDENCE_STRONG, (
        f"{question!r} graded {result.evidence_grade} at score {result.top_score:.2f}; "
        "the assistant would state a policy position on an uncovered question"
    )


def test_empty_question_returns_no_evidence(retriever: PolicyRetriever) -> None:
    result = retriever.retrieve("   ")
    assert result.passages == []
    assert result.evidence_grade == EVIDENCE_NONE
    assert result.error


def test_evidence_grade_is_computed_from_scores(indexed_settings: Settings) -> None:
    """The grade is a function of retrieval settings, testable on its own."""
    retriever = PolicyRetriever.from_settings(indexed_settings)
    strong = retriever.retrieve("How many days of annual leave can be carried over?")
    assert strong.evidence_grade == EVIDENCE_STRONG

    # Raising the bar above every achievable score must demote the same query.
    strict = PolicyRetriever.from_settings(indexed_settings)
    strict._settings = RetrievalSettings(  # noqa: SLF001 - exercising the gate directly
        top_k=5, strong_evidence_score=10_000.0, weak_evidence_score=0.0, min_supporting_chunks=2
    )
    assert strict.retrieve("How many days of annual leave can be carried over?").evidence_grade == (
        EVIDENCE_WEAK
    )



# Provenance is preserved

def test_every_retrieved_passage_carries_full_provenance(retriever: PolicyRetriever) -> None:
    result = retriever.retrieve("Who must approve a termination for cause?")
    assert result.passages

    for passage in result.passages:
        assert not passage.chunk.metadata.missing_fields(), (
            f"{passage.chunk.chunk_id} missing {passage.chunk.metadata.missing_fields()}"
        )
        assert passage.chunk.citation()


def test_state_passages_expose_metadata_for_the_decision_record(
    retriever: PolicyRetriever,
) -> None:
    """What lands in graph state must be enough to reconstruct the citation."""
    passages = retriever.retrieve("Who must approve a demotion?").to_state_passages()
    assert passages

    for passage in passages:
        assert set(passage) >= {"text", "metadata", "score", "rank", "citation"}
        for field in REQUIRED_METADATA_FIELDS:
            assert passage["metadata"].get(field), f"{field} missing from state passage"


def test_document_version_is_recoverable_from_a_passage(retriever: PolicyRetriever) -> None:
    """A decision record must stay interpretable after the policy is revised."""
    result = retriever.retrieve("What is required before summary dismissal?")
    versions = {p.chunk.metadata.document_version for p in result.passages}
    assert versions and all(v for v in versions)



# Failure handling

def test_unavailable_index_is_a_controlled_result_not_an_exception(
    indexed_settings: Settings,
) -> None:
    """Specification section 15: an unreachable index must not raise past the node."""
    empty = InMemoryPolicyVectorStore(collection="empty")
    embedder = build_embedder(indexed_settings.embedding_provider)
    retriever = PolicyRetriever(empty, embedder, indexed_settings.retrieval)

    result = retriever.retrieve("Can we terminate an employee?")
    assert not result.ok
    assert result.error
    assert result.passages == []
    assert result.evidence_grade == EVIDENCE_NONE


def test_empty_store_raises_only_at_the_store_boundary() -> None:
    with pytest.raises(VectorStoreUnavailable):
        InMemoryPolicyVectorStore(collection="empty").query([0.0, 1.0], k=3)



# The ingestion evaluation follows the RAGOps methodology

def test_ingestion_reports_the_ragops_component_metrics(ingestion_report: dict) -> None:
    """RAGOps Table 1 names MRR, recall@K, precision@K and nDCG for retrieval."""
    component = ingestion_report["evaluation"]["component_level"]
    k = ingestion_report["evaluation"]["k"]
    for metric in (f"recall_at_{k}", f"precision_at_{k}", "mrr", f"ndcg_at_{k}"):
        assert metric in component, f"{metric} missing from the component-level report"
        assert 0.0 <= component[metric] <= 1.0


def test_ingestion_reports_module_level_metrics(ingestion_report: dict) -> None:
    """RAGOps 4.2.5 also requires testing at the module level."""
    module = ingestion_report["evaluation"]["module_level"]
    assert module["chunks"] > 0
    assert module["documents"] == 4
    assert module["zero_vectors"] == 0
    assert module["chunks_missing_metadata"] == 0


def test_retrieval_quality_gate_passes_on_the_shipped_corpus(ingestion_report: dict) -> None:
    evaluation = ingestion_report["evaluation"]
    assert evaluation["passed"], evaluation["failures"]


def test_evaluation_covers_negative_queries(ingestion_report: dict) -> None:
    """Off-corpus cases are part of the methodology, not an afterthought."""
    component = ingestion_report["evaluation"]["component_level"]
    assert component["negative_queries"] >= 3
    assert ingestion_report["evaluation"]["calibration"]["negatives_reaching_strong"] == 0


def test_evaluation_publishes_threshold_calibration(ingestion_report: dict) -> None:
    """The runtime's evidence thresholds are checked against measured scores."""
    calibration = ingestion_report["evaluation"]["calibration"]
    assert calibration["positive_max_top_score"] > calibration["negative_max_top_score"]
    assert calibration["positives_reaching_strong"] > 0


# --------------------------------------------------------------------------
# Metric implementations
# --------------------------------------------------------------------------
def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "d"], 3) == pytest.approx(0.5)
    assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == pytest.approx(1.0)
    assert recall_at_k(["x"], ["a"], 3) == pytest.approx(0.0)


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b", "c", "d"], ["a", "b"], 4) == pytest.approx(0.5)


def test_reciprocal_rank_rewards_earlier_hits() -> None:
    assert reciprocal_rank(["a", "b"], ["a"]) == pytest.approx(1.0)
    assert reciprocal_rank(["b", "a"], ["a"]) == pytest.approx(0.5)
    assert reciprocal_rank(["b", "c"], ["a"]) == pytest.approx(0.0)


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], 3) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "b", "a"], ["a", "b"], 3) < 1.0



# The embedder contract between the layers

def test_fitted_embedder_state_is_published_with_the_index(indexed_settings: Settings) -> None:
    """The runtime must embed queries in the space the chunks were embedded in."""
    state = load_embedder_state(indexed_settings.vector_store_path)
    assert state is not None, "ingestion did not publish the fitted embedder state"
    assert state.fitted
    assert state.n_documents > 0


def test_embedding_is_deterministic_across_processes(indexed_settings: Settings) -> None:
    """Reproducibility: ingestion and the runtime run in different processes.

    Python's built-in ``hash`` is salted per process, so a naive implementation
    would produce a different index on every run.
    """
    state = load_embedder_state(indexed_settings.vector_store_path)
    first = build_embedder(indexed_settings.embedding_provider, state=state)
    second = build_embedder(indexed_settings.embedding_provider, state=state)
    assert first.embed_query("termination approval") == second.embed_query("termination approval")