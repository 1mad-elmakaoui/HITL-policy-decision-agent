"""The policy-review ``StateGraph``.

contains graph structure only -- nodes, edges, routes and compilation.
the graph
should express routing and state transitions; the logic file should contain SQL
validation, retrieval scoring, policy checks, and other ordinary application
behavior". Here the domain logic lives in ``agent/retrieval``, ``agent/risk``,
``agent/human_review`` and ``agent/nodes``.

Shape, following  HITL recipe (section 4.3, Figure 2) with retrieval
prepended:

    START
      -> retrieve_policy
           |-- retrieval failed -------------------> record_failure -> END
           |-- evidence retrieved ----------------> draft_assessment
      -> draft_assessment
      -> classify_risk
           |-- low / medium risk -----------------> finalize_decision -> END
           |-- high risk or forced --------------> interrupt_for_review (pause)
                 resume(reviewer_input) --------> apply_feedback
                                                 -> finalize_decision -> END

The draft precedes risk scoring because the interrupt payload must carry a
preliminary assessment for the reviewer to evaluate.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agent.llm.client import LLMClient, build_llm_client
from agent.nodes.apply_feedback import make_apply_feedback_node
from agent.nodes.classify_risk import make_classify_risk_node
from agent.nodes.draft_assessment import make_draft_assessment_node
from agent.nodes.finalize_decision import make_finalize_decision_node
from agent.nodes.interrupt_for_review import make_interrupt_for_review_node
from agent.nodes.record_failure import make_record_failure_node
from agent.nodes.retrieve_policy import make_retrieve_policy_node
from agent.observability import EventLog, build_event_log
from agent.retrieval.retriever import PolicyRetriever
from agent.risk.classifier import RiskClassifier
from agent.routes import (
    ROUTE_DRAFT,
    ROUTE_FAIL,
    ROUTE_FINALIZE,
    ROUTE_INTERRUPT,
    route_after_retrieval,
    route_after_risk,
)
from agent.state import PolicyReviewState
from config.settings import Settings, load_settings

# Node names, used by the builder and asserted by the graph tests.
NODE_RETRIEVE = "retrieve_policy"
NODE_DRAFT = "draft_assessment"
NODE_CLASSIFY = "classify_risk"
NODE_INTERRUPT = "interrupt_for_review"
NODE_APPLY_FEEDBACK = "apply_feedback"
NODE_FINALIZE = "finalize_decision"
NODE_FAIL = "record_failure"


def build_policy_review_graph(
    retriever: PolicyRetriever,
    classifier: RiskClassifier,
    llm: LLMClient,
    log_factory: Callable[[], EventLog],
) -> StateGraph:
    """Assemble the uncompiled graph.

    Returned uncompiled so that compilation -- and therefore the checkpointer --
    is the caller's decision. The runtime service supplies a durable one.
    """
    builder = StateGraph(PolicyReviewState)

    builder.add_node(NODE_RETRIEVE, make_retrieve_policy_node(retriever, log_factory))
    builder.add_node(NODE_DRAFT, make_draft_assessment_node(llm, log_factory))
    builder.add_node(NODE_CLASSIFY, make_classify_risk_node(classifier, log_factory))
    builder.add_node(NODE_INTERRUPT, make_interrupt_for_review_node(log_factory))
    builder.add_node(NODE_APPLY_FEEDBACK, make_apply_feedback_node(log_factory))
    builder.add_node(NODE_FINALIZE, make_finalize_decision_node(log_factory))
    builder.add_node(NODE_FAIL, make_record_failure_node(log_factory))

    builder.add_edge(START, NODE_RETRIEVE)

    builder.add_conditional_edges(
        NODE_RETRIEVE,
        route_after_retrieval,
        {ROUTE_DRAFT: NODE_DRAFT, ROUTE_FAIL: NODE_FAIL},
    )
    builder.add_edge(NODE_DRAFT, NODE_CLASSIFY)

    # The governance gate. Every high-risk path leads through NODE_INTERRUPT;
    # there is no edge from classification straight to finalisation for a
    # high-risk request.
    builder.add_conditional_edges(
        NODE_CLASSIFY,
        route_after_risk,
        {ROUTE_INTERRUPT: NODE_INTERRUPT, ROUTE_FINALIZE: NODE_FINALIZE},
    )

    builder.add_edge(NODE_INTERRUPT, NODE_APPLY_FEEDBACK)
    builder.add_edge(NODE_APPLY_FEEDBACK, NODE_FINALIZE)
    builder.add_edge(NODE_FINALIZE, END)
    builder.add_edge(NODE_FAIL, END)

    return builder


def compile_policy_review_graph(
    checkpointer: BaseCheckpointSaver,
    settings: Optional[Settings] = None,
    retriever: Optional[PolicyRetriever] = None,
    classifier: Optional[RiskClassifier] = None,
    llm: Optional[LLMClient] = None,
    log: Optional[EventLog] = None,
) -> Any:
    """Build and compile the graph against a checkpointer.

    ``checkpointer`` is required, not optional with an in-memory default.
    Compiling without durable persistence is the failure the LangGraph paper
    warns about, so the API does not offer it.

    The collaborators are injectable for testing; by default they are
    constructed from settings.
    """
    settings = settings or load_settings()
    llm = llm or build_llm_client(settings)
    retriever = retriever or PolicyRetriever.from_settings(settings)
    classifier = classifier or RiskClassifier(settings.risk, llm=llm)
    event_log = log if log is not None else build_event_log(settings.observability_path)

    builder = build_policy_review_graph(retriever, classifier, llm, lambda: event_log)
    return builder.compile(checkpointer=checkpointer)