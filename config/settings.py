"""Single source of configuration for both layers.

Layered resolution: dataclass defaults, then ``config/policy_review.yaml``, then
environment variables (``POLICY_REVIEW_*``). Ingestion and runtime read the same
settings object, which is what keeps the embedding model and collection name in
agreement across the two layers -- a mismatch there would silently produce
meaningless retrieval.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from shared.run_mode import RunMode, current_run_mode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "policy_review.yaml"


@dataclass
class RetrievalSettings:
    """Runtime retrieval and evidence-gating parameters.

    ``strong_evidence_score`` / ``weak_evidence_score`` implement the agentic RAG
    recipe's evidence grade (LangGraph paper section 4.2): evidence quality is a
    routing input held in state, not a sentence buried in a prompt.
    """

    top_k: int = 5
    # Thresholds are on the BM25 scale the default embedder produces, and are
    # calibrated by the ingestion pipeline's evaluation step against the labelled
    # query set (see ingestion/steps/evaluate_retrieval.py). Changing the
    # embedding provider changes the scale, so they must be re-calibrated with it.
    strong_evidence_score: float = 1.00
    weak_evidence_score: float = 0.55
    min_supporting_chunks: int = 2


@dataclass
class EvaluationSettings:
    """Offline retrieval-evaluation gate for the ingestion pipeline.


    """

    k: int = 5
    eval_set_path: str = "ingestion/eval/policy_eval_set.yaml"
    min_recall_at_k: float = 0.80
    min_mrr: float = 0.60
    min_ndcg_at_k: float = 0.60
    fail_pipeline_below_threshold: bool = True


@dataclass
class RiskSettings:
    """Risk-classification parameters.

    ``escalate_only_llm`` encodes the governance invariant chosen for this
    project: the optional model pass may raise a risk level but never lower one,
    so a model can never talk the workflow out of human review.
    """

    use_llm_second_opinion: bool = True
    escalate_only_llm: bool = True
    high_risk_on_classifier_error: bool = True


@dataclass
class Settings:
    run_mode: RunMode = RunMode.MOCK

    # --- knowledge layer -------------------------------------------------
    corpus_dir: str = "policy_corpus"
    vector_store_backend: str = "chroma"
    vector_store_dir: str = ".policy_index"
    vector_store_collection: str = "policy_chunks"
    embedding_provider: str = "hashing"
    embedding_model: str = ""
    embedding_dimension: int = 16384
    max_chunk_chars: int = 1400 
    chunk_overlap_chars: int = 160

    # --- runtime layer ---------------------------------------------------
    checkpoint_db: str = ".policy_state/checkpoints.sqlite"
    observability_log: str = ".policy_state/events.jsonl"
    llm_provider: str = "mock"
    llm_model: str = "claude-sonnet-4-5"
    llm_max_tokens: int = 1200

    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)

    # ---------------------------------------------------------------- paths
    def path(self, value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    @property
    def corpus_path(self) -> Path:
        return self.path(self.corpus_dir)

    @property
    def vector_store_path(self) -> Path:
        return self.path(self.vector_store_dir)

    @property
    def checkpoint_path(self) -> Path:
        return self.path(self.checkpoint_db)

    @property
    def observability_path(self) -> Path:
        return self.path(self.observability_log)

    @property
    def eval_set_full_path(self) -> Path:
        return self.path(self.evaluation.eval_set_path)

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["run_mode"] = self.run_mode.value
        return raw


def _coerce(value: Any, target: Any) -> Any:
    if isinstance(target, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(target, int):
        return int(value)
    if isinstance(target, float):
        return float(value)
    return value


def _apply_mapping(target: Any, mapping: Dict[str, Any]) -> None:
    for key, value in (mapping or {}).items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_mapping(current, value)
        else:
            setattr(target, key, _coerce(value, current))


def _apply_environment(settings: Settings) -> None:
    """Environment overrides, e.g. POLICY_REVIEW_VECTOR_STORE_DIR."""
    for f in fields(settings):
        if f.name == "run_mode" or hasattr(getattr(settings, f.name), "__dataclass_fields__"):
            continue
        env_key = f"POLICY_REVIEW_{f.name.upper()}"
        if env_key in os.environ:
            setattr(settings, f.name, _coerce(os.environ[env_key], getattr(settings, f.name)))

    for group_name in ("retrieval", "evaluation", "risk"):
        group = getattr(settings, group_name)
        for f in fields(group):
            env_key = f"POLICY_REVIEW_{group_name.upper()}_{f.name.upper()}"
            if env_key in os.environ:
                setattr(group, f.name, _coerce(os.environ[env_key], getattr(group, f.name)))


def load_settings(config_file: Optional[Path | str] = None) -> Settings:
    settings = Settings(run_mode=current_run_mode())

    path = Path(config_file) if config_file else DEFAULT_CONFIG_FILE
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw.pop("run_mode", None)  # run mode is environment-driven only
        _apply_mapping(settings, raw)

    _apply_environment(settings)

    # In mock mode the deterministic providers are mandatory, not merely the
    # default: a mock run that silently reached for a live provider would defeat
    # the reproducibility the mode exists to give.
    if settings.run_mode.is_mock:
        settings.llm_provider = "mock"
        settings.embedding_provider = "hashing"
    elif settings.llm_provider == "mock":
        # Live mode with the mock provider still selected is almost always a
        # half-finished switch, and it fails silently: the graph runs, the
        # answers just are not from a model. Prefer the real provider.
        settings.llm_provider = "anthropic"

    return settings