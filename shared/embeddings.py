from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Filename of the fitted embedder state, written into the index directory by the
#: ingestion pipeline and read by the runtime retriever.
EMBEDDER_STATE_FILENAME = "embedding_state.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very common words carry no retrieval signal and would otherwise dominate the
# term-frequency vector for short policy questions.
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its
    may must not of on or our shall should than that the their them there these this
    to under upon us was we were what when where which who will with would you your""".split()
)


#: Similarity a provider's vectors are meant to be compared under. The vector
#: store follows the embedder rather than assuming cosine, because a length-aware
#: lexical model must not have its length normalisation divided back out.
SIMILARITY_COSINE = "cosine"
SIMILARITY_INNER_PRODUCT = "ip"


@runtime_checkable
class EmbeddingModel(Protocol):
    """Contract every embedding provider must satisfy."""

    name: str
    dimension: int
    similarity: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _stem(token: str) -> str:
    """Crude suffix stripping so 'termination'/'terminate' share a dimension."""
    for suffix in ("ations", "ation", "ing", "ements", "ement", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


@dataclass
class EmbedderState:
    """Serialisable fitted state, published with the index."""

    provider: str
    dimension: int
    n_documents: int = 0
    average_length: float = 0.0
    document_frequencies: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "dimension": self.dimension,
            "n_documents": self.n_documents,
            "average_length": self.average_length,
            "document_frequencies": self.document_frequencies,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EmbedderState:
        return cls(
            provider=str(raw.get("provider", "hashing")),
            dimension=int(raw.get("dimension", 2048)),
            n_documents=int(raw.get("n_documents", 0)),
            average_length=float(raw.get("average_length", 0.0)),
            document_frequencies={str(k): int(v) for k, v in (raw.get("document_frequencies") or {}).items()},
        )

    @property
    def fitted(self) -> bool:
        return self.n_documents > 0 and bool(self.document_frequencies)


class HashingBM25Embedder:
    """Deterministic hashed BM25 embedder.

    Tokens are stemmed and hashed into a fixed-width vector, with adjacent-token
    bigrams hashed in too, so a phrase like "summary dismissal" carries weight
    beyond its two words.

    Scoring is BM25 rather than TF-IDF cosine, for a reason that showed up
    immediately in the retrieval evaluation on this corpus. Cosine divides every
    document vector by its own L2 norm, which penalises long passages in
    proportion to how much they contain -- and in a policy manual the long
    sections are precisely the ones that carry the operative rules. Under cosine,
    a short tangential section outranked "Section 6. Summary Dismissal" for the
    question "can we terminate an employee immediately for expense fraud?".

    BM25 normalises length explicitly (the ``b`` term) and saturates term
    frequency (the ``k1`` term), so a long section is neither rewarded nor
    punished for its length beyond what its content justifies. The document
    weights carry that normalisation, so the vectors must be compared by inner
    product -- cosine would divide it straight back out. Hence ``similarity``.

    The resulting score has a meaningful absolute scale (a strong match on two
    or three discriminative terms lands well above 1.0, an incidental match well
    below), which is what lets the evidence grade be an absolute threshold rather
    than a guess.
    """

    #: Term-frequency saturation. Standard BM25 default.
    K1 = 1.5
    #: Length normalisation strength. Standard BM25 default.
    B = 0.75

    similarity = SIMILARITY_INNER_PRODUCT

    def __init__(self, dimension: int = 16384, state: EmbedderState | None = None) -> None:
        self.dimension = state.dimension if state else dimension
        self._state = state or EmbedderState(provider="hashing", dimension=self.dimension)
        self.name = f"hashing-bm25-{self.dimension}" + ("" if self._state.fitted else "-unfitted")

    # ------------------------------------------------------------------ fit
    def fit(self, texts: Sequence[str]) -> HashingBM25Embedder:
        """Compute document frequencies and mean document length."""
        frequencies: dict[str, int] = {}
        lengths: list[int] = []
        for text in texts:
            indices = self._indices(text)
            lengths.append(len(indices))
            for index in {str(i) for i in indices}:
                frequencies[index] = frequencies.get(index, 0) + 1

        self._state = EmbedderState(
            provider="hashing",
            dimension=self.dimension,
            n_documents=len(texts),
            average_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
            document_frequencies=frequencies,
        )
        self.name = f"hashing-bm25-{self.dimension}"
        return self

    @property
    def state(self) -> EmbedderState:
        return self._state

    @property
    def fitted(self) -> bool:
        return self._state.fitted

    # -------------------------------------------------------------- encoding
    def _indices(self, text: str) -> list[int]:
        tokens = [_stem(t) for t in _tokenize(text)]
        indices = [hash_to_index(t, self.dimension) for t in tokens]
        indices += [
            hash_to_index(f"{left}_{right}", self.dimension)
            for left, right in zip(tokens, tokens[1:])
        ]
        return indices

    def _idf(self, index: int) -> float:
        if not self._state.fitted:
            return 1.0
        total = self._state.n_documents
        df = self._state.document_frequencies.get(str(index), 0)
        # Lucene's smoothed BM25 IDF: always positive, so a term appearing in
        # every document contributes ~0 rather than a negative score.
        return math.log(1.0 + (total - df + 0.5) / (df + 0.5))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._document_vector(t) for t in texts]

    def _document_vector(self, text: str) -> list[float]:
        """BM25 document-side weights, including length normalisation."""
        indices = self._indices(text)
        if not indices:
            return [0.0] * self.dimension

        counts: dict[int, float] = {}
        for index in indices:
            counts[index] = counts.get(index, 0.0) + 1.0

        average = self._state.average_length or float(len(indices))
        norm = self.K1 * (1.0 - self.B + self.B * (len(indices) / average)) if average else self.K1

        vector = [0.0] * self.dimension
        for index, tf in counts.items():
            vector[index] = self._idf(index) * (tf * (self.K1 + 1.0)) / (tf + norm)
        return vector

    def embed_query(self, text: str) -> list[float]:
        """Query-side indicator vector, normalised by query length.

        Binary rather than term-frequency weighted: a term the user happened to
        repeat should not count twice against the corpus. The IDF weighting
        already lives in the document vectors, so the inner product of the two
        reproduces the BM25 sum over matched terms.

        The sum is then divided by the number of query terms, which matters
        because the evidence grade is an absolute threshold. A raw BM25 sum grows
        with query length, so "Who must approve a demotion?" scores far below
        "Who must approve a demotion to a lower job level?" even though both find
        the same section at rank 1 -- and a fixed floor would reject the terse
        one. Dividing by query length turns the score into a mean per-term match
        quality, which is comparable across phrasings and is what the threshold
        can meaningfully be set against.
        """
        indices = set(self._indices(text))
        if not indices:
            return [0.0] * self.dimension

        weight = 1.0 / len(indices)
        vector = [0.0] * self.dimension
        for index in indices:
            vector[index] = weight
        return vector


class SentenceTransformerEmbedder:
    """Live provider backed by a sentence-transformers model."""

    similarity = SIMILARITY_COSINE

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only in live mode
            raise RuntimeError(
                "sentence-transformers is required for the live embedding provider. "
                "Install it, or run with POLICY_REVIEW_RUN_MODE=mock."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(list(texts), normalize_embeddings=True)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._model.encode([text], normalize_embeddings=True)[0]))


def hash_to_index(token: str, dimension: int) -> int:
    """Stable, process-independent hash.

    Python's built-in ``hash`` is salted per process, which would make the index
    non-reproducible across restarts -- unacceptable for a store that ingestion
    writes in one process and the runtime reads in another.
    """
    digest = 1469598103934665603
    for byte in token.encode("utf-8"):
        digest ^= byte
        digest = (digest * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return digest % dimension


def inner_product(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_embedder(
    provider: str,
    model_name: str = "",
    dimension: int = 16384,
    state: EmbedderState | None = None,
) -> EmbeddingModel:
    """Factory used by both the ingestion pipeline and the runtime retriever.

    Both layers must embed in the same space or retrieval is meaningless, so both
    call this with the same configured provider and -- for a fitted provider --
    the same published state.
    """
    provider = (provider or "hashing").lower()
    if provider in {"hashing", "hashing-bm25", "bm25", "mock"}:
        return HashingBM25Embedder(dimension=dimension, state=state)
    if provider in {"sentence-transformers", "sentence_transformer", "st"}:
        return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedding provider: {provider!r}")


def fit_embedder(
    provider: str, texts: Sequence[str], model_name: str = "", dimension: int = 16384
) -> EmbeddingModel:
    """Build an embedder and fit it to the corpus, where the provider supports it.

    Providers with no fitted state (a pretrained sentence-transformer) are
    returned unchanged, so the ingestion pipeline calls this unconditionally.
    """
    embedder = build_embedder(provider, model_name, dimension)
    fit = getattr(embedder, "fit", None)
    if callable(fit):
        fit(list(texts))
    return embedder


def embedder_state_of(embedder: EmbeddingModel) -> EmbedderState | None:
    state = getattr(embedder, "state", None)
    return state if isinstance(state, EmbedderState) else None


def coerce_embedder_state(raw: dict[str, Any] | None) -> EmbedderState | None:
    """Normalise whatever the embed step handed downstream into a state object.

    Inside a ZenML pipeline a step's outputs are passed as artifact references,
    not values, so the index and evaluation steps receive the *whole* embedding
    report rather than the state nested inside it. Accepting either shape keeps
    the pipeline definition readable instead of forcing an unwrapping step
    between them.
    """
    if not raw:
        return None
    if "document_frequencies" in raw or "n_documents" in raw:
        return EmbedderState.from_dict(raw)
    nested = raw.get("embedder_state")
    return EmbedderState.from_dict(nested) if nested else None


def save_embedder_state(directory: Path | str, state: EmbedderState | None) -> Path | None:
    """Publish fitted state into the index directory."""
    if state is None:
        return None
    path = Path(directory) / EMBEDDER_STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict()), encoding="utf-8")
    return path


def load_embedder_state(directory: Path | str) -> EmbedderState | None:
    """Load fitted state published beside an index, if any."""
    path = Path(directory) / EMBEDDER_STATE_FILENAME
    if not path.exists():
        return None
    try:
        return EmbedderState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None