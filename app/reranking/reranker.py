"""Cross-encoder reranking of retrieval candidates (Slice 4A).

A *bi-encoder* (our nomic embedder) encodes the query and each chunk
separately, then compares the two vectors. That is fast enough to search
thousands of chunks, but the model never sees the question and the code
in the same forward pass.

A *cross-encoder* takes ``(query, chunk)`` as one pair and outputs a
relevance score. It is slower (one pass per candidate) and more precise,
which is why we only run it on a shortlist (the hybrid top 20–50), not on
the whole index.

    hybrid retrieval  →  candidate pool  →  this module  →  top-k for the LLM

Slice 4B (``search_and_rerank``) fetches a broad fused pool and calls
this module. ``ask()`` still uses RRF order until Slice 4C.

The neural net itself lives in ``sentence-transformers``. We own the
contract: pair construction, score attachment, sort, and truncation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from app.retrieval.vector_store import ScoredChunk

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANK_DEVICE = "cpu"


@runtime_checkable
class Reranker(Protocol):
    """Anything that can reorder candidate chunks for a query."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        *,
        limit: int = 5,
    ) -> list[ScoredChunk]:
        """Return the top ``limit`` candidates, highest score first."""


def apply_scores(
    candidates: Sequence[ScoredChunk],
    scores: Sequence[float],
    *,
    limit: int,
) -> list[ScoredChunk]:
    """Attach scores, sort descending, keep the top ``limit`` chunks.

    Ties break on ``(path, start_line, end_line)`` so the order is
    deterministic. Previous retriever scores (cosine, BM25, RRF) are
    replaced — a rerank score is not on the same scale.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if len(candidates) != len(scores):
        raise ValueError(
            f"got {len(scores)} scores for {len(candidates)} candidates"
        )

    scored = [_with_score(chunk, float(score)) for chunk, score in zip(candidates, scores)]
    scored.sort(key=lambda chunk: (-chunk.score, chunk.path, chunk.start_line, chunk.end_line))
    return scored[:limit]


class CrossEncoderReranker:
    """Score ``(query, chunk.text)`` pairs with a local cross-encoder.

    The Hugging Face model is loaded lazily on the first ``rerank`` call
    so unit tests can inject a stub via ``model=``. Device defaults to
    CPU because this project has no discrete GPU.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        model: Any = None,
        device: str = DEFAULT_RERANK_DEVICE,
    ) -> None:
        self._model_name = model_name
        self._model = model
        self._device = device

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        *,
        limit: int = 5,
    ) -> list[ScoredChunk]:
        stripped = query.strip()
        if not stripped:
            raise ValueError("query must be non-empty")
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if not candidates:
            return []

        pairs = [(stripped, chunk.text) for chunk in candidates]
        raw = self._predict(pairs)
        scores = [float(value) for value in raw]
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )
        return apply_scores(candidates, scores, limit=limit)

    def _predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        model = self._get_model()
        try:
            return model.predict(pairs, show_progress_bar=False)
        except TypeError:
            return model.predict(pairs)

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _load_cross_encoder(self._model_name, device=self._device)
        return self._model


def _load_cross_encoder(model_name: str, *, device: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(_import_failure_message(exc)) from exc

    try:
        return CrossEncoder(model_name, device=device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load reranker {model_name!r}: {exc}. "
            "The first run downloads weights from Hugging Face "
            "(about 1 GB for BAAI/bge-reranker-base). "
            "Need network access and free disk under ~/.cache/huggingface."
        ) from exc


def _import_failure_message(exc: BaseException) -> str:
    """Explain why ``from sentence_transformers import CrossEncoder`` failed.

    ``ModuleNotFoundError`` is a subclass of ``ImportError``. Transformers can
    raise it for *its own* broken import graph (for example an old Pillow
    missing ``PIL.Image.Resampling``). That is not the same as
    sentence-transformers being uninstalled.
    """
    missing = getattr(exc, "name", None)
    if isinstance(exc, ModuleNotFoundError) and missing == "sentence_transformers":
        return (
            "sentence-transformers is required for CrossEncoderReranker. "
            "Install with: pip install -r requirements.txt"
        )
    return (
        "Failed to import CrossEncoder because a dependency could not load "
        f"({type(exc).__name__}: {exc}). "
        "A common cause is Pillow < 9.1 (need PIL.Image.Resampling). "
        "Fix with: pip install -U 'pillow>=10'"
    )


def _with_score(chunk: ScoredChunk, score: float) -> ScoredChunk:
    return ScoredChunk(
        path=chunk.path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        text=chunk.text,
        score=score,
        language=chunk.language,
        symbol=chunk.symbol,
        kind=chunk.kind,
        name=chunk.name,
        parent=chunk.parent,
    )
