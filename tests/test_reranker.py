"""Tests for cross-encoder reranking (Slice 4A)."""

from __future__ import annotations

import os

import pytest

from app.reranking import CrossEncoderReranker, apply_scores
from app.reranking.reranker import _import_failure_message
from app.retrieval.vector_store import ScoredChunk
from tests.fakes import FakeReranker


def _chunk(
    path: str,
    text: str,
    *,
    start: int = 1,
    end: int = 8,
    score: float = 0.0,
    symbol: str | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        path=path,
        start_line=start,
        end_line=end,
        text=text,
        score=score,
        language="python",
        symbol=symbol,
        kind="function" if symbol else None,
        name=symbol,
    )


def test_apply_scores_sorts_descending_and_truncates() -> None:
    candidates = [
        _chunk("a.py", "aaa", score=0.99, symbol="a"),
        _chunk("b.py", "bbb", score=0.50, symbol="b"),
        _chunk("c.py", "ccc", score=0.10, symbol="c"),
    ]
    ranked = apply_scores(candidates, [0.2, 0.9, 0.4], limit=2)

    assert [hit.symbol for hit in ranked] == ["b", "c"]
    assert ranked[0].score == 0.9
    assert ranked[1].score == 0.4
    assert ranked[0].path == "b.py"
    assert ranked[0].language == "python"


def test_apply_scores_tie_breaks_on_path() -> None:
    left = _chunk("b.py", "same")
    right = _chunk("a.py", "same")
    ranked = apply_scores([left, right], [1.0, 1.0], limit=2)
    assert [hit.path for hit in ranked] == ["a.py", "b.py"]


def test_apply_scores_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="2 scores for 1"):
        apply_scores([_chunk("a.py", "code")], [0.1, 0.2], limit=1)


def test_apply_scores_rejects_bad_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        apply_scores([], [], limit=0)


def test_cross_encoder_scores_query_and_text_together() -> None:
    captured: list[tuple[str, str]] = []

    class StubModel:
        def predict(self, pairs, show_progress_bar=False):
            captured.extend(pairs)
            return [0.1, 0.8]

    reranker = CrossEncoderReranker(model=StubModel())
    query = "Where is spam detected?"
    noisy = _chunk("app/chunker.py", "def split_windows(lines):\n    return lines")
    relevant = _chunk(
        "src/scoring.py",
        "def compute_genuineness(job):\n    return {'is_spam': False}",
        symbol="compute_genuineness",
    )

    ranked = reranker.rerank(query, [noisy, relevant], limit=1)

    assert captured == [
        (query, noisy.text),
        (query, relevant.text),
    ]
    assert ranked[0].symbol == "compute_genuineness"
    assert ranked[0].score == 0.8


def test_cross_encoder_empty_candidates() -> None:
    class Exploding:
        def predict(self, pairs, show_progress_bar=False):
            raise AssertionError("should not score an empty list")

    reranker = CrossEncoderReranker(model=Exploding())
    assert reranker.rerank("spam?", []) == []


def test_cross_encoder_rejects_empty_query() -> None:
    reranker = CrossEncoderReranker(model=object())
    with pytest.raises(ValueError, match="query must be non-empty"):
        reranker.rerank("  ", [_chunk("a.py", "code")])


def test_import_error_for_missing_package_is_actionable() -> None:
    exc = ModuleNotFoundError("No module named 'sentence_transformers'")
    exc.name = "sentence_transformers"
    message = _import_failure_message(exc)
    assert "pip install -r requirements.txt" in message
    assert "Pillow" not in message


def test_import_error_from_old_pillow_is_not_masked() -> None:
    """transformers may raise ModuleNotFoundError for PreTrainedModel."""
    exc = ModuleNotFoundError(
        "Could not import module 'PreTrainedModel'. "
        "Are this object's requirements defined correctly?"
    )
    message = _import_failure_message(exc)
    assert "dependency could not load" in message
    assert "pillow>=10" in message
    assert "sentence-transformers is required" not in message


def test_cross_encoder_rejects_score_count_mismatch() -> None:
    class StubModel:
        def predict(self, pairs, show_progress_bar=False):
            return [0.5]

    reranker = CrossEncoderReranker(model=StubModel())
    with pytest.raises(RuntimeError, match="1 scores for 2"):
        reranker.rerank("q", [_chunk("a.py", "a"), _chunk("b.py", "b")])


def test_fake_reranker_promotes_lexical_overlap() -> None:
    """Stand-in for the neural net: query tokens in the body rank higher."""
    unrelated = _chunk("app/chunker.py", "def split_windows(lines):\n    return lines")
    relevant = _chunk(
        "src/scoring.py",
        "def compute_genuineness(job):\n    flag spam jobs as is_spam",
        symbol="compute_genuineness",
    )
    ranked = FakeReranker().rerank(
        "detect spam jobs",
        [unrelated, relevant],
        limit=2,
    )
    assert ranked[0].symbol == "compute_genuineness"
    assert ranked[0].score > ranked[1].score


@pytest.mark.integration
def test_live_bge_reranker_prefers_relevant_chunk() -> None:
    """Optional: downloads BAAI/bge-reranker-base on first run (~1 GB).

    Example::

        pip install -r requirements.txt
        LIVE_RERANKER=1 PYTHONPATH=. pytest tests/test_reranker.py -m integration -v
    """
    if os.environ.get("LIVE_RERANKER") != "1":
        pytest.skip("Set LIVE_RERANKER=1 to load the real cross-encoder")

    reranker = CrossEncoderReranker()
    noisy = _chunk("app/chunker.py", "def split_windows(lines):\n    return lines[i:j]")
    relevant = _chunk(
        "src/scoring.py",
        "def compute_genuineness(job):\n    return {'is_spam': job looks like spam}",
        symbol="compute_genuineness",
    )
    ranked = reranker.rerank(
        "Where is spam detected?",
        [noisy, relevant],
        limit=2,
    )
    assert ranked[0].symbol == "compute_genuineness"
    assert ranked[0].score > ranked[1].score
