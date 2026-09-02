"""Tests for context engineering primitives (Slice 7A)."""

from __future__ import annotations

import pytest

from app.context import (
    CHARS_PER_TOKEN,
    TRUNCATION_MARKER,
    ContextBudget,
    dedupe_chunks,
    estimate_tokens,
    location_key,
    truncate_text,
)
from app.retrieval.vector_store import ScoredChunk


def _scored(
    path: str = "src/a.py",
    start: int = 1,
    end: int = 5,
    text: str = "def foo():\n    pass",
    score: float = 0.9,
) -> ScoredChunk:
    return ScoredChunk(
        path=path,
        start_line=start,
        end_line=end,
        text=text,
        score=score,
    )


# --- estimate_tokens ---


def test_estimate_tokens_empty_is_zero() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_non_empty_at_least_one() -> None:
    assert estimate_tokens("x") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * CHARS_PER_TOKEN) == 1
    assert estimate_tokens("a" * (CHARS_PER_TOKEN + 1)) == 2


# --- ContextBudget ---


def test_context_budget_available() -> None:
    budget = ContextBudget(max_tokens=1000, reserved_tokens=200)
    assert budget.available == 800


def test_context_budget_fits() -> None:
    budget = ContextBudget(max_tokens=20, reserved_tokens=10)
    assert budget.available == 10
    assert budget.fits("x" * 40)  # 10 tokens
    assert not budget.fits("x" * 41)


def test_context_budget_remaining_after() -> None:
    budget = ContextBudget(max_tokens=50, reserved_tokens=10)
    assert budget.remaining_after(15) == 25
    assert budget.remaining_after(100) == 0


def test_context_budget_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ContextBudget(max_tokens=0)
    with pytest.raises(ValueError, match="reserved_tokens"):
        ContextBudget(max_tokens=10, reserved_tokens=-1)
    with pytest.raises(ValueError, match="reserved_tokens must be less"):
        ContextBudget(max_tokens=10, reserved_tokens=10)


# --- truncate_text ---


def test_truncate_text_unchanged_when_fits() -> None:
    text = "line one\nline two"
    assert truncate_text(text, estimate_tokens(text)) == text


def test_truncate_text_zero_budget_returns_empty() -> None:
    assert truncate_text("hello", 0) == ""


def test_truncate_text_start_keeps_beginning_and_marker() -> None:
    lines = "\n".join(f"line {i}" for i in range(50))
    result = truncate_text(lines, 10, strategy="start")
    assert result.startswith("line 0")
    assert TRUNCATION_MARKER in result
    assert "line 49" not in result


def test_truncate_text_end_keeps_tail_and_marker() -> None:
    lines = "\n".join(f"line {i}" for i in range(50))
    result = truncate_text(lines, 10, strategy="end")
    assert "line 49" in result
    assert TRUNCATION_MARKER in result
    assert "line 0" not in result


def test_truncate_text_middle_keeps_head_and_tail() -> None:
    lines = "\n".join(f"line {i}" for i in range(80))
    result = truncate_text(lines, 20, strategy="middle")
    assert result.startswith("line 0")
    assert "line 79" in result
    assert TRUNCATION_MARKER in result
    assert "line 40" not in result


def test_truncate_text_rejects_negative_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        truncate_text("hello", -1)


# --- dedupe_chunks ---


def test_location_key() -> None:
    assert location_key("a.py", 1, 10) == ("a.py", 1, 10)


def test_dedupe_chunks_keeps_first_occurrence() -> None:
    first = _scored(path="a.py", start=1, end=5, score=0.95)
    duplicate = _scored(path="a.py", start=1, end=5, score=0.50, text="other body")
    other = _scored(path="b.py", start=10, end=20, score=0.80)
    result = dedupe_chunks([first, duplicate, other])
    assert result == [first, other]


def test_dedupe_chunks_preserves_order() -> None:
    a = _scored(path="a.py", start=1, end=2)
    b = _scored(path="b.py", start=3, end=4)
    c = _scored(path="c.py", start=5, end=6)
    assert dedupe_chunks([c, a, b]) == [c, a, b]


def test_dedupe_chunks_empty() -> None:
    assert dedupe_chunks([]) == []


# --- ContextBuilder (Slice 7B) ---


def test_order_lost_in_middle_places_best_at_edges() -> None:
    from app.context.builder import order_lost_in_middle

    chunks = [_scored(path=f"{i}.py", score=1.0 - i * 0.1) for i in range(5)]
    ordered = order_lost_in_middle(chunks)
    assert ordered[0].path == "0.py"
    assert ordered[-1].path == "1.py"
    assert [chunk.path for chunk in ordered] == ["0.py", "2.py", "4.py", "3.py", "1.py"]


def test_context_builder_dedupes_before_budgeting() -> None:
    from app.context import ContextBudget, ContextBuilder

    duplicate = _scored(path="a.py", start=1, end=5, score=0.9)
    copy = _scored(path="a.py", start=1, end=5, score=0.5, text="duplicate body")
    other = _scored(path="b.py", start=1, end=5, score=0.8)
    built = ContextBuilder().build(
        [duplicate, copy, other],
        ContextBudget(max_tokens=500, reserved_tokens=0),
    )
    assert built.deduped == 1
    assert len(built.chunks) == 2
    assert built.chunks[0].path == "a.py"


def test_context_builder_truncates_oversized_chunk() -> None:
    from app.context import ContextBudget, ContextBuilder, TRUNCATION_MARKER

    huge = _scored(text="x\n" * 500, score=0.95)
    built = ContextBuilder().build(
        [huge],
        ContextBudget(max_tokens=30, reserved_tokens=0),
    )
    assert built.truncated == 1
    assert TRUNCATION_MARKER in built.text
    assert built.tokens_used <= 30


def test_context_builder_drops_chunks_when_budget_full() -> None:
    from app.context import ContextBudget, ContextBuilder

    chunks = [
        _scored(path="a.py", text="a" * 200, score=0.95),
        _scored(path="b.py", text="b" * 200, score=0.90),
        _scored(path="c.py", text="c" * 200, score=0.85),
    ]
    built = ContextBuilder().build(
        chunks,
        ContextBudget(max_tokens=40, reserved_tokens=0),
    )
    assert len(built.chunks) < len(chunks)
    assert built.dropped >= 1
    assert built.tokens_used <= 40
