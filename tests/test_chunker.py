"""Tests for naive line-window chunking (Slice 1a)."""

from __future__ import annotations

import pytest

from app.indexing.chunker import Chunk, chunk_file


def test_empty_content_returns_no_chunks() -> None:
    assert chunk_file("a.py", "") == []


def test_single_short_file_one_chunk() -> None:
    content = "alpha\nbeta\ngamma"
    chunks = chunk_file("demo.py", content, chunk_size=40, overlap=10)

    assert len(chunks) == 1
    assert chunks[0] == Chunk(
        text="alpha\nbeta\ngamma",
        path="demo.py",
        start_line=1,
        end_line=3,
    )


def test_overlapping_windows_on_95_lines() -> None:
    sample = "\n".join(f"line {i}" for i in range(1, 96))
    chunks = chunk_file("demo.py", sample, chunk_size=40, overlap=10)

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 40
    assert chunks[0].text.startswith("line 1")
    assert chunks[0].text.endswith("line 40")

    # Next window starts at 40 - 10 + 1 = 31
    assert chunks[1].start_line == 31
    assert chunks[1].end_line == 70

    assert chunks[-1].end_line == 95
    assert all(c.path == "demo.py" for c in chunks)
    assert all(c.end_line - c.start_line + 1 <= 40 for c in chunks)

    # Full coverage: every line appears in at least one chunk
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start_line, chunk.end_line + 1))
    assert covered == set(range(1, 96))


def test_overlap_zero_advances_by_chunk_size() -> None:
    sample = "\n".join(f"line {i}" for i in range(1, 21))
    chunks = chunk_file("x.py", sample, chunk_size=10, overlap=0)

    assert [ (c.start_line, c.end_line) for c in chunks ] == [
        (1, 10),
        (11, 20),
    ]


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap must be < chunk_size"):
        chunk_file("a.py", "x", chunk_size=10, overlap=10)


def test_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        chunk_file("a.py", "x", chunk_size=0, overlap=0)
