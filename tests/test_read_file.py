"""Tests for sandboxed read_file (Slice 5B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools import ReadFileTool
from app.tools.paths import resolve_inside_root

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "mini_repo"


def test_resolve_inside_root_accepts_relative_path() -> None:
    resolved = resolve_inside_root(FIXTURE_REPO, "src/scoring.py")
    assert resolved == (FIXTURE_REPO / "src" / "scoring.py").resolve()
    assert resolved.is_file()


def test_resolve_inside_root_rejects_absolute_and_escape() -> None:
    with pytest.raises(ValueError, match="relative"):
        resolve_inside_root(FIXTURE_REPO, "/etc/passwd")
    with pytest.raises(ValueError, match="escapes"):
        resolve_inside_root(FIXTURE_REPO, "../README.md")


def test_read_file_spec_is_openai_shaped() -> None:
    spec = ReadFileTool(root=FIXTURE_REPO).spec()
    assert spec["type"] == "function"
    function = spec["function"]
    assert function["name"] == "read_file"
    assert "relative" in function["description"].lower()
    params = function["parameters"]
    assert params["required"] == ["path"]
    assert "path" in params["properties"]
    assert "start_line" in params["properties"]
    assert "end_line" in params["properties"]


def test_read_file_returns_numbered_lines() -> None:
    result = ReadFileTool(root=FIXTURE_REPO).run(path="src/scoring.py")
    assert result.name == "read_file"
    assert "src/scoring.py:1-" in result.content
    assert "compute_genuineness" in result.content
    assert "1|" in result.content
    assert "is_spam" in result.content


def test_read_file_respects_line_range() -> None:
    result = ReadFileTool(root=FIXTURE_REPO).run(
        path="src/scoring.py",
        start_line=10,
        end_line=11,
    )
    assert "src/scoring.py:10-11" in result.content
    assert "rank_score" in result.content
    assert "compute_genuineness" not in result.content
    assert "10|" in result.content


def test_read_file_rejects_path_escape_as_observation() -> None:
    result = ReadFileTool(root=FIXTURE_REPO).run(path="../../etc/passwd")
    assert result.content.startswith("error:")
    assert "escapes" in result.content


def test_read_file_rejects_absolute_path_as_observation() -> None:
    result = ReadFileTool(root=FIXTURE_REPO).run(path="/etc/passwd")
    assert "relative" in result.content


def test_read_file_missing_file() -> None:
    result = ReadFileTool(root=FIXTURE_REPO).run(path="src/missing.py")
    assert "not found" in result.content


def test_read_file_rejects_bad_range() -> None:
    result = ReadFileTool(root=FIXTURE_REPO).run(
        path="src/scoring.py",
        start_line=5,
        end_line=2,
    )
    assert "start_line must be <=" in result.content


def test_read_file_truncates_oversized_span() -> None:
    tool = ReadFileTool(root=FIXTURE_REPO, max_lines=3)
    result = tool.run(path="src/scoring.py", start_line=1, end_line=20)
    assert "truncated to 3 lines" in result.content
    assert "src/scoring.py:1-3" in result.content
    assert "4|" not in result.content
