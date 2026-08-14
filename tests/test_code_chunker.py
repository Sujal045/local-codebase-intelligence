"""Tests for Python code-aware chunking (Slice 2B)."""

from __future__ import annotations

import pytest

from app.indexing.code_chunker import (
    CLASS_KIND,
    FUNCTION_KIND,
    IMPORTS_KIND,
    METHOD_KIND,
    MODULE_KIND,
    chunk_python_file,
)
from app.indexing.chunker import Chunk

SAMPLE = """\
import os
from pathlib import Path

x = 1

@decorator
def greet(name):
    def inner():
        return name
    return inner()

class UserService:
    \"\"\"A service.\"\"\"

    def __init__(self):
        self.x = 1

    @classmethod
    def create_user(cls, name):
        return cls()

    async def fetch(self):
        return 1

def leftover():
    pass
"""


def _by_symbol(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {chunk.symbol: chunk for chunk in chunks if chunk.symbol}


def test_empty_content_returns_no_chunks() -> None:
    assert chunk_python_file("a.py", "") == []


def test_rejects_non_string_content() -> None:
    with pytest.raises(TypeError, match="content must be str"):
        chunk_python_file("a.py", b"def f():\n    pass\n")  # type: ignore[arg-type]


def test_python_chunks_carry_symbol_metadata() -> None:
    chunks = chunk_python_file("src/scoring.py", SAMPLE)
    found = _by_symbol(chunks)

    assert found["imports"].kind == IMPORTS_KIND
    assert found["imports"].language == "python"
    assert found["imports"].path == "src/scoring.py"
    assert found["imports"].start_line == 1
    assert found["imports"].end_line == 2
    assert "import os" in found["imports"].text

    greet = found["greet"]
    assert greet.kind == FUNCTION_KIND
    assert greet.name == "greet"
    assert greet.parent is None
    assert greet.text.startswith("@decorator")
    assert "def inner" in greet.text

    leftover = found["leftover"]
    assert leftover.kind == FUNCTION_KIND
    assert leftover.start_line == 25
    assert leftover.end_line == 26


def test_class_header_excludes_method_bodies() -> None:
    chunks = chunk_python_file("src/scoring.py", SAMPLE)
    found = _by_symbol(chunks)

    header = found["UserService"]
    assert header.kind == CLASS_KIND
    assert header.start_line == 12
    assert header.end_line == 13
    assert "class UserService" in header.text
    assert '"""A service."""' in header.text
    assert "def __init__" not in header.text
    assert "def create_user" not in header.text
    assert "async def fetch" not in header.text


def test_methods_are_separate_chunks_with_parent() -> None:
    chunks = chunk_python_file("src/scoring.py", SAMPLE)
    found = _by_symbol(chunks)

    init = found["UserService.__init__"]
    assert init.kind == METHOD_KIND
    assert init.parent == "UserService"
    assert init.start_line == 15

    create = found["UserService.create_user"]
    assert create.text.startswith("@classmethod")
    assert create.name == "create_user"

    fetch = found["UserService.fetch"]
    assert "async def fetch" in fetch.text


def test_nested_function_is_not_its_own_chunk() -> None:
    chunks = chunk_python_file("a.py", SAMPLE)
    symbols = {chunk.symbol for chunk in chunks}
    assert "inner" not in symbols
    assert "greet.inner" not in symbols


def test_module_remainder_keeps_unnamed_code() -> None:
    chunks = chunk_python_file("a.py", SAMPLE)
    module_chunks = [c for c in chunks if c.kind == MODULE_KIND]
    assert len(module_chunks) == 1
    remainder = module_chunks[0]
    assert remainder.symbol == "<module>"
    assert remainder.text == "x = 1"
    assert remainder.start_line == 4
    assert remainder.end_line == 4


def test_assignment_only_file_becomes_module_chunk() -> None:
    chunks = chunk_python_file("consts.py", "x = 1\ny = 2\n")
    assert len(chunks) == 1
    assert chunks[0].kind == MODULE_KIND
    assert chunks[0].text == "x = 1\ny = 2"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2


def test_empty_class_is_one_header_chunk() -> None:
    source = "class Marker:\n    pass\n"
    chunks = chunk_python_file("m.py", source)
    assert [c.symbol for c in chunks] == ["Marker"]
    assert chunks[0].kind == CLASS_KIND
    assert "pass" in chunks[0].text


def test_nested_class_header_does_not_include_inner_method() -> None:
    source = """\
class Outer:
    class Inner:
        def method(self):
            pass
"""
    chunks = chunk_python_file("nested.py", source)
    found = _by_symbol(chunks)
    assert "def method" not in found["Outer"].text
    assert "def method" not in found["Outer.Inner"].text
    assert found["Outer.Inner.method"].kind == METHOD_KIND
    assert found["Outer.Inner.method"].parent == "Outer.Inner"


def test_chunks_are_ordered_by_source_position() -> None:
    chunks = chunk_python_file("a.py", SAMPLE)
    starts = [c.start_line for c in chunks]
    assert starts == sorted(starts)
