"""Tests for Tree-sitter Python symbol extraction (Slice 2A)."""

from __future__ import annotations

import pytest

from app.parsing import CodeSymbol, SymbolKind, extract_python_symbols

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


def _by_qualified(symbols: list[CodeSymbol]) -> dict[str, CodeSymbol]:
    return {symbol.qualified_name: symbol for symbol in symbols}


def test_empty_source_returns_no_symbols() -> None:
    assert extract_python_symbols("") == []


def test_rejects_non_string_source() -> None:
    with pytest.raises(TypeError, match="source must be str"):
        extract_python_symbols(b"def f():\n    pass\n")  # type: ignore[arg-type]


def test_extracts_functions_classes_methods_and_imports() -> None:
    symbols = extract_python_symbols(SAMPLE, path="src/scoring.py")
    found = _by_qualified(symbols)

    assert [s.qualified_name for s in symbols] == [
        "imports",
        "greet",
        "UserService",
        "UserService.__init__",
        "UserService.create_user",
        "UserService.fetch",
        "leftover",
    ]

    imports = found["imports"]
    assert imports.kind == SymbolKind.IMPORTS
    assert imports.path == "src/scoring.py"
    assert imports.language == "python"
    assert imports.start_line == 1
    assert imports.end_line == 2
    assert "import os" in imports.text
    assert "from pathlib import Path" in imports.text

    greet = found["greet"]
    assert greet.kind == SymbolKind.FUNCTION
    assert greet.parent is None
    assert greet.start_line == 6
    assert greet.end_line == 10
    assert greet.text.startswith("@decorator")
    assert "def greet" in greet.text

    service = found["UserService"]
    assert service.kind == SymbolKind.CLASS
    assert service.start_line == 12
    assert service.end_line == 23
    assert "class UserService" in service.text

    init = found["UserService.__init__"]
    assert init.kind == SymbolKind.METHOD
    assert init.name == "__init__"
    assert init.parent == "UserService"
    assert init.start_line == 15
    assert init.end_line == 16

    create = found["UserService.create_user"]
    assert create.kind == SymbolKind.METHOD
    assert create.start_line == 18
    assert create.text.startswith("@classmethod")

    fetch = found["UserService.fetch"]
    assert fetch.kind == SymbolKind.METHOD
    assert "async def fetch" in fetch.text

    leftover = found["leftover"]
    assert leftover.kind == SymbolKind.FUNCTION
    assert leftover.start_line == 25
    assert leftover.end_line == 26


def test_nested_function_is_not_extracted() -> None:
    symbols = extract_python_symbols(SAMPLE)
    names = {symbol.qualified_name for symbol in symbols}
    assert "inner" not in names
    assert "greet.inner" not in names


def test_nested_class_uses_dotted_qualified_name() -> None:
    source = """\
class Outer:
    class Inner:
        def method(self):
            pass
"""
    symbols = extract_python_symbols(source, path="nested.py")
    found = _by_qualified(symbols)

    assert found["Outer"].kind == SymbolKind.CLASS
    assert found["Outer.Inner"].kind == SymbolKind.CLASS
    assert found["Outer.Inner"].parent == "Outer"
    assert found["Outer.Inner.method"].kind == SymbolKind.METHOD
    assert found["Outer.Inner.method"].parent == "Outer.Inner"
    assert found["Outer.Inner.method"].path == "nested.py"


def test_split_import_groups_are_separate_symbols() -> None:
    source = """\
import os

x = 1

import sys
"""
    symbols = extract_python_symbols(source)
    imports = [s for s in symbols if s.kind == SymbolKind.IMPORTS]
    assert len(imports) == 2
    assert imports[0].text == "import os"
    assert imports[0].start_line == 1
    assert imports[1].text == "import sys"
    assert imports[1].start_line == 5


def test_syntax_error_still_extracts_valid_definitions() -> None:
    source = """\
def good():
    return 1

def broken(
"""
    symbols = extract_python_symbols(source)
    names = [s.qualified_name for s in symbols]
    assert "good" in names


def test_assignment_only_file_has_no_definitions() -> None:
    symbols = extract_python_symbols("x = 1\ny = 2\n")
    assert symbols == []
