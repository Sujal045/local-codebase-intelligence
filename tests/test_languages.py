"""Tests for JS/TS/Go parsers and suffix dispatch (Slice 2D)."""

from __future__ import annotations

import pytest

from app.indexing.code_chunker import chunk_code_file
from app.parsing import extract_symbols, language_for_path
from app.parsing.go_parser import extract_go_symbols
from app.parsing.javascript_parser import (
    extract_javascript_symbols,
    extract_tsx_symbols,
    extract_typescript_symbols,
)
from app.parsing.symbols import SymbolKind

JS_SAMPLE = """\
import { foo } from './foo';

export const LIMIT = 10;

export function greet(name) {
  function inner() {
    return name;
  }
  return inner();
}

export class UserService {
  constructor() {
    this.x = 1;
  }

  createUser(name) {
    return name;
  }
}

const helper = (x) => x + 1;
"""

TS_SAMPLE = """\
import type { User } from './user';

export interface Job {
  company?: string;
}

export type Score = number;

export function rank(a: number, b: number): number {
  return a + b;
}
"""

GO_SAMPLE = """\
package scoring

import (
    "fmt"
    "os"
)

type UserService struct {
    Name string
}

func (s *UserService) CreateUser(name string) string {
    return name
}

func Rank(a, b int) int {
    return a + b
}

const Limit = 10
"""


def _names(symbols) -> list[str]:
    return [s.qualified_name for s in symbols]


def test_language_for_path_known_and_unknown() -> None:
    assert language_for_path("app/cli.py") == "python"
    assert language_for_path("web/app.tsx") == "tsx"
    assert language_for_path("cmd/main.go") == "go"
    assert language_for_path("README.md") is None
    assert language_for_path("lib/parser.rs") is None


def test_extract_symbols_dispatches_by_suffix() -> None:
    js = extract_symbols("export function add(a, b) { return a + b; }", path="a.js")
    go = extract_symbols("package p\nfunc Add(a, b int) int { return a + b }\n", path="a.go")
    md = extract_symbols("# hello\n", path="README.md")
    assert js[0].language == "javascript"
    assert js[0].name == "add"
    assert go[0].language == "go"
    assert go[0].name == "Add"
    assert md == []


def test_javascript_functions_classes_methods_and_arrows() -> None:
    symbols = extract_javascript_symbols(JS_SAMPLE, path="src/app.js")
    names = _names(symbols)
    assert names == [
        "imports",
        "greet",
        "UserService",
        "UserService.constructor",
        "UserService.createUser",
        "helper",
    ]
    found = {s.qualified_name: s for s in symbols}
    assert found["greet"].kind == SymbolKind.FUNCTION
    assert found["greet"].text.startswith("export function greet")
    assert found["UserService"].kind == SymbolKind.CLASS
    assert found["UserService.createUser"].kind == SymbolKind.METHOD
    assert found["UserService.createUser"].parent == "UserService"
    assert found["helper"].kind == SymbolKind.FUNCTION
    assert "inner" not in names


def test_javascript_does_not_treat_const_number_as_function() -> None:
    symbols = extract_javascript_symbols(JS_SAMPLE)
    assert "LIMIT" not in _names(symbols)


def test_typescript_interfaces_and_aliases_are_types() -> None:
    symbols = extract_typescript_symbols(TS_SAMPLE, path="src/types.ts")
    found = {s.qualified_name: s for s in symbols}
    assert found["Job"].kind == SymbolKind.TYPE
    assert found["Job"].language == "typescript"
    assert found["Score"].kind == SymbolKind.TYPE
    assert found["rank"].kind == SymbolKind.FUNCTION


def test_tsx_extracts_component_function() -> None:
    source = """\
export function Button(props: { label: string }) {
  return <button>{props.label}</button>;
}
"""
    symbols = extract_tsx_symbols(source, path="Button.tsx")
    assert symbols[0].name == "Button"
    assert symbols[0].language == "tsx"
    assert "<button>" in symbols[0].text


def test_go_struct_method_and_function() -> None:
    symbols = extract_go_symbols(GO_SAMPLE, path="scoring.go")
    found = {s.qualified_name: s for s in symbols}
    assert found["imports"].kind == SymbolKind.IMPORTS
    assert found["UserService"].kind == SymbolKind.CLASS
    assert found["UserService.CreateUser"].kind == SymbolKind.METHOD
    assert found["UserService.CreateUser"].parent == "UserService"
    assert found["Rank"].kind == SymbolKind.FUNCTION
    assert found["Rank"].language == "go"
    assert "Limit" not in found


def test_js_chunks_include_class_header_without_methods() -> None:
    chunks = chunk_code_file("src/app.js", JS_SAMPLE)
    found = {c.symbol: c for c in chunks}
    assert found["UserService"].kind == "class"
    assert "createUser" not in found["UserService"].text
    assert found["UserService.createUser"].kind == "method"
    assert found["helper"].kind == "function"
    assert found["<module>"].text == "export const LIMIT = 10;"


def test_go_struct_header_keeps_full_type_block() -> None:
    chunks = chunk_code_file("scoring.go", GO_SAMPLE)
    found = {c.symbol: c for c in chunks}
    assert "type UserService struct" in found["UserService"].text
    assert "CreateUser" not in found["UserService"].text
    assert found["UserService.CreateUser"].kind == "method"
    module_texts = {c.text for c in chunks if c.kind == "module"}
    assert "package scoring" in module_texts
    assert "const Limit = 10" in module_texts


def test_rejects_non_string_javascript() -> None:
    with pytest.raises(TypeError, match="source must be str"):
        extract_javascript_symbols(b"function f() {}")  # type: ignore[arg-type]
