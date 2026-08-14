"""Structured code symbols extracted from a syntax tree.

A symbol is the parser's answer to: "what named code units exist in this
file, and where?" Slice 2B turns selected symbols into retrievable Chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SymbolKind(str, Enum):
    """What kind of code unit a symbol represents."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORTS = "imports"
    TYPE = "type"


@dataclass(frozen=True)
class CodeSymbol:
    """One named unit of code discovered by the parser.

    Attributes:
        path: Repo-relative path supplied by the caller (not read from disk).
        language: Source language id (``python``, ``javascript``, ``go``, …).
        name: Unqualified name (``create_user``, or ``imports``).
        qualified_name: Dotted name including parent classes
            (``UserService.create_user``).
        kind: Function, class, method, or import group.
        start_line: First line of the unit (1-based, inclusive).
        end_line: Last line of the unit (1-based, inclusive).
        text: Exact source slice for this unit, including decorators.
        parent: Qualified parent class, or None for module-level symbols.
    """

    path: str
    language: str
    name: str
    qualified_name: str
    kind: SymbolKind
    start_line: int
    end_line: int
    text: str
    parent: str | None = None
