"""Code parsing for Version 2 (Slice 2A).

Slice 2A extracts Python symbols with Tree-sitter. It does not replace
the Version 1 line-window chunker yet — that happens in a later slice.
"""

from app.parsing.python_parser import extract_python_symbols
from app.parsing.symbols import CodeSymbol, SymbolKind

__all__ = [
    "CodeSymbol",
    "SymbolKind",
    "extract_python_symbols",
]
