"""Code parsing for Version 2.

Slice 2A extracts Python symbols with Tree-sitter.
Slice 2B turns symbols into chunks.
Slice 2D adds JS/TS/Go extractors and suffix dispatch.
"""

from app.parsing.languages import (
    PARSED_SUFFIXES,
    extract_symbols,
    language_for_path,
)
from app.parsing.python_parser import extract_python_symbols
from app.parsing.symbols import CodeSymbol, SymbolKind

__all__ = [
    "CodeSymbol",
    "PARSED_SUFFIXES",
    "SymbolKind",
    "extract_python_symbols",
    "extract_symbols",
    "language_for_path",
]
