"""Code parsing for Version 2.

Slice 2A extracts Python symbols with Tree-sitter.
Slice 2B turns those symbols into chunks (``app.indexing.code_chunker``).
"""

from app.parsing.python_parser import extract_python_symbols
from app.parsing.symbols import CodeSymbol, SymbolKind

__all__ = [
    "CodeSymbol",
    "SymbolKind",
    "extract_python_symbols",
]
