"""Map file suffixes to Tree-sitter extractors (Slice 2D).

The walker already collects many languages. Only some of them have a
grammar wired up. Unknown suffixes return None so the indexer can fall
back to naive line windows.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.parsing.go_parser import extract_go_symbols
from app.parsing.javascript_parser import (
    extract_javascript_symbols,
    extract_tsx_symbols,
    extract_typescript_symbols,
)
from app.parsing.python_parser import extract_python_symbols
from app.parsing.symbols import CodeSymbol

Extractor = Callable[..., list[CodeSymbol]]

# Suffix → language id used on CodeSymbol.language / Chunk.language.
SUFFIX_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
}

EXTRACTORS: dict[str, Extractor] = {
    "python": extract_python_symbols,
    "javascript": extract_javascript_symbols,
    "typescript": extract_typescript_symbols,
    "tsx": extract_tsx_symbols,
    "go": extract_go_symbols,
}

PARSED_SUFFIXES = frozenset(SUFFIX_LANGUAGE)


def language_for_path(path: str) -> str | None:
    """Return a language id, or None if we should use naive chunking."""
    suffix = Path(path).suffix.lower()
    return SUFFIX_LANGUAGE.get(suffix)


def extract_symbols(source: str, *, path: str) -> list[CodeSymbol]:
    """Dispatch to the extractor for ``path``'s suffix.

    Unknown suffixes yield an empty list (the caller should fall back).
    """
    if not isinstance(source, str):
        raise TypeError(f"source must be str, got {type(source)!r}")
    language = language_for_path(path)
    if language is None:
        return []
    return EXTRACTORS[language](source, path=path)
