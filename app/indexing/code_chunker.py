"""Code-aware chunking for Python (Slice 2B).

Slice 2A tells us which symbols exist. This module decides which of those
units become retrievable chunks:

    CodeSymbol[]  →  Chunk[]  (text + symbol metadata)

Policy (why not 1 symbol = 1 chunk?):

* Functions, methods, and import groups → one chunk each (whole unit).
* Classes → a *header* chunk only (signature + docstring + attributes
  before the first child). The full class text would duplicate every method.
* Nested ``def`` is already omitted by the parser.
* Module-level leftovers (constants, module docstring) → ``module`` chunks
  so we do not drop code the parser does not name.

This module does not embed, talk to Qdrant, or replace the index pipeline.
Unsupported languages still use ``chunk_file`` until a later slice.
"""

from __future__ import annotations

from app.indexing.chunker import Chunk
from app.parsing import CodeSymbol, SymbolKind, extract_python_symbols

_KIND_FUNCTION = SymbolKind.FUNCTION.value
_KIND_CLASS = SymbolKind.CLASS.value
_KIND_METHOD = SymbolKind.METHOD.value
_KIND_IMPORTS = SymbolKind.IMPORTS.value
_KIND_MODULE = "module"


def chunk_python_file(path: str, content: str) -> list[Chunk]:
    """Turn Python source into symbol-aware chunks.

    Args:
        path: Identifier stored on every chunk (usually repo-relative).
        content: Full file text.

    Returns:
        Chunks in source order. Empty content yields an empty list.
    """
    if not isinstance(content, str):
        raise TypeError(f"content must be str, got {type(content)!r}")
    if content == "":
        return []

    lines = content.splitlines()
    symbols = extract_python_symbols(content, path=path)
    chunks: list[Chunk] = []

    children_by_parent = _children_by_parent(symbols)

    for symbol in symbols:
        if symbol.kind == SymbolKind.CLASS:
            header = _class_header_chunk(lines, symbol, children_by_parent.get(symbol.qualified_name, []))
            if header is not None:
                chunks.append(header)
            continue
        chunks.append(_chunk_from_symbol(symbol))

    chunks.extend(_module_remainder_chunks(path, lines, symbols))
    chunks.sort(key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.kind or ""))
    return chunks


def _children_by_parent(symbols: list[CodeSymbol]) -> dict[str, list[CodeSymbol]]:
    grouped: dict[str, list[CodeSymbol]] = {}
    for symbol in symbols:
        if symbol.parent is None:
            continue
        grouped.setdefault(symbol.parent, []).append(symbol)
    return grouped


def _class_header_chunk(
    lines: list[str],
    class_symbol: CodeSymbol,
    children: list[CodeSymbol],
) -> Chunk | None:
    """Keep the class prefix up to (but not including) the first child.

    Example::

        class UserService:          ← kept
            \"\"\"A service.\"\"\"  ← kept

            def create_user(...):   ← first child; not part of the header
    """
    if children:
        header_end = min(child.start_line for child in children) - 1
    else:
        header_end = class_symbol.end_line
    header_end = min(header_end, class_symbol.end_line)
    header_start = class_symbol.start_line
    if header_end < header_start:
        return None

    while header_end >= header_start and not lines[header_end - 1].strip():
        header_end -= 1
    if header_end < header_start:
        return None

    text = "\n".join(lines[header_start - 1 : header_end])
    return _chunk_from_symbol(
        class_symbol,
        text=text,
        start_line=header_start,
        end_line=header_end,
    )


def _module_remainder_chunks(
    path: str,
    lines: list[str],
    symbols: list[CodeSymbol],
) -> list[Chunk]:
    """Chunk module-level lines that no symbol owns (e.g. ``x = 1``)."""
    covered: set[int] = set()
    for symbol in symbols:
        covered.update(range(symbol.start_line, symbol.end_line + 1))

    chunks: list[Chunk] = []
    n_lines = len(lines)
    start: int | None = None

    def flush(end: int) -> None:
        nonlocal start
        if start is None:
            return
        trimmed_start, trimmed_end = _trim_blank_edges(lines, start, end)
        start = None
        if trimmed_start is None or trimmed_end is None:
            return
        chunks.append(
            Chunk(
                text="\n".join(lines[trimmed_start - 1 : trimmed_end]),
                path=path,
                start_line=trimmed_start,
                end_line=trimmed_end,
                language="python",
                symbol="<module>",
                kind=_KIND_MODULE,
                name="<module>",
                parent=None,
            )
        )

    for line_no in range(1, n_lines + 1):
        if line_no in covered:
            if start is not None:
                flush(line_no - 1)
            continue
        if start is None:
            start = line_no

    if start is not None:
        flush(n_lines)
    return chunks


def _trim_blank_edges(
    lines: list[str],
    start: int,
    end: int,
) -> tuple[int | None, int | None]:
    while start <= end and not lines[start - 1].strip():
        start += 1
    while end >= start and not lines[end - 1].strip():
        end -= 1
    if start > end:
        return None, None
    return start, end


def _chunk_from_symbol(
    symbol: CodeSymbol,
    *,
    text: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> Chunk:
    return Chunk(
        text=symbol.text if text is None else text,
        path=symbol.path,
        start_line=symbol.start_line if start_line is None else start_line,
        end_line=symbol.end_line if end_line is None else end_line,
        language=symbol.language,
        symbol=symbol.qualified_name,
        kind=symbol.kind.value,
        name=symbol.name,
        parent=symbol.parent,
    )


# Re-export kind strings so tests can assert without importing the enum.
FUNCTION_KIND = _KIND_FUNCTION
CLASS_KIND = _KIND_CLASS
METHOD_KIND = _KIND_METHOD
IMPORTS_KIND = _KIND_IMPORTS
MODULE_KIND = _KIND_MODULE
