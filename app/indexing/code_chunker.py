"""Code-aware chunking (Slices 2B / 2D).

Parsers tell us which symbols exist. This module decides which of those
units become retrievable chunks:

    CodeSymbol[]  →  Chunk[]  (text + symbol metadata)

Policy (why not 1 symbol = 1 chunk?):

* Functions, methods, types, and import groups → one chunk each.
* Classes / structs → a *header* chunk only (signature + docstring +
  attributes before the first child). Full type text would duplicate methods.
* Nested functions are already omitted by the parsers.
* Module-level leftovers → ``module`` chunks.

The index pipeline calls ``chunk_code_file`` for suffixes that have a
Tree-sitter extractor. Other files still use naive ``chunk_file`` windows.
"""

from __future__ import annotations

from app.indexing.chunker import Chunk
from app.parsing import CodeSymbol, SymbolKind, extract_python_symbols
from app.parsing.languages import extract_symbols, language_for_path

_KIND_FUNCTION = SymbolKind.FUNCTION.value
_KIND_CLASS = SymbolKind.CLASS.value
_KIND_METHOD = SymbolKind.METHOD.value
_KIND_IMPORTS = SymbolKind.IMPORTS.value
_KIND_MODULE = "module"


def chunk_code_file(path: str, content: str) -> list[Chunk]:
    """Turn a parsed language file into symbol-aware chunks.

    ``path`` chooses the extractor (``.py``, ``.js``, ``.go``, …).
    """
    language = language_for_path(path)
    if language is None:
        raise ValueError(f"No Tree-sitter extractor for path {path!r}")
    return chunks_from_symbols(
        path,
        content,
        extract_symbols(content, path=path),
        language=language,
    )


def chunk_python_file(path: str, content: str) -> list[Chunk]:
    """Turn Python source into symbol-aware chunks (Slice 2B helper)."""
    if not isinstance(content, str):
        raise TypeError(f"content must be str, got {type(content)!r}")
    if content == "":
        return []
    return chunks_from_symbols(
        path,
        content,
        extract_python_symbols(content, path=path),
        language="python",
    )


def chunks_from_symbols(
    path: str,
    content: str,
    symbols: list[CodeSymbol],
    *,
    language: str,
) -> list[Chunk]:
    if not isinstance(content, str):
        raise TypeError(f"content must be str, got {type(content)!r}")
    if content == "":
        return []

    lines = content.splitlines()
    chunks: list[Chunk] = []
    children_by_parent = _children_by_parent(symbols)

    for symbol in symbols:
        if symbol.kind == SymbolKind.CLASS:
            header = _class_header_chunk(
                lines,
                symbol,
                children_by_parent.get(symbol.qualified_name, []),
            )
            if header is not None:
                chunks.append(header)
            continue
        chunks.append(_chunk_from_symbol(symbol))

    chunks.extend(
        _module_remainder_chunks(path, lines, symbols, language=language)
    )
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
    """Keep the class prefix up to (but not including) the first overlapping child.

    Go methods live *after* the struct, so they do not shrink the header.
    JS/Python methods sit inside the class, so they do.
    """
    overlapping = [
        child
        for child in children
        if child.start_line <= class_symbol.end_line
        and child.end_line >= class_symbol.start_line
    ]
    if overlapping:
        header_end = min(child.start_line for child in overlapping) - 1
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
    *,
    language: str,
) -> list[Chunk]:
    """Chunk module-level lines that no symbol owns (e.g. ``x = 1``)."""
    n_lines = len(lines)
    covered: set[int] = set()
    for symbol in symbols:
        start = max(1, symbol.start_line)
        end = min(n_lines, symbol.end_line)
        if start <= end:
            covered.update(range(start, end + 1))

    chunks: list[Chunk] = []
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
                language=language,
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


FUNCTION_KIND = _KIND_FUNCTION
CLASS_KIND = _KIND_CLASS
METHOD_KIND = _KIND_METHOD
IMPORTS_KIND = _KIND_IMPORTS
MODULE_KIND = _KIND_MODULE
