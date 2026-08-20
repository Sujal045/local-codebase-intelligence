"""``get_symbol`` and ``find_references`` tools (Slice 5C).

Tree-sitter already stored ``symbol`` / ``name`` / ``kind`` on each chunk.
We do **not** yet have a call graph (Version 8). This slice uses that
metadata plus simple text matching:

    get_symbol("compute_genuineness")
        → chunks whose symbol/name *is* that definition

    find_references("compute_genuineness")
        → other chunks whose *body text* mentions the name
          (approximate; can false-positive on short names)

Typical agent flow later (Version 6):

    search_code("spam") → see compute_genuineness
    get_symbol("compute_genuineness") → full definition chunk
    find_references("compute_genuineness") → call sites
    read_file(...) → surrounding lines if needed
"""

from __future__ import annotations

import re
from typing import Any

from app.indexing.chunker import Chunk
from app.llm.prompt import format_context
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk
from app.tools.base import ToolResult

GET_SYMBOL_NAME = "get_symbol"
FIND_REFERENCES_NAME = "find_references"

DEFAULT_SYMBOL_LIMIT = 5
DEFAULT_REFERENCE_LIMIT = 10

# Chunk kinds that represent a *definition* site (not module leftovers).
DEFINITION_KINDS = frozenset({"function", "class", "method", "type"})

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def symbol_names(query: str) -> list[str]:
    """Names to match: full query plus the last dotted segment."""
    stripped = query.strip()
    if not stripped:
        return []
    names = [stripped]
    if "." in stripped:
        names.append(stripped.rsplit(".", 1)[-1])
    return list(dict.fromkeys(names))


def definition_rank(chunk: Chunk, query: str) -> float | None:
    """How well ``chunk`` is a definition of ``query``, or None if not."""
    if chunk.kind not in DEFINITION_KINDS:
        return None
    stripped = query.strip()
    if not stripped:
        return None
    if chunk.symbol == stripped:
        return 3.0
    if chunk.name == stripped:
        return 2.0
    if chunk.symbol and chunk.symbol.endswith("." + stripped):
        return 1.0
    return None


def text_mentions_symbol(text: str, query: str) -> bool:
    """True if ``text`` contains the symbol as a whole identifier."""
    for name in symbol_names(query):
        if not name:
            continue
        if not _IDENT.fullmatch(name) and "." not in name:
            # Unusual token; fall back to substring for dotted forms only.
            if name in text:
                return True
            continue
        if "." in name:
            if name in text:
                return True
            continue
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
        if re.search(pattern, text):
            return True
    return False


def find_definitions(
    chunks: list[Chunk],
    query: str,
    *,
    limit: int = DEFAULT_SYMBOL_LIMIT,
) -> list[ScoredChunk]:
    """Return definition chunks for ``query``, best match first."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    ranked: list[ScoredChunk] = []
    for chunk in chunks:
        score = definition_rank(chunk, query)
        if score is None:
            continue
        ranked.append(_scored(chunk, score))
    ranked.sort(key=lambda hit: (-hit.score, hit.path, hit.start_line, hit.end_line))
    return ranked[:limit]


def find_reference_sites(
    chunks: list[Chunk],
    query: str,
    *,
    limit: int = DEFAULT_REFERENCE_LIMIT,
) -> list[ScoredChunk]:
    """Return non-definition chunks whose text mentions ``query``."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if not query.strip():
        return []

    hits: list[ScoredChunk] = []
    for chunk in chunks:
        if definition_rank(chunk, query) is not None:
            continue
        if not text_mentions_symbol(chunk.text, query):
            continue
        hits.append(_scored(chunk, 1.0))
    hits.sort(key=lambda hit: (hit.path, hit.start_line, hit.end_line))
    return hits[:limit]


def _scored(chunk: Chunk, score: float) -> ScoredChunk:
    return ScoredChunk(
        path=chunk.path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        text=chunk.text,
        score=score,
        language=chunk.language,
        symbol=chunk.symbol,
        kind=chunk.kind,
        name=chunk.name,
        parent=chunk.parent,
    )


class GetSymbolTool:
    """Look up the definition chunk(s) for a symbol name."""

    name = GET_SYMBOL_NAME

    def __init__(self, *, store: QdrantVectorStore) -> None:
        self._store = store

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Look up where a symbol is *defined* (function, class, "
                    "method, or type) using indexed metadata. Prefer this "
                    "over search_code when you already know the exact name "
                    "(e.g. 'compute_genuineness' or 'UserService.create_user')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": (
                                "Unqualified or qualified symbol name "
                                "(e.g. 'compute_genuineness')."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                f"Maximum definitions to return "
                                f"(default {DEFAULT_SYMBOL_LIMIT})."
                            ),
                            "minimum": 1,
                        },
                    },
                    "required": ["symbol"],
                },
            },
        }

    def run(self, **arguments: Any) -> ToolResult:
        parsed = _parse_symbol_args(arguments, default_limit=DEFAULT_SYMBOL_LIMIT)
        if isinstance(parsed, str):
            return ToolResult(name=self.name, content=parsed)
        symbol, limit = parsed

        try:
            hits = find_definitions(self._store.list_chunks(), symbol, limit=limit)
        except Exception as exc:
            return ToolResult(name=self.name, content=f"error: {exc}")

        if not hits:
            return ToolResult(
                name=self.name,
                content=f"No definitions found for {symbol!r}.",
            )
        header = f"Found {len(hits)} definition(s) for {symbol!r}:\n\n"
        return ToolResult(
            name=self.name,
            content=header + format_context(hits),
            hits=tuple(hits),
        )


class FindReferencesTool:
    """Find approximate textual reference sites for a symbol."""

    name = FIND_REFERENCES_NAME

    def __init__(self, *, store: QdrantVectorStore) -> None:
        self._store = store

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Find code chunks that *mention* a symbol but are not "
                    "its definition. Matching is lexical (identifier in the "
                    "chunk text), not a full call graph — short names can "
                    "false-positive. Use after get_symbol to see call sites."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": (
                                "Symbol to search for in chunk bodies "
                                "(e.g. 'compute_genuineness')."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                f"Maximum reference chunks to return "
                                f"(default {DEFAULT_REFERENCE_LIMIT})."
                            ),
                            "minimum": 1,
                        },
                    },
                    "required": ["symbol"],
                },
            },
        }

    def run(self, **arguments: Any) -> ToolResult:
        parsed = _parse_symbol_args(
            arguments,
            default_limit=DEFAULT_REFERENCE_LIMIT,
        )
        if isinstance(parsed, str):
            return ToolResult(name=self.name, content=parsed)
        symbol, limit = parsed

        try:
            hits = find_reference_sites(
                self._store.list_chunks(),
                symbol,
                limit=limit,
            )
        except Exception as exc:
            return ToolResult(name=self.name, content=f"error: {exc}")

        if not hits:
            return ToolResult(
                name=self.name,
                content=(
                    f"No reference sites found for {symbol!r} "
                    "(excluding definition chunks)."
                ),
            )
        header = (
            f"Found {len(hits)} reference site(s) for {symbol!r} "
            "(excluding definitions):\n\n"
        )
        return ToolResult(
            name=self.name,
            content=header + format_context(hits),
            hits=tuple(hits),
        )


def _parse_symbol_args(
    arguments: dict[str, Any],
    *,
    default_limit: int,
) -> tuple[str, int] | str:
    """Return ``(symbol, limit)`` or an ``error: ...`` observation string."""
    extra = sorted(set(arguments) - {"symbol", "limit"})
    if extra:
        return f"error: unexpected arguments {extra}"

    symbol = arguments.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return "error: symbol must be a non-empty string"

    limit = arguments.get("limit", default_limit)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        return "error: limit must be an integer >= 1"
    return symbol.strip(), limit
