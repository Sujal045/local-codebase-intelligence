"""Shared Tree-sitter node helpers (Slice 2D).

Parsers for each language still walk their own node types. These helpers
only hide byte/point conversions so line numbers stay consistent.

Important: Tree-sitter's C tree points at the *source bytes* we pass to
``parse()``. If that ``bytes`` object is a temporary (``source.encode()``
inline), Python may free it while we still read ``start_point`` / ``text``.
That shows up as garbage line numbers and can segfault. Callers must keep
the ``source_bytes`` returned by ``parse_utf8`` alive until the walk
finishes.
"""

from __future__ import annotations

from tree_sitter import Node, Parser, Tree

from app.parsing.symbols import CodeSymbol, SymbolKind


def parse_utf8(parser: Parser, source: str) -> tuple[Tree, bytes]:
    """Parse ``source`` and keep the UTF-8 buffer alive for the tree."""
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree, source_bytes


def node_text(node: Node) -> str:
    raw = node.text
    if raw is None:
        return ""
    return raw.decode("utf-8")


def line_span(node: Node, source_bytes: bytes) -> tuple[int, int]:
    """1-based inclusive line numbers from byte offsets (not ``start_point``).

    ``start_point.row`` can be wrong if the source buffer was collected.
    Byte offsets stay valid for the lifetime of ``source_bytes``.
    Character offsets are *not* the same as UTF-8 byte offsets, so we
    count newlines on the bytes object, not the Python ``str``.
    """
    start_line = source_bytes.count(b"\n", 0, node.start_byte) + 1
    end_byte = node.end_byte
    if end_byte <= node.start_byte:
        return start_line, start_line
    end_line = source_bytes.count(b"\n", 0, end_byte - 1) + 1
    return start_line, end_line


def field_text(node: Node, field: str = "name") -> str | None:
    name_node = node.child_by_field_name(field)
    if name_node is None or name_node.text is None:
        return None
    name = name_node.text.decode("utf-8")
    return name or None


def make_callable_symbol(
    span_node: Node,
    *,
    name: str,
    path: str,
    language: str,
    class_stack: list[str],
    source_bytes: bytes,
) -> CodeSymbol:
    if class_stack:
        kind = SymbolKind.METHOD
        qualified = ".".join(class_stack + [name])
        parent = ".".join(class_stack)
    else:
        kind = SymbolKind.FUNCTION
        qualified = name
        parent = None
    start_line, end_line = line_span(span_node, source_bytes)
    return CodeSymbol(
        path=path,
        language=language,
        name=name,
        qualified_name=qualified,
        kind=kind,
        start_line=start_line,
        end_line=end_line,
        text=node_text(span_node),
        parent=parent,
    )


def make_named_symbol(
    span_node: Node,
    *,
    name: str,
    path: str,
    language: str,
    kind: SymbolKind,
    source_bytes: bytes,
    class_stack: list[str] | None = None,
) -> CodeSymbol:
    class_stack = class_stack or []
    qualified = ".".join(class_stack + [name])
    parent = ".".join(class_stack) if class_stack else None
    start_line, end_line = line_span(span_node, source_bytes)
    return CodeSymbol(
        path=path,
        language=language,
        name=name,
        qualified_name=qualified,
        kind=kind,
        start_line=start_line,
        end_line=end_line,
        text=node_text(span_node),
        parent=parent,
    )


def make_imports_symbol(
    nodes: list[Node],
    *,
    path: str,
    language: str,
    source_bytes: bytes,
) -> CodeSymbol:
    start_line, _ = line_span(nodes[0], source_bytes)
    _, end_line = line_span(nodes[-1], source_bytes)
    text = "\n".join(node_text(node) for node in nodes)
    return CodeSymbol(
        path=path,
        language=language,
        name="imports",
        qualified_name="imports",
        kind=SymbolKind.IMPORTS,
        start_line=start_line,
        end_line=end_line,
        text=text,
        parent=None,
    )
