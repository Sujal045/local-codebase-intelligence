"""Extract Python symbols using Tree-sitter (Slice 2A).

Tree-sitter is a parser generator. Given a *grammar* (here: Python), it
turns source bytes into a concrete syntax tree:

    source text
         │
         ▼
    Tree-sitter parser  (uses tree-sitter-python grammar)
         │
         ▼
    syntax tree (module → class_definition → function_definition → …)
         │
         ▼
    CodeSymbol list  (name, kind, lines, text)

We walk that tree ourselves instead of using query language, so the
extraction rules stay visible in Python.

This module does not chunk, embed, or talk to Qdrant.
"""

from __future__ import annotations

from tree_sitter import Language, Node, Parser
import tree_sitter_python as tspython

from app.parsing.symbols import CodeSymbol, SymbolKind
from app.parsing.tree import (
    field_text,
    make_callable_symbol,
    make_imports_symbol,
    make_named_symbol,
    parse_utf8,
)

_LANGUAGE = Language(tspython.language())
_PARSER = Parser(_LANGUAGE)

_DEFINITION_TYPES = frozenset({"function_definition", "class_definition"})
_IMPORT_TYPES = frozenset({"import_statement", "import_from_statement"})
_LANGUAGE_ID = "python"


def extract_python_symbols(source: str, *, path: str = "") -> list[CodeSymbol]:
    """Return functions, classes, methods, and import groups in ``source``.

    Nested functions (``def`` inside another ``def``) are skipped: they are
    usually implementation details, not retrieval units.

    Nested classes are extracted with dotted names (``Outer.Inner``).

    Decorators are included in the symbol span so ``@classmethod`` stays
    attached to the method that uses it.

    Args:
        source: Full Python file text.
        path: Identifier stored on every symbol (usually repo-relative).

    Returns:
        Symbols in source order. Empty source yields an empty list.
        Syntax errors do not raise: Tree-sitter still produces a tree, and
        we keep any definitions that parsed cleanly.
    """
    if not isinstance(source, str):
        raise TypeError(f"source must be str, got {type(source)!r}")
    if source == "":
        return []

    tree, source_bytes = parse_utf8(_PARSER, source)
    symbols: list[CodeSymbol] = []
    _walk_body(
        list(tree.root_node.named_children),
        symbols,
        path=path,
        class_stack=[],
        source_bytes=source_bytes,
    )
    return symbols


def _walk_body(
    nodes: list[Node],
    symbols: list[CodeSymbol],
    *,
    path: str,
    class_stack: list[str],
    source_bytes: bytes,
) -> None:
    """Walk one body (module or class block) and collect symbols."""
    import_group: list[Node] = []

    def flush_imports() -> None:
        nonlocal import_group
        if import_group and not class_stack:
            symbols.append(
                make_imports_symbol(
                    import_group,
                    path=path,
                    language=_LANGUAGE_ID,
                    source_bytes=source_bytes,
                )
            )
        import_group = []

    for node in nodes:
        if node.type in _IMPORT_TYPES and not class_stack:
            import_group.append(node)
            continue

        flush_imports()

        span_node, definition = _unwrap_definition(node)
        if definition is None:
            continue

        name = field_text(definition)
        if name is None:
            continue

        if definition.type == "function_definition":
            symbols.append(
                make_callable_symbol(
                    span_node,
                    name=name,
                    path=path,
                    language=_LANGUAGE_ID,
                    class_stack=class_stack,
                    source_bytes=source_bytes,
                )
            )
            # Do not walk the function body: nested defs are skipped.
            continue

        if definition.type == "class_definition":
            symbols.append(
                make_named_symbol(
                    span_node,
                    name=name,
                    path=path,
                    language=_LANGUAGE_ID,
                    kind=SymbolKind.CLASS,
                    class_stack=class_stack,
                    source_bytes=source_bytes,
                )
            )
            body = definition.child_by_field_name("body")
            if body is not None:
                _walk_body(
                    list(body.named_children),
                    symbols,
                    path=path,
                    class_stack=class_stack + [name],
                    source_bytes=source_bytes,
                )

    flush_imports()


def _unwrap_definition(node: Node) -> tuple[Node, Node | None]:
    """Return (span_node, definition_node).

    ``@decorator`` wraps a def/class in ``decorated_definition``. The inner
    node has the name; the outer node has the correct start line and text.
    """
    if node.type == "decorated_definition":
        for child in node.named_children:
            if child.type in _DEFINITION_TYPES:
                return node, child
        return node, None
    if node.type in _DEFINITION_TYPES:
        return node, node
    return node, None
