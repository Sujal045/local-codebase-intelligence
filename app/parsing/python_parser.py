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

    tree = _PARSER.parse(source.encode("utf-8"))
    symbols: list[CodeSymbol] = []
    _walk_body(
        list(tree.root_node.named_children),
        symbols,
        path=path,
        class_stack=[],
    )
    return symbols


def _walk_body(
    nodes: list[Node],
    symbols: list[CodeSymbol],
    *,
    path: str,
    class_stack: list[str],
) -> None:
    """Walk one body (module or class block) and collect symbols."""
    import_group: list[Node] = []

    def flush_imports() -> None:
        nonlocal import_group
        if import_group and not class_stack:
            symbols.append(_imports_symbol(import_group, path=path))
        import_group = []

    for node in nodes:
        if node.type in _IMPORT_TYPES and not class_stack:
            import_group.append(node)
            continue

        flush_imports()

        span_node, definition = _unwrap_definition(node)
        if definition is None:
            continue

        name = _definition_name(definition)
        if name is None:
            continue

        if definition.type == "function_definition":
            symbols.append(
                _function_or_method_symbol(
                    span_node,
                    name=name,
                    path=path,
                    class_stack=class_stack,
                )
            )
            # Do not walk the function body: nested defs are skipped.
            continue

        if definition.type == "class_definition":
            qualified = ".".join(class_stack + [name])
            parent = ".".join(class_stack) if class_stack else None
            start_line, end_line = _line_span(span_node)
            symbols.append(
                CodeSymbol(
                    path=path,
                    language=_LANGUAGE_ID,
                    name=name,
                    qualified_name=qualified,
                    kind=SymbolKind.CLASS,
                    start_line=start_line,
                    end_line=end_line,
                    text=_node_text(span_node),
                    parent=parent,
                )
            )
            body = definition.child_by_field_name("body")
            if body is not None:
                _walk_body(
                    list(body.named_children),
                    symbols,
                    path=path,
                    class_stack=class_stack + [name],
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


def _function_or_method_symbol(
    span_node: Node,
    *,
    name: str,
    path: str,
    class_stack: list[str],
) -> CodeSymbol:
    if class_stack:
        kind = SymbolKind.METHOD
        qualified = ".".join(class_stack + [name])
        parent = ".".join(class_stack)
    else:
        kind = SymbolKind.FUNCTION
        qualified = name
        parent = None
    start_line, end_line = _line_span(span_node)
    return CodeSymbol(
        path=path,
        language=_LANGUAGE_ID,
        name=name,
        qualified_name=qualified,
        kind=kind,
        start_line=start_line,
        end_line=end_line,
        text=_node_text(span_node),
        parent=parent,
    )


def _imports_symbol(nodes: list[Node], *, path: str) -> CodeSymbol:
    start_line, _ = _line_span(nodes[0])
    _, end_line = _line_span(nodes[-1])
    text = "\n".join(_node_text(node) for node in nodes)
    return CodeSymbol(
        path=path,
        language=_LANGUAGE_ID,
        name="imports",
        qualified_name="imports",
        kind=SymbolKind.IMPORTS,
        start_line=start_line,
        end_line=end_line,
        text=text,
        parent=None,
    )


def _definition_name(node: Node) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is None or name_node.text is None:
        return None
    name = name_node.text.decode("utf-8")
    return name or None


def _node_text(node: Node) -> str:
    raw = node.text
    if raw is None:
        return ""
    return raw.decode("utf-8")


def _line_span(node: Node) -> tuple[int, int]:
    """Convert Tree-sitter 0-based points to 1-based inclusive line numbers.

    ``end_point`` is the position *after* the last character. If that lands
    at column 0, the node ended at the previous line's newline.
    """
    start_line = node.start_point.row + 1
    end_row = node.end_point.row
    end_col = node.end_point.column
    if end_col == 0 and end_row > node.start_point.row:
        end_line = end_row
    else:
        end_line = end_row + 1
    return start_line, end_line
