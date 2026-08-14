"""JavaScript / TypeScript symbol extraction (Slice 2D).

JS and TS share most node types (``function_declaration``,
``class_declaration``, ``method_definition``). TypeScript adds
``interface_declaration`` and ``type_alias_declaration``.

``export`` wraps a declaration the same way Python ``@decorator`` does:
the inner node has the name; the outer node has the correct span.
"""

from __future__ import annotations

from tree_sitter import Language, Node, Parser
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript

from app.parsing.symbols import CodeSymbol, SymbolKind
from app.parsing.tree import (
    field_text,
    make_callable_symbol,
    make_imports_symbol,
    make_named_symbol,
    parse_utf8,
)

_JS_PARSER = Parser(Language(tsjavascript.language()))
_TS_PARSER = Parser(Language(tstypescript.language_typescript()))
_TSX_PARSER = Parser(Language(tstypescript.language_tsx()))

_FUNCTION_DECLS = frozenset(
    {"function_declaration", "generator_function_declaration"}
)
_CLASS_DECLS = frozenset({"class_declaration"})
_TYPE_DECLS = frozenset({"interface_declaration", "type_alias_declaration"})
_BINDINGS = frozenset({"lexical_declaration", "variable_declaration"})
_VALUE_FUNCTIONS = frozenset(
    {"arrow_function", "function_expression", "generator_function"}
)
_IMPORT_TYPES = frozenset({"import_statement"})


def extract_javascript_symbols(source: str, *, path: str = "") -> list[CodeSymbol]:
    return _extract_ecma(source, path=path, language="javascript", parser=_JS_PARSER)


def extract_typescript_symbols(source: str, *, path: str = "") -> list[CodeSymbol]:
    return _extract_ecma(source, path=path, language="typescript", parser=_TS_PARSER)


def extract_tsx_symbols(source: str, *, path: str = "") -> list[CodeSymbol]:
    return _extract_ecma(source, path=path, language="tsx", parser=_TSX_PARSER)


def _extract_ecma(
    source: str,
    *,
    path: str,
    language: str,
    parser: Parser,
) -> list[CodeSymbol]:
    if not isinstance(source, str):
        raise TypeError(f"source must be str, got {type(source)!r}")
    if source == "":
        return []

    tree, source_bytes = parse_utf8(parser, source)
    symbols: list[CodeSymbol] = []
    _walk_body(
        list(tree.root_node.named_children),
        symbols,
        path=path,
        language=language,
        class_stack=[],
        source_bytes=source_bytes,
    )
    return symbols


def _walk_body(
    nodes: list[Node],
    symbols: list[CodeSymbol],
    *,
    path: str,
    language: str,
    class_stack: list[str],
    source_bytes: bytes,
) -> None:
    import_group: list[Node] = []

    def flush_imports() -> None:
        nonlocal import_group
        if import_group and not class_stack:
            symbols.append(
                make_imports_symbol(
                    import_group,
                    path=path,
                    language=language,
                    source_bytes=source_bytes,
                )
            )
        import_group = []

    for node in nodes:
        if node.type in _IMPORT_TYPES and not class_stack:
            import_group.append(node)
            continue

        flush_imports()

        if node.type == "method_definition":
            name = field_text(node)
            if name:
                symbols.append(
                    make_callable_symbol(
                        node,
                        name=name,
                        path=path,
                        language=language,
                        class_stack=class_stack,
                        source_bytes=source_bytes,
                    )
                )
            continue

        span_node, inner = _unwrap_export(node)
        if inner is None:
            continue

        if inner.type in _FUNCTION_DECLS:
            name = field_text(inner)
            if name:
                symbols.append(
                    make_callable_symbol(
                        span_node,
                        name=name,
                        path=path,
                        language=language,
                        class_stack=class_stack,
                        source_bytes=source_bytes,
                    )
                )
            continue

        if inner.type in _CLASS_DECLS:
            name = field_text(inner)
            if not name:
                continue
            symbols.append(
                make_named_symbol(
                    span_node,
                    name=name,
                    path=path,
                    language=language,
                    kind=SymbolKind.CLASS,
                    class_stack=class_stack,
                    source_bytes=source_bytes,
                )
            )
            body = inner.child_by_field_name("body")
            if body is not None:
                _walk_body(
                    list(body.named_children),
                    symbols,
                    path=path,
                    language=language,
                    class_stack=class_stack + [name],
                    source_bytes=source_bytes,
                )
            continue

        if inner.type in _TYPE_DECLS:
            name = field_text(inner)
            if name:
                symbols.append(
                    make_named_symbol(
                        span_node,
                        name=name,
                        path=path,
                        language=language,
                        kind=SymbolKind.TYPE,
                        source_bytes=source_bytes,
                    )
                )
            continue

        if inner.type in _BINDINGS and not class_stack:
            _extract_bound_functions(
                span_node,
                inner,
                symbols,
                path=path,
                language=language,
                source_bytes=source_bytes,
            )

    flush_imports()


def _unwrap_export(node: Node) -> tuple[Node, Node | None]:
    if node.type != "export_statement":
        return node, node
    for child in node.named_children:
        if child.type in (
            _FUNCTION_DECLS
            | _CLASS_DECLS
            | _TYPE_DECLS
            | _BINDINGS
        ):
            return node, child
    return node, None


def _extract_bound_functions(
    span_node: Node,
    declaration: Node,
    symbols: list[CodeSymbol],
    *,
    path: str,
    language: str,
    source_bytes: bytes,
) -> None:
    """``const greet = (name) => ...`` counts as a named function."""
    for declarator in declaration.named_children:
        if declarator.type != "variable_declarator":
            continue
        name = field_text(declarator)
        value = declarator.child_by_field_name("value")
        if not name or value is None or value.type not in _VALUE_FUNCTIONS:
            continue
        symbols.append(
            make_callable_symbol(
                span_node,
                name=name,
                path=path,
                language=language,
                class_stack=[],
                source_bytes=source_bytes,
            )
        )
