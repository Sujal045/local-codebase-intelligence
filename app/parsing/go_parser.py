"""Go symbol extraction (Slice 2D).

Go keeps methods *next to* types, not nested inside them:

    type UserService struct { ... }          ← class
    func (s *UserService) CreateUser(...)    ← method, parent UserService
    func Rank(a, b int) int                  ← function

The chunker still builds a class header from line spans: methods sit
outside the struct, so the header is the whole type block.
"""

from __future__ import annotations

from tree_sitter import Language, Node, Parser
import tree_sitter_go as tsgo

from app.parsing.symbols import CodeSymbol, SymbolKind
from app.parsing.tree import (
    field_text,
    make_callable_symbol,
    make_imports_symbol,
    make_named_symbol,
    node_text,
    parse_utf8,
)

_PARSER = Parser(Language(tsgo.language()))
_LANGUAGE_ID = "go"
_IMPORT_TYPES = frozenset({"import_declaration"})


def extract_go_symbols(source: str, *, path: str = "") -> list[CodeSymbol]:
    if not isinstance(source, str):
        raise TypeError(f"source must be str, got {type(source)!r}")
    if source == "":
        return []

    tree, source_bytes = parse_utf8(_PARSER, source)
    symbols: list[CodeSymbol] = []
    import_group: list[Node] = []

    def flush_imports() -> None:
        nonlocal import_group
        if import_group:
            symbols.append(
                make_imports_symbol(
                    import_group,
                    path=path,
                    language=_LANGUAGE_ID,
                    source_bytes=source_bytes,
                )
            )
        import_group = []

    for node in tree.root_node.named_children:
        if node.type in _IMPORT_TYPES:
            import_group.append(node)
            continue
        flush_imports()

        if node.type == "function_declaration":
            name = field_text(node)
            if name:
                symbols.append(
                    make_callable_symbol(
                        node,
                        name=name,
                        path=path,
                        language=_LANGUAGE_ID,
                        class_stack=[],
                        source_bytes=source_bytes,
                    )
                )
            continue

        if node.type == "method_declaration":
            name = field_text(node)
            receiver = _receiver_type(node)
            stack = [receiver] if name and receiver else []
            if name:
                symbols.append(
                    make_callable_symbol(
                        node,
                        name=name,
                        path=path,
                        language=_LANGUAGE_ID,
                        class_stack=stack,
                        source_bytes=source_bytes,
                    )
                )
            continue

        if node.type == "type_declaration":
            _extract_type_declaration(
                node, symbols, path=path, source_bytes=source_bytes
            )

    flush_imports()
    return symbols


def _extract_type_declaration(
    node: Node,
    symbols: list[CodeSymbol],
    *,
    path: str,
    source_bytes: bytes,
) -> None:
    for spec in node.named_children:
        if spec.type != "type_spec":
            continue
        name = field_text(spec)
        if not name:
            continue
        type_node = spec.child_by_field_name("type")
        kind = SymbolKind.CLASS
        if type_node is not None and type_node.type != "struct_type":
            kind = SymbolKind.TYPE
        symbols.append(
            make_named_symbol(
                node,
                name=name,
                path=path,
                language=_LANGUAGE_ID,
                kind=kind,
                source_bytes=source_bytes,
            )
        )


def _receiver_type(method_node: Node) -> str | None:
    receiver = method_node.child_by_field_name("receiver")
    if receiver is None:
        return None
    return _first_type_identifier(receiver)


def _first_type_identifier(node: Node) -> str | None:
    if node.type == "type_identifier":
        text = node_text(node)
        return text or None
    for child in node.named_children:
        found = _first_type_identifier(child)
        if found:
            return found
    return None
