"""
Tree-sitter-based C parser for function and struct extraction.
Produces AST-level chunks with calls_functions and uses_structs metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tree_sitter
    from tree_sitter import Language, Parser, Node
except ImportError:
    tree_sitter = None  # type: ignore[assignment]

# Optional: tree_sitter_c provides C grammar
_C_LANGUAGE: Any = None
_PARSER: Parser | None = None


def _init_c_language() -> bool:
    """Load Tree-sitter C language. Returns True if available."""
    global _C_LANGUAGE, _PARSER
    if _C_LANGUAGE is not None:
        return _PARSER is not None
    if tree_sitter is None:
        return False
    # Prefer tree-sitter-c; fallback to tree-sitter-languages
    try:
        from tree_sitter_c import language as c_lang_fn
        _C_LANGUAGE = c_lang_fn()
        _PARSER = Parser(_C_LANGUAGE)
        return True
    except Exception:
        pass
    try:
        from tree_sitter_languages import get_parser
        _PARSER = get_parser("c")
        _C_LANGUAGE = _PARSER.language if hasattr(_PARSER, "language") else True
        return True
    except Exception:
        return False


@dataclass
class CChunk:
    """One C chunk (function or struct) with Graph RAG metadata."""
    start_line: int
    end_line: int
    code_snippet: str
    name: str | None  # function or struct name
    kind: str  # "function" | "struct"
    calls_functions: list[str]
    uses_structs: list[str]


def _node_text(source_bytes: bytes, node: Node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _line_of_offset(source: str, offset: int) -> int:
    return source[:offset].count("\n") + 1


def _declarator_name(source_bytes: bytes, node: Node | None) -> str | None:
    """Recursively find identifier name in a declarator (function_declarator, pointer_declarator, etc.)."""
    if not node:
        return None
    if node.type == "identifier":
        return _node_text(source_bytes, node).strip() or None
    child = node.child_by_field_name("declarator")
    return _declarator_name(source_bytes, child) if child else None


def _extract_calls_and_structs(source_bytes: bytes, root: Node, source: str) -> tuple[list[str], list[str]]:
    """Walk tree and collect function call names and struct type usages."""
    calls: list[str] = []
    structs: set[str] = set()

    def visit(node: Node) -> None:
        type_name = node.type
        if type_name == "call_expression":
            # First child is usually the function name (identifier)
            child = node.child_by_field_name("function")
            if child and child.type == "identifier":
                name = _node_text(source_bytes, child).strip()
                if name and name not in ("sizeof", "typeof", "alignof"):
                    calls.append(name)
        elif type_name == "struct_specifier":
            decl = node.child_by_field_name("name")
            if decl and decl.type == "type_identifier":
                structs.add(_node_text(source_bytes, decl).strip())
        elif type_name == "type_identifier":
            # Could be a struct type use (e.g. "struct foo" already handled by struct_specifier)
            pass
        for i in range(node.child_count):
            visit(node.child(i))

    visit(root)
    return calls, list(structs)


def _extract_from_node(
    parser: Parser,
    source: str,
    source_bytes: bytes,
    node: Node,
    file_path: str,
) -> CChunk | None:
    """Turn a function_definition or struct_specifier node into a CChunk."""
    start_line = _line_of_offset(source, node.start_byte)
    end_line = _line_of_offset(source, node.end_byte)
    snippet = _node_text(source_bytes, node)

    if node.type == "function_definition":
        decl = node.child_by_field_name("declarator")
        name = _declarator_name(source_bytes, decl) if decl else None
        body = node.child_by_field_name("body")
        calls, structs = _extract_calls_and_structs(source_bytes, node, source) if body else ([], [])
        return CChunk(
            start_line=start_line,
            end_line=end_line,
            code_snippet=snippet,
            name=name or None,
            kind="function",
            calls_functions=calls,
            uses_structs=structs,
        )

    if node.type == "struct_specifier":
        name_node = node.child_by_field_name("name")
        name = _node_text(source_bytes, name_node).strip() if name_node else None
        body_node = node.child_by_field_name("body")
        calls: list[str] = []
        structs: list[str] = []
        if body_node:
            calls, structs = _extract_calls_and_structs(source_bytes, body_node, source)
        return CChunk(
            start_line=start_line,
            end_line=end_line,
            code_snippet=snippet,
            name=name or None,
            kind="struct",
            calls_functions=calls,
            uses_structs=structs,
        )

    return None


def chunk_c_ast(file_path: Path | str, text: str) -> list[CChunk]:
    """
    Parse C/C++ source with Tree-sitter and return one chunk per function_definition
    and struct_specifier (at file scope). Returns [] if Tree-sitter is unavailable or parse fails.
    """
    if not _init_c_language() or not text.strip():
        return []
    parser = _PARSER
    assert parser is not None
    source_bytes = text.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node
    if not root:
        return []

    chunks: list[CChunk] = []
    for i in range(root.child_count):
        node = root.child(i)
        if node.type in ("function_definition", "struct_specifier"):
            c = _extract_from_node(parser, text, source_bytes, node, str(file_path))
            if c:
                chunks.append(c)
    return chunks


def is_c_ast_available() -> bool:
    return _init_c_language()
