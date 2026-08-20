"""Code intelligence tools (Version 5).

Slice 5A: ``search_code`` — hybrid/rerank retrieval as a named function.
Slice 5B: ``read_file`` — sandboxed file read with optional line range.
Slice 5C: ``get_symbol`` / ``find_references`` — definition lookup and
lexical mention scan (not a call graph).
Slice 5D: ``search_documentation`` — BM25 over README / docs paths only.

The LLM does not call these yet. Version 6 will send ``spec()`` to Ollama
and run ``tool.run(**arguments)`` when the model requests a tool.
"""

from __future__ import annotations

from app.tools.base import Tool, ToolResult
from app.tools.read_file import READ_FILE_NAME, ReadFileTool
from app.tools.search_code import SEARCH_CODE_NAME, SearchCodeTool
from app.tools.search_docs import SEARCH_DOCUMENTATION_NAME, SearchDocumentationTool
from app.tools.symbol_lookup import (
    FIND_REFERENCES_NAME,
    GET_SYMBOL_NAME,
    FindReferencesTool,
    GetSymbolTool,
)

__all__ = [
    "FIND_REFERENCES_NAME",
    "GET_SYMBOL_NAME",
    "FindReferencesTool",
    "GetSymbolTool",
    "READ_FILE_NAME",
    "ReadFileTool",
    "SEARCH_CODE_NAME",
    "SEARCH_DOCUMENTATION_NAME",
    "SearchCodeTool",
    "SearchDocumentationTool",
    "Tool",
    "ToolResult",
]
