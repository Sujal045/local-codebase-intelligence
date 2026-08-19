"""Code intelligence tools (Version 5).

Slice 5A: ``search_code`` — hybrid/rerank retrieval as a named function.
Slice 5B: ``read_file`` — sandboxed file read with optional line range.

The LLM does not call these yet. Version 6 will send ``spec()`` to Ollama
and run ``tool.run(**arguments)`` when the model requests a tool.
"""

from __future__ import annotations

from app.tools.base import Tool, ToolResult
from app.tools.read_file import READ_FILE_NAME, ReadFileTool
from app.tools.search_code import SEARCH_CODE_NAME, SearchCodeTool

__all__ = [
    "READ_FILE_NAME",
    "ReadFileTool",
    "SEARCH_CODE_NAME",
    "SearchCodeTool",
    "Tool",
    "ToolResult",
]
