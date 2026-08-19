"""Code intelligence tools (Version 5).

Slice 5A adds ``search_code``: the same hybrid/rerank retrieval as ``ask()``,
exposed as a named function with a JSON schema.

The LLM does not call these yet. Version 6 will send ``spec()`` to Ollama
and run ``tool.run(**arguments)`` when the model requests a tool.
"""

from __future__ import annotations

from app.tools.base import Tool, ToolResult
from app.tools.search_code import SEARCH_CODE_NAME, SearchCodeTool

__all__ = [
    "SEARCH_CODE_NAME",
    "SearchCodeTool",
    "Tool",
    "ToolResult",
]
