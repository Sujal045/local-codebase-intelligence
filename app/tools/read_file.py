"""``read_file`` tool (Slice 5B).

``search_code`` returns short ranked chunks. Agents often need surrounding
code or a full function body. This tool opens a path from the repository
root and returns numbered lines.

    LLM decides: read_file(path="src/scoring.py", start_line=1, end_line=20)
            │
            ▼
    resolve path under root (sandbox)
            │
            ▼
    observation with numbered lines

Paths must be relative (as returned by ``search_code``). Absolute paths
and ``../`` escapes become error observations — the agent should recover,
not crash.

This module does not talk to Qdrant or the LLM. Version 6 will call
``run()`` when the model requests ``read_file``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.base import ToolResult
from app.tools.paths import resolve_inside_root

READ_FILE_NAME = "read_file"
DEFAULT_MAX_READ_BYTES = 200_000
DEFAULT_MAX_READ_LINES = 250


class ReadFileTool:
    """Read a text file under a fixed repository root."""

    name = READ_FILE_NAME

    def __init__(
        self,
        *,
        root: str | Path,
        max_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_lines: int = DEFAULT_MAX_READ_LINES,
    ) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        if max_lines < 1:
            raise ValueError(f"max_lines must be >= 1, got {max_lines}")
        self._root = Path(root).resolve()
        self._max_bytes = max_bytes
        self._max_lines = max_lines

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Read a source file from the repository. Use after "
                    "search_code when you need more lines around a hit. "
                    "Paths must be relative to the repo root "
                    "(e.g. 'src/scoring.py'). Optional start_line and "
                    "end_line are 1-based inclusive."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Relative path from the repository root, "
                                "as returned by search_code."
                            ),
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "First line to include (1-based).",
                            "minimum": 1,
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "Last line to include (1-based).",
                            "minimum": 1,
                        },
                    },
                    "required": ["path"],
                },
            },
        }

    def run(self, **arguments: Any) -> ToolResult:
        """Read file text. Bad arguments / sandbox violations → error text."""
        extra = sorted(set(arguments) - {"path", "start_line", "end_line"})
        if extra:
            return ToolResult(
                name=self.name,
                content=f"error: unexpected arguments {extra}",
            )

        path_arg = arguments.get("path")
        try:
            target = resolve_inside_root(self._root, path_arg)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            return ToolResult(name=self.name, content=f"error: {exc}")

        start = arguments.get("start_line")
        end = arguments.get("end_line")
        for label, value in (("start_line", start), ("end_line", end)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                return ToolResult(
                    name=self.name,
                    content=f"error: {label} must be an integer >= 1",
                )

        if start is not None and end is not None and start > end:
            return ToolResult(
                name=self.name,
                content="error: start_line must be <= end_line",
            )

        if not target.exists():
            return ToolResult(
                name=self.name,
                content=f"error: file not found: {path_arg.strip()}",
            )
        if not target.is_file():
            return ToolResult(
                name=self.name,
                content=f"error: not a file: {path_arg.strip()}",
            )

        try:
            size = target.stat().st_size
        except OSError as exc:
            return ToolResult(name=self.name, content=f"error: {exc}")
        if size > self._max_bytes:
            return ToolResult(
                name=self.name,
                content=(
                    f"error: file is {size} bytes; max allowed is "
                    f"{self._max_bytes}. Pass a tighter start_line/end_line."
                ),
            )

        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                name=self.name,
                content=f"error: file is not valid UTF-8 text: {path_arg.strip()}",
            )
        except OSError as exc:
            return ToolResult(name=self.name, content=f"error: {exc}")

        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            relative = path_arg.strip()
            return ToolResult(
                name=self.name,
                content=f"{relative}:1-0 (0 lines in file)\n(empty file)",
            )

        first = 1 if start is None else start
        last = total if end is None else end
        if first > total:
            return ToolResult(
                name=self.name,
                content=(
                    f"error: start_line {first} is past end of file "
                    f"({total} lines)"
                ),
            )
        last = min(last, total)

        span = last - first + 1
        notes: list[str] = []
        if span > self._max_lines:
            last = first + self._max_lines - 1
            notes.append(
                f"truncated to {self._max_lines} lines; pass a smaller range "
                f"to read more"
            )

        slice_lines = lines[first - 1 : last]
        relative = path_arg.strip()
        content = _format_numbered(relative, slice_lines, first=first, total=total)
        if notes:
            content = content + "\n(" + "; ".join(notes) + ")"
        return ToolResult(name=self.name, content=content)


def _format_numbered(
    path: str,
    lines: list[str],
    *,
    first: int,
    total: int,
) -> str:
    if not lines:
        return f"{path}:{first}-{first - 1} ({total} lines in file)\n(empty range)"
    last = first + len(lines) - 1
    width = len(str(last))
    body = "\n".join(f"{number:>{width}}| {line}" for number, line in enumerate(lines, start=first))
    return f"{path}:{first}-{last} ({total} lines in file)\n{body}"
