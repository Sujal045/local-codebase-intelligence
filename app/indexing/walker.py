"""Walk a local repository and collect text files to index (Slice 1e).

The walker is the first step of indexing:

    repo root → filter files → read UTF-8 text → SourceFile(path, content)

It does not chunk, embed, or talk to Qdrant. Those happen later in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Directories we never descend into (noise, deps, VCS, build artifacts).
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".cursor",
    ".vscode",
    ".idea",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
}

# Version 1 indexes common source / docs; not lockfiles or binaries.
INDEX_SUFFIXES = {
    ".py",
    ".pyi",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".swift",
    ".md",
    ".rst",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
}

SKIP_FILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "Cargo.lock",
}

DEFAULT_MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class SourceFile:
    """One file selected for indexing."""

    path: str
    content: str


def walk_repository(
    root: str | Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[SourceFile]:
    """Return indexable files under ``root``, paths relative to the repo.

    Skips known junk directories, non-text / oversized files, and names
    that are almost never useful as code context (lockfiles).
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    files: list[SourceFile] = []
    for dirpath, dirnames, filenames in os_walk_filtered(root_path):
        current = Path(dirpath)
        for name in filenames:
            if name in SKIP_FILE_NAMES:
                continue
            file_path = current / name
            if file_path.suffix.lower() not in INDEX_SUFFIXES:
                continue
            source = _read_source_file(root_path, file_path, max_file_bytes=max_file_bytes)
            if source is not None:
                files.append(source)

    files.sort(key=lambda item: item.path)
    return files


def os_walk_filtered(root_path: Path):
    """``os.walk`` that prunes ``SKIP_DIR_NAMES`` in-place."""
    import os

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        yield dirpath, dirnames, filenames


def _read_source_file(
    root_path: Path,
    file_path: Path,
    *,
    max_file_bytes: int,
) -> SourceFile | None:
    try:
        size = file_path.stat().st_size
    except OSError:
        return None
    if size > max_file_bytes:
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None

    relative = file_path.relative_to(root_path).as_posix()
    return SourceFile(path=relative, content=content)
