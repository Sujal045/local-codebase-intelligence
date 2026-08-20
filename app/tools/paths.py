"""Resolve user-supplied paths inside a repository root (Slice 5B).

``search_code`` returns relative paths such as ``src/scoring.py``. The
``read_file`` tool must open those paths without letting the model escape
the repo (classic ``../`` / absolute-path attacks).

    root = /home/me/project
    "src/a.py"           → /home/me/project/src/a.py   OK
    "../secrets.txt"     → rejected (outside root)
    "/etc/passwd"        → rejected (absolute)
"""

from __future__ import annotations

from pathlib import Path


def resolve_inside_root(root: str | Path, user_path: str) -> Path:
    """Return an absolute path that is strictly under ``root``.

    Raises:
        ValueError: empty path, absolute path, or path that escapes ``root``.
    """
    if not isinstance(user_path, str) or not user_path.strip():
        raise ValueError("path must be a non-empty string")

    raw = user_path.strip()
    if Path(raw).is_absolute():
        raise ValueError("path must be relative to the repository root")

    root_resolved = Path(root).resolve()
    if not root_resolved.is_dir():
        raise ValueError(f"repository root is not a directory: {root_resolved}")

    resolved = (root_resolved / raw).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("path escapes the repository root") from exc
    return resolved
