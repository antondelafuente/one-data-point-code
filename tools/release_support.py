"""Small filesystem helpers for release preparation; no service dependencies."""
import os
import tempfile
from pathlib import Path


def atomic_write(path, content):
    """Replace one file only after its complete contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            name = f.name
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def release_path(root, relative):
    """Resolve a manifest path without allowing it to escape the release tree."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path is outside the release: {relative}")
    return path
