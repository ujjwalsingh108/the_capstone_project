from __future__ import annotations

from pathlib import Path


def collect_text_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob("*") if path.is_file())
