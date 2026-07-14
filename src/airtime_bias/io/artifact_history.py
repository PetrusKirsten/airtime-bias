from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def list_artifacts(
    directory: str | Path,
    pattern: str,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Return saved artifact files ordered from newest to oldest."""
    directory = Path(directory)
    if not directory.exists():
        return []

    iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
    paths = [path for path in iterator if path.is_file()]
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def artifact_label(path: str | Path, root: str | Path | None = None) -> str:
    """Create a compact, human-readable label for a saved artifact."""
    path = Path(path)
    display_path = path

    if root is not None:
        try:
            display_path = path.relative_to(Path(root))
        except ValueError:
            display_path = path

    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    size_mb = stat.st_size / (1024**2)
    return f"{display_path} · {modified} · {size_mb:.2f} MB"


def preferred_artifact_index(paths: list[Path], preferred: str | Path | None) -> int:
    """Return the index of a preferred artifact, falling back to the newest file."""
    if not paths or preferred is None:
        return 0

    preferred_path = Path(preferred)
    for index, path in enumerate(paths):
        if path == preferred_path:
            return index
    return 0


def parameter_fingerprint(parameters: dict[str, Any], length: int = 10) -> str:
    """Create a short deterministic identifier for one parameter configuration."""
    payload = json.dumps(
        parameters,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Persist run metadata beside generated tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = dict(payload)
    manifest.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, default=str)
    temporary_path.replace(path)
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a saved run manifest, returning an empty dictionary when unavailable."""
    path = Path(path)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}
