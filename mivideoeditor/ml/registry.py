"""Simple standardized model registry on filesystem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path("models/registry.json")


@dataclass
class ModelRecord:
    """Model record."""

    name: str
    version: str
    path: Path
    metadata: dict[str, Any]


def _load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    if not path.exists():
        return {"models": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(data: dict[str, Any], path: Path = DEFAULT_REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def register_model(
    name: str, version: str, model_dir: Path, metadata: dict[str, Any] | None = None
) -> None:
    """Register a model."""
    reg = _load_registry()
    entry = {
        "name": name,
        "version": version,
        "path": str(model_dir.resolve()),
        "metadata": metadata or {},
        "registered_at": datetime.now(UTC).isoformat(),
    }
    # replace if exists
    reg["models"] = [
        m for m in reg["models"] if not (m["name"] == name and m["version"] == version)
    ]
    reg["models"].append(entry)
    _save_registry(reg)


def resolve_model(name: str, version: str | None = None) -> Path | None:
    """Resolve a model by name and version."""
    reg = _load_registry()
    candidates = [m for m in reg.get("models", []) if m.get("name") == name]
    if not candidates:
        return None
    if version is None:
        # pick most recent by registered_at
        candidates.sort(key=lambda m: m.get("registered_at", ""), reverse=True)
        return Path(candidates[0]["path"]) if candidates else None
    for m in candidates:
        if m.get("version") == version:
            return Path(m["path"])
    return None


def list_models() -> list[ModelRecord]:
    """List all models."""
    reg = _load_registry()
    return [
        ModelRecord(
            name=m.get("name", "unknown"),
            version=m.get("version", "unknown"),
            path=Path(m.get("path", ".")),
            metadata=m.get("metadata", {}),
        )
        for m in reg.get("models", [])
    ]
