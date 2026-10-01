"""Tiny JSON-on-disk cache so repeated retrieval / LLM calls cost nothing (cost control)."""

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, directory: Path):
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None  # corrupt entry: treat as a miss

    def set(self, key: str, value: Any) -> None:
        (self._dir / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")
