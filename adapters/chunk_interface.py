"""Chunk loading interface for finalized chunks.json artifacts."""

from __future__ import annotations

import json
from pathlib import Path


class ChunkInterface:
    def load_chunks(self, path: str | Path) -> list[dict]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"chunks.json not found: {source}")
        data = json.loads(source.read_text(encoding="utf-8"))
        chunks = data.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("Chunk artifact must contain a chunks list.")
        return chunks
