"""Final chunk artifact exporter."""

from __future__ import annotations

import json
from pathlib import Path

from rag_doc_compiler.models import ChunkingResult


class ChunkExporter:
    def write(self, chunking: ChunkingResult, output_dir: str | Path, filename: str = "chunks.json") -> Path:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(json.dumps(chunking.to_dict(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return path
