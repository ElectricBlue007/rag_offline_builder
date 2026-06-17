"""Advanced JSON index exporter."""

from __future__ import annotations

import json
from pathlib import Path

from rag_doc_compiler import __version__
from rag_doc_compiler.models import ChunkCandidate, StructureAnalysis


class MetadataExporter:
    def build_index(self, analysis: StructureAnalysis, chunks: list[ChunkCandidate]) -> dict:
        return {
            "compiler": {"name": "rag-document-intelligence-compiler", "version": __version__},
            "outputs": {"markdown_preserved": True, "final_chunking_decision": False},
            "flat_chunk_view": [chunk.to_dict() for chunk in chunks],
            "hierarchical_structure_view": analysis.to_dict(),
            "observability": {
                "chunk_count": len(chunks),
                "section_count": sum(1 for node in analysis.flat_map.values() if node.node_type == "section"),
                "block_count": len(analysis.ordered_blocks),
            },
        }

    def write(self, index: dict, output_dir: str | Path, filename: str = "index.json") -> Path:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return path
