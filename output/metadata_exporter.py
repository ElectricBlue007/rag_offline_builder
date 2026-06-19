"""Advanced JSON index exporter."""

from __future__ import annotations

import json
from pathlib import Path

from rag_doc_compiler import __version__
from rag_doc_compiler.models import ChunkCandidate, ChunkingResult, StructureAnalysis


class MetadataExporter:
    def build_index(
        self,
        analysis: StructureAnalysis,
        chunks: list[ChunkCandidate],
        final_chunks: ChunkingResult | None = None,
    ) -> dict:
        final_chunk_dict = final_chunks.to_dict() if final_chunks else None
        return {
            "compiler": {"name": "rag-document-intelligence-compiler", "version": __version__},
            "outputs": {
                "markdown_preserved": True,
                "candidate_chunking_available": True,
                "final_chunking_decision": final_chunks is not None,
            },
            "flat_chunk_view": [chunk.to_dict() for chunk in chunks],
            "final_chunk_view": final_chunk_dict["chunks"] if final_chunk_dict else [],
            "chunking_strategy": final_chunk_dict["strategy"] if final_chunk_dict else None,
            "hierarchical_structure_view": analysis.to_dict(),
            "observability": {
                "candidate_chunk_count": len(chunks),
                "final_chunk_count": len(final_chunk_dict["chunks"]) if final_chunk_dict else 0,
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
