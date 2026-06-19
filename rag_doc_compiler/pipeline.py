"""Composable compiler orchestration."""

from __future__ import annotations

from pathlib import Path

from adapters.file_loader import FileLoader
from core.chunk_preparer import ChunkPreparer
from core.chunking_strategy import HierarchyAwareChunkingStrategy
from core.markdown_normalizer import MarkdownNormalizer
from core.structure_analyzer import StructureAnalyzer
from output.chunk_exporter import ChunkExporter
from output.markdown_writer import MarkdownWriter
from output.metadata_exporter import MetadataExporter


class DocumentCompiler:
    def __init__(self) -> None:
        self.loader = FileLoader()
        self.normalizer = MarkdownNormalizer()
        self.analyzer = StructureAnalyzer()
        self.chunk_preparer = ChunkPreparer()
        self.chunking_strategy = HierarchyAwareChunkingStrategy()
        self.markdown_writer = MarkdownWriter()
        self.chunk_exporter = ChunkExporter()
        self.metadata_exporter = MetadataExporter()

    def compile_text(self, markdown: str) -> dict:
        normalized = self.normalizer.normalize(markdown)
        analysis = self.analyzer.analyze(normalized)
        chunk_candidates = self.chunk_preparer.prepare(analysis, normalized.flags_by_line)
        final_chunks = self.chunking_strategy.chunk(analysis)
        index = self.metadata_exporter.build_index(analysis, chunk_candidates, final_chunks)
        return {
            "markdown": normalized.markdown,
            "analysis": analysis,
            "chunks": chunk_candidates,
            "final_chunks": final_chunks,
            "index": index,
        }

    def run_file(self, input_path: str | Path, output_dir: str | Path) -> dict:
        result = self.compile_text(self.loader.load_text(input_path))
        markdown_path = self.markdown_writer.write(result["markdown"], output_dir)
        chunks_path = self.chunk_exporter.write(result["final_chunks"], output_dir)
        index_path = self.metadata_exporter.write(result["index"], output_dir)
        return {**result, "markdown_path": markdown_path, "chunks_path": chunks_path, "index_path": index_path}

    def validate_file(self, input_path: str | Path) -> dict:
        result = self.compile_text(self.loader.load_text(input_path))
        return {
            "valid": True,
            "candidate_chunk_count": len(result["chunks"]),
            "final_chunk_count": len(result["final_chunks"].chunks),
        }
