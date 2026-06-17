"""Composable compiler orchestration."""

from __future__ import annotations

from pathlib import Path

from adapters.file_loader import FileLoader
from core.chunk_preparer import ChunkPreparer
from core.markdown_normalizer import MarkdownNormalizer
from core.structure_analyzer import StructureAnalyzer
from output.markdown_writer import MarkdownWriter
from output.metadata_exporter import MetadataExporter


class DocumentCompiler:
    def __init__(self) -> None:
        self.loader = FileLoader()
        self.normalizer = MarkdownNormalizer()
        self.analyzer = StructureAnalyzer()
        self.chunk_preparer = ChunkPreparer()
        self.markdown_writer = MarkdownWriter()
        self.metadata_exporter = MetadataExporter()

    def compile_text(self, markdown: str) -> dict:
        normalized = self.normalizer.normalize(markdown)
        analysis = self.analyzer.analyze(normalized)
        chunks = self.chunk_preparer.prepare(analysis, normalized.flags_by_line)
        index = self.metadata_exporter.build_index(analysis, chunks)
        return {"markdown": normalized.markdown, "analysis": analysis, "chunks": chunks, "index": index}

    def run_file(self, input_path: str | Path, output_dir: str | Path) -> dict:
        result = self.compile_text(self.loader.load_text(input_path))
        markdown_path = self.markdown_writer.write(result["markdown"], output_dir)
        index_path = self.metadata_exporter.write(result["index"], output_dir)
        return {**result, "markdown_path": markdown_path, "index_path": index_path}

    def validate_file(self, input_path: str | Path) -> dict:
        result = self.compile_text(self.loader.load_text(input_path))
        return {"valid": True, "chunk_count": len(result["chunks"])}
