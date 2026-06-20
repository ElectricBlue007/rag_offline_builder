"""Composable compiler orchestration."""

from __future__ import annotations

from pathlib import Path

from adapters.file_loader import FileLoader
from core.chunk_preparer import ChunkPreparer
from core.markdown_normalizer import MarkdownNormalizer
from core.structure_analyzer import StructureAnalyzer
from embedding.chunk_embedder import ChunkEmbedder
from embedding.model_loader import EmbeddingModelLoader
from output.markdown_writer import MarkdownWriter
from output.metadata_exporter import MetadataExporter


class DocumentCompiler:
    def __init__(self, enable_embeddings: bool = False) -> None:
        self.loader = FileLoader()
        self.normalizer = MarkdownNormalizer()
        self.analyzer = StructureAnalyzer()
        self.chunk_preparer = ChunkPreparer()
        self.markdown_writer = MarkdownWriter()
        self.metadata_exporter = MetadataExporter()
        self.enable_embeddings = enable_embeddings
        self._embedder: ChunkEmbedder | None = None
    
    @property
    def embedder(self) -> ChunkEmbedder:
        """Lazy-initialize the chunk embedder."""
        if self._embedder is None:
            model_loader = EmbeddingModelLoader()
            self._embedder = ChunkEmbedder(model_loader=model_loader)
        return self._embedder

    def compile_text(self, markdown: str) -> dict:
        normalized = self.normalizer.normalize(markdown)
        analysis = self.analyzer.analyze(normalized)
        chunks = self.chunk_preparer.prepare(analysis, normalized.flags_by_line)
        index = self.metadata_exporter.build_index(analysis, chunks)
        
        result = {
            "markdown": normalized.markdown,
            "analysis": analysis,
            "chunks": chunks,
            "index": index,
        }
        
        # Add embeddings if enabled
        if self.enable_embeddings:
            embedded_chunks = self.embedder.embed_chunks(chunks)
            result["embedded_chunks"] = embedded_chunks
        
        return result

    def run_file(self, input_path: str | Path, output_dir: str | Path) -> dict:
        result = self.compile_text(self.loader.load_text(input_path))
        markdown_path = self.markdown_writer.write(result["markdown"], output_dir)
        index_path = self.metadata_exporter.write(result["index"], output_dir)
        return {**result, "markdown_path": markdown_path, "index_path": index_path}

    def validate_file(self, input_path: str | Path) -> dict:
        result = self.compile_text(self.loader.load_text(input_path))
        return {"valid": True, "chunk_count": len(result["chunks"])}
    
    def download_embedding_model(self, force: bool = False) -> Path:
        """Download the embedding model for offline use.
        
        Args:
            force: If True, force redownload even if model exists.
            
        Returns:
            Path to the cached model directory.
        """
        model_loader = EmbeddingModelLoader()
        return model_loader.download_model(force=force)
