"""End-to-end orchestration from extraction to vector records."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from chunking.splitters.stream_aware_splitter import StreamAwareChunker
from core.contracts.embedding import EmbeddingRequest
from core.contracts.extraction import SourceDocumentRequest
from core.contracts.vector import VectorBatch, VectorRecord
from core.logging import configure_logging, log_event
from embeddings.provider import DeterministicEmbeddingProvider
from enrichment.processor import EnrichmentProcessor
from extractors.content.extractor import ContentExtractor
from extractors.images.extractor import ImageExtractor
from extractors.metadata.extractor import MetadataExtractor
from extractors.tables.extractor import TableExtractor
from layout.reading_order.extractor import ReadingOrderExtractor
from layout.stream_merger.merger import StreamMerger
from markdown.generators.generator import RAGMarkdownGenerator

class PipelineOrchestrator:
    orchestrator_name = "pipeline"

    def run(self, file_path: str) -> VectorBatch:
        logger = configure_logging()
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")

        log_event(logger, 20, "pipeline_start", "pipeline execution started", file_path=str(path))

        # 1. Extract raw artifacts
        metadata_result = MetadataExtractor().extract(SourceDocumentRequest(file_path=str(path)))
        content = ContentExtractor().extract(str(path))
        tables = TableExtractor().extract(str(path)) if path.suffix.lower() == ".docx" else None
        reading_order = ReadingOrderExtractor().extract(str(path)) if path.suffix.lower() == ".docx" else None

        images = None
        if path.suffix.lower() == ".docx":
            try:
                images = ImageExtractor().extract_images(str(path))
            except Exception as e:
                logger.warning(f"Image extraction failed: {e}")

        # 2. Merge into Unified Document Stream
        stream = StreamMerger().merge(
            source_path=str(path),
            reading_order=reading_order,
            content=content,
            tables=tables,
            images=images,
        )

        # 3. Generate RAG-Optimized Markdown from Stream (Pass metadata for frontmatter)
        metadata_dict = asdict(metadata_result.metadata)
        markdown_result = RAGMarkdownGenerator().generate(stream, metadata=metadata_dict)

        # 4. Enrich
        enrichment = EnrichmentProcessor().enrich(str(path), markdown_result)

        # 5. Stream-Aware Chunking
        chunks = StreamAwareChunker().chunk_stream(stream, metadata_result.metadata.source_hash)

        # 6. Embed and assemble batch
        records = [self._chunk_to_vector_record(chunk) for chunk in chunks.chunks]

        batch = VectorBatch(
            source_path=str(path),
            document_id=metadata_result.metadata.source_hash,
            records=records,
            chunks=chunks.chunks,
            debug_info={
                "orchestrator": self.orchestrator_name,
                "stream_blocks": len(stream.blocks),
                "total_chunks": len(chunks.chunks),
            },
        )

        batch.validate()
        log_event(
            logger, 20, "pipeline_end", "pipeline execution completed",
            file_path=str(path), chunks=len(batch.chunks),
        )
        return batch

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "orchestrator": self.orchestrator_name}

    def debug(self) -> dict[str, Any]:
        return {"orchestrator": self.orchestrator_name}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "orchestrator": self.orchestrator_name}

    def _chunk_to_vector_record(self, chunk: Any) -> VectorRecord:
        response = DeterministicEmbeddingProvider().embed(
            EmbeddingRequest(request_id=chunk.chunk_id, text=chunk.text, dimensions=8)
        )
        return VectorRecord(
            record_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            embedding=response.vector,
            metadata={
                "section_path": chunk.section_path,
                "chunk_index": chunk.chunk_index,
                "embedding_model": response.model_name,
            },
        )

def run_pipeline(file_path: str) -> VectorBatch:
    return PipelineOrchestrator().run(file_path)

def result_to_json(result: VectorBatch) -> str:
    return json.dumps(
        {
            "source_path": result.source_path,
            "document_id": result.document_id,
            "records": [record.debug() for record in result.records],
            "chunks": [chunk.debug() for chunk in result.chunks],
            "debug_info": result.debug_info,
        },
        indent=2,
        sort_keys=True,
        default=str,
    )