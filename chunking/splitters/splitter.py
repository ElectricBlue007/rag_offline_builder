"""Markdown chunking for RAG records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.contracts.chunking import ChunkArtifact, ChunkingResult
from core.contracts.enrichment import EnrichmentResult
from core.contracts.markdown import MarkdownGenerationResult


class ChunkSplitter:
    splitter_name = "chunking"

    def split(
        self,
        source_path: str,
        document_id: str,
        markdown: MarkdownGenerationResult,
        enrichment: EnrichmentResult | None = None,
    ) -> ChunkingResult:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")

        markdown.validate()
        if enrichment is not None:
            enrichment.validate()

        chunks = self._split_markdown(document_id, markdown)
        result = ChunkingResult(
            source_path=str(path),
            document_id=document_id,
            chunks=chunks,
            markdown=markdown,
            enrichment=enrichment,
            debug_info={"splitter": self.splitter_name},
        )
        result.validate()
        return result

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "splitter": self.splitter_name}

    def debug(self) -> dict[str, Any]:
        return {"splitter": self.splitter_name}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "splitter": self.splitter_name}

    def _split_markdown(self, document_id: str, markdown: MarkdownGenerationResult) -> list[ChunkArtifact]:
        chunks: list[ChunkArtifact] = []
        chunk_index = 0
        for section in markdown.sections:
            section_text = self._extract_section_text(markdown.markdown, section.title)
            if not section_text:
                continue
            for sentence in self._split_sentences(section_text):
                chunks.append(
                    ChunkArtifact(
                        chunk_id=f"{document_id}-chunk-{chunk_index}",
                        document_id=document_id,
                        text=sentence,
                        section_path=[section.title],
                        chunk_index=chunk_index,
                        metadata={"section_id": section.section_id},
                    )
                )
                chunk_index += 1
        if not chunks and markdown.markdown.strip():
            chunks.append(
                ChunkArtifact(
                    chunk_id=f"{document_id}-chunk-0",
                    document_id=document_id,
                    text=markdown.markdown.strip(),
                    section_path=[section.title for section in markdown.sections] if markdown.sections else [],
                    chunk_index=0,
                )
            )
        return chunks

    def _extract_section_text(self, markdown_text: str, section_title: str) -> str:
        pattern = re.compile(rf"^#+\s+{re.escape(section_title)}\s*$", re.MULTILINE)
        match = pattern.search(markdown_text)
        if match is None:
            return ""
        remainder = markdown_text[match.end() :]
        next_heading = re.search(r"^#+\s+", remainder, re.MULTILINE)
        if next_heading:
            remainder = remainder[: next_heading.start()]
        return remainder.strip()

    def _split_sentences(self, text: str) -> list[str]:
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
        return sentences or [text.strip()]


def chunk_markdown(
    source_path: str,
    document_id: str,
    markdown: MarkdownGenerationResult,
    enrichment: EnrichmentResult | None = None,
) -> ChunkingResult:
    return ChunkSplitter().split(source_path, document_id, markdown, enrichment)


def result_to_json(result: ChunkingResult) -> str:
    return json.dumps(
        {
            "source_path": result.source_path,
            "document_id": result.document_id,
            "chunks": [chunk.debug() for chunk in result.chunks],
            "debug_info": result.debug_info,
        },
        indent=2,
        sort_keys=True,
        default=str,
    )

