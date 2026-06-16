"""Chunking contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin

if TYPE_CHECKING:
    from core.contracts.enrichment import EnrichmentResult
    from core.contracts.markdown import MarkdownGenerationResult


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class ChunkArtifact(ContractMixin):
    chunk_id: str
    document_id: str
    text: str
    section_path: list[str] = field(default_factory=list)
    parent_chunk_id: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.chunk_id.strip()), "chunk_id is required")
        _require(bool(self.document_id.strip()), "document_id is required")
        _require(bool(self.text.strip()), "text is required")
        _require(self.chunk_index >= 0, "chunk_index must be non-negative")


@dataclass(slots=True)
class ChunkingResult(ContractMixin):
    source_path: str
    document_id: str
    chunks: list[ChunkArtifact]
    # Both fields are optional: StreamAwareChunker operates on DocumentStream, not Markdown
    markdown: MarkdownGenerationResult | None = None
    enrichment: EnrichmentResult | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.source_path.strip()), "source_path is required")
        _require(bool(self.document_id.strip()), "document_id is required")
        _require(bool(self.chunks), "chunks are required")
        for chunk in self.chunks:
            chunk.validate()