"""Vector database record contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.contracts.chunking import ChunkArtifact
from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class VectorRecord(ContractMixin):
    record_id: str
    document_id: str
    chunk_id: str
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.record_id.strip()), "record_id is required")
        _require(bool(self.document_id.strip()), "document_id is required")
        _require(bool(self.chunk_id.strip()), "chunk_id is required")
        _require(bool(self.text.strip()), "text is required")
        _require(bool(self.embedding), "embedding is required")


@dataclass(slots=True)
class VectorBatch(ContractMixin):
    source_path: str
    document_id: str
    records: list[VectorRecord]
    chunks: list[ChunkArtifact]
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.source_path.strip()), "source_path is required")
        _require(bool(self.document_id.strip()), "document_id is required")
        _require(bool(self.records), "records are required")
        for record in self.records:
            record.validate()
        for chunk in self.chunks:
            chunk.validate()

