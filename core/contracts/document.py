"""Document-level contracts for extraction and downstream processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class DocumentMetadata(ContractMixin):
    file_name: str
    mime_type: str
    source_hash: str
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.file_name.strip()), "file_name is required")
        _require(bool(self.mime_type.strip()), "mime_type is required")
        _require(bool(self.source_hash.strip()), "source_hash is required")


@dataclass(slots=True)
class DocumentSection(ContractMixin):
    section_id: str
    title: str
    level: int
    parent_id: str | None = None
    source_page: int | None = None
    path: list[str] = field(default_factory=list)

    def validate(self) -> None:
        _require(bool(self.section_id.strip()), "section_id is required")
        _require(bool(self.title.strip()), "title is required")
        _require(self.level >= 0, "level must be non-negative")


@dataclass(slots=True)
class ContentBlock(ContractMixin):
    block_id: str
    block_type: Literal["paragraph", "heading", "list", "table", "image", "equation", "footnote", "comment", "reference"]
    text: str | None = None
    page_number: int | None = None
    section_id: str | None = None
    source_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.block_id.strip()), "block_id is required")
        _require(bool(self.block_type), "block_type is required")
        if self.page_number is not None:
            _require(self.page_number >= 0, "page_number must be non-negative")


@dataclass(slots=True)
class TableCell(ContractMixin):
    row_index: int
    column_index: int
    text: str
    rowspan: int = 1
    colspan: int = 1

    def validate(self) -> None:
        _require(self.row_index >= 0, "row_index must be non-negative")
        _require(self.column_index >= 0, "column_index must be non-negative")
        _require(self.rowspan >= 1, "rowspan must be at least 1")
        _require(self.colspan >= 1, "colspan must be at least 1")


@dataclass(slots=True)
class TableArtifact(ContractMixin):
    table_id: str
    cells: list[TableCell]
    caption: str | None = None
    page_number: int | None = None
    source_block_id: str | None = None

    def validate(self) -> None:
        _require(bool(self.table_id.strip()), "table_id is required")
        _require(bool(self.cells), "cells are required")
        for cell in self.cells:
            cell.validate()


@dataclass(slots=True)
class MarkdownDocument(ContractMixin):
    document_id: str
    markdown: str
    section_map: list[DocumentSection] = field(default_factory=list)
    source_metadata: DocumentMetadata | None = None

    def validate(self) -> None:
        _require(bool(self.document_id.strip()), "document_id is required")
        _require(bool(self.markdown.strip()), "markdown is required")
        for section in self.section_map:
            section.validate()
        if self.source_metadata is not None:
            self.source_metadata.validate()


@dataclass(slots=True)
class ChunkRecord(ContractMixin):
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
class ImageArtifact(ContractMixin):
    """Represents an image extracted from a document with optional vision description."""
    id: str                     # Unique identifier (e.g., 'image1' or file path)
    alt_text: str = ""          # Alternative text from document if available
    caption: str = ""           # Caption text if available
    description: str = ""       # Vision model generated description (OCR + scene understanding)
    bounding_box: tuple[int, int, int, int] | None = None  # (x, y, width, height)
    ocr_text: str = ""          # Any text extracted from the image (by OCR or fallback)
    source_anchors: list[str] = field(default_factory=list)  # e.g., file name or page reference
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.id.strip()), "id is required")
        # description can be empty, but we allow it
        if self.page_number is not None:
            _require(self.page_number >= 0, "page_number must be non-negative") 