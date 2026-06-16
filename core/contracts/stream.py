"""Unified document stream contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin
from core.contracts.document import ImageArtifact
from core.contracts.tables import TableCellData


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class OrderedDocumentBlock(ContractMixin):
    """A single block in the unified document stream."""
    block_id: str
    block_type: Literal["heading", "paragraph", "table", "image"]
    
    # Payload fields (only one should be populated based on block_type)
    text: str | None = None
    level: int | None = None  # For headings
    table_cells: list[TableCellData] | None = None  # For tables
    image: ImageArtifact | None = None  # For images
    
    page_number: int | None = None
    order_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.block_id.strip()), "block_id is required")
        _require(bool(self.block_type), "block_type is required")
        _require(self.order_index >= 0, "order_index must be non-negative")
        
        if self.block_type in ("heading", "paragraph"):
            _require(bool(self.text and self.text.strip()), f"text is required for {self.block_type}")
        elif self.block_type == "table":
            _require(self.table_cells is not None and bool(self.table_cells), "table_cells are required for table")
        elif self.block_type == "image":
            _require(self.image is not None, "image artifact is required for image block")


@dataclass(slots=True)
class DocumentStream(ContractMixin):
    """The unified, strictly ordered stream of all document blocks."""
    source_path: str
    blocks: list[OrderedDocumentBlock]
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.source_path.strip()), "source_path is required")
        _require(bool(self.blocks), "blocks are required")
        for block in self.blocks:
            block.validate()