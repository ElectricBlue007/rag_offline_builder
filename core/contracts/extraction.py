"""Extraction contracts for source-specific modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.contracts.document import DocumentMetadata
from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class SourceDocumentRequest(ContractMixin):
    file_path: str
    source_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.file_path.strip()), "file_path is required")


@dataclass(slots=True)
class MetadataExtractionResult(ContractMixin):
    request: SourceDocumentRequest
    metadata: DocumentMetadata
    extractor_name: str = "metadata"
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.request.validate()
        self.metadata.validate()
        _require(bool(self.extractor_name.strip()), "extractor_name is required")

