"""Vision and OCR contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class VisionRegion(ContractMixin):
    region_id: str
    region_type: Literal["ocr", "chart", "diagram", "figure", "table"]
    text: str | None = None
    confidence: float | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.region_id.strip()), "region_id is required")
        _require(bool(self.region_type), "region_type is required")
        if self.confidence is not None:
            _require(0.0 <= self.confidence <= 1.0, "confidence must be between 0 and 1")
        if self.page_number is not None:
            _require(self.page_number >= 0, "page_number must be non-negative")


@dataclass(slots=True)
class OCRResult(ContractMixin):
    source_path: str
    regions: list[VisionRegion]
    engine: str = "fallback"
    warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.source_path.strip()), "source_path is required")
        _require(bool(self.regions), "regions are required")
        for region in self.regions:
            region.validate()

