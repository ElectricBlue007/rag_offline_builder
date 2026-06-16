"""Contracts for image descriptions and OCR outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class VisionDescriptionRequest(ContractMixin):
    request_id: str
    image_path: str
    prompt: str

    def validate(self) -> None:
        _require(bool(self.request_id.strip()), "request_id is required")
        _require(bool(self.image_path.strip()), "image_path is required")
        _require(bool(self.prompt.strip()), "prompt is required")


@dataclass(slots=True)
class VisionDescriptionResponse(ContractMixin):
    request_id: str
    description: str
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.request_id.strip()), "request_id is required")
        _require(bool(self.description.strip()), "description is required")
        _require(bool(self.model_name.strip()), "model_name is required")

