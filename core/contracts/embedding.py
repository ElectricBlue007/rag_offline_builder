"""Embedding contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.exceptions.base import ContractValidationError
from core.interfaces.base import ContractMixin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


@dataclass(slots=True)
class EmbeddingRequest(ContractMixin):
    request_id: str
    text: str
    dimensions: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.request_id.strip()), "request_id is required")
        _require(bool(self.text.strip()), "text is required")
        _require(self.dimensions > 0, "dimensions must be positive")


@dataclass(slots=True)
class EmbeddingResponse(ContractMixin):
    request_id: str
    vector: list[float]
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require(bool(self.request_id.strip()), "request_id is required")
        _require(bool(self.vector), "vector is required")
        _require(bool(self.model_name.strip()), "model_name is required")

