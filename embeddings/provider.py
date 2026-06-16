"""Deterministic local embedding provider."""

from __future__ import annotations

import hashlib
from typing import Any

from core.contracts.embedding import EmbeddingRequest, EmbeddingResponse


class DeterministicEmbeddingProvider:
    provider_name = "deterministic"

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        request.validate()
        vector = self._embed_text(request.text, request.dimensions)
        response = EmbeddingResponse(
            request_id=request.request_id,
            vector=vector,
            model_name=self.provider_name,
            metadata={"dimensions": request.dimensions},
        )
        response.validate()
        return response

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "provider": self.provider_name}

    def debug(self) -> dict[str, Any]:
        return {"provider": self.provider_name}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "provider": self.provider_name}

    def _embed_text(self, text: str, dimensions: int) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index % len(digest)] / 255.0 for index in range(dimensions)]

