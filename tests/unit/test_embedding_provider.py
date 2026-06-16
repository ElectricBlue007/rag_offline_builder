from __future__ import annotations

import unittest

from core.contracts.embedding import EmbeddingRequest
from embeddings.provider import DeterministicEmbeddingProvider


class TestEmbeddingProvider(unittest.TestCase):
    def test_embed_text(self) -> None:
        provider = DeterministicEmbeddingProvider()
        response = provider.embed(EmbeddingRequest(request_id="r1", text="hello world", dimensions=4))

        self.assertEqual(response.model_name, "deterministic")
        self.assertEqual(len(response.vector), 4)

    def test_validate_and_health_check(self) -> None:
        provider = DeterministicEmbeddingProvider()
        self.assertEqual(provider.validate()["status"], "ok")
        self.assertEqual(provider.health_check()["provider"], "deterministic")


if __name__ == "__main__":
    unittest.main()

