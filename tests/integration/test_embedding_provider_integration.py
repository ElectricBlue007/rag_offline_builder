from __future__ import annotations

import unittest

from core.contracts.embedding import EmbeddingRequest
from embeddings.provider import DeterministicEmbeddingProvider


class TestEmbeddingProviderIntegration(unittest.TestCase):
    def test_pipeline_level_embedding_boundary(self) -> None:
        response = DeterministicEmbeddingProvider().embed(EmbeddingRequest(request_id="r1", text="sample text"))

        self.assertEqual(response.request_id, "r1")
        self.assertTrue(all(0.0 <= value <= 1.0 for value in response.vector))


if __name__ == "__main__":
    unittest.main()

