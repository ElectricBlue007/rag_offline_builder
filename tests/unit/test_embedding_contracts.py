from __future__ import annotations

import unittest

from core.contracts.embedding import EmbeddingRequest, EmbeddingResponse


class TestEmbeddingContracts(unittest.TestCase):
    def test_embedding_request_validation(self) -> None:
        request = EmbeddingRequest(request_id="r1", text="Body text")
        request.validate()
        self.assertEqual(request.debug()["request_id"], "r1")

    def test_embedding_response_validation(self) -> None:
        response = EmbeddingResponse(request_id="r1", vector=[0.1, 0.2], model_name="pseudo")
        response.validate()


if __name__ == "__main__":
    unittest.main()

