from __future__ import annotations

import unittest

from core.contracts.chunking import ChunkArtifact
from core.contracts.vector import VectorBatch, VectorRecord


class TestVectorContracts(unittest.TestCase):
    def test_vector_record_validation(self) -> None:
        record = VectorRecord(
            record_id="v1",
            document_id="d1",
            chunk_id="c1",
            text="Body",
            embedding=[0.1, 0.2, 0.3],
        )
        record.validate()
        self.assertEqual(record.debug()["record_id"], "v1")

    def test_vector_batch_validation(self) -> None:
        batch = VectorBatch(
            source_path="/tmp/sample.docx",
            document_id="d1",
            records=[
                VectorRecord(
                    record_id="v1",
                    document_id="d1",
                    chunk_id="c1",
                    text="Body",
                    embedding=[0.1, 0.2],
                )
            ],
            chunks=[ChunkArtifact(chunk_id="c1", document_id="d1", text="Body")],
        )
        batch.validate()


if __name__ == "__main__":
    unittest.main()

