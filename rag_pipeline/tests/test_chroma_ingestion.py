"""
tests/test_chroma_ingestion.py

Unit tests for ChromaIngestor — using a mocked ChromaDB client, no live
service/install required. Special focus on the metadata sanitization
that prevents real ingestion crashes (list-valued fields like
page_numbers, which ChromaDB does not accept as-is).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config import ChromaIngestionConfig
from rag_pipeline.storage.chroma_ingestion import (
    ChromaIngestor,
    IngestionSummary,
    _sanitize_metadata_for_chroma,
    _get_field,
)
from rag_pipeline.embedding.embed_generator import EmbeddingResult


def make_mock_collection(count_return: int = 0):
    mock = MagicMock()
    mock.count.return_value = count_return
    return mock


class TestSanitizeMetadata(unittest.TestCase):

    def test_list_becomes_comma_separated_string(self):
        result = _sanitize_metadata_for_chroma({"page_numbers": [1, 2, 3]})
        self.assertEqual(result["page_numbers"], "1,2,3")

    def test_single_item_list(self):
        result = _sanitize_metadata_for_chroma({"page_numbers": [5]})
        self.assertEqual(result["page_numbers"], "5")

    def test_empty_list(self):
        result = _sanitize_metadata_for_chroma({"page_numbers": []})
        self.assertEqual(result["page_numbers"], "")

    def test_none_value_dropped(self):
        result = _sanitize_metadata_for_chroma({"foo": None, "bar": "kept"})
        self.assertNotIn("foo", result)
        self.assertEqual(result["bar"], "kept")

    def test_string_passed_through(self):
        result = _sanitize_metadata_for_chroma({"breadcrumb": "### Section"})
        self.assertEqual(result["breadcrumb"], "### Section")

    def test_int_passed_through(self):
        result = _sanitize_metadata_for_chroma({"chunk_index": 5})
        self.assertEqual(result["chunk_index"], 5)
        self.assertIsInstance(result["chunk_index"], int)

    def test_float_passed_through(self):
        result = _sanitize_metadata_for_chroma({"score": 0.95})
        self.assertEqual(result["score"], 0.95)

    def test_bool_passed_through(self):
        result = _sanitize_metadata_for_chroma({"flag": True})
        self.assertEqual(result["flag"], True)
        self.assertIsInstance(result["flag"], bool)

    def test_unknown_type_stringified(self):
        class Weird:
            def __str__(self):
                return "weird_value"
        result = _sanitize_metadata_for_chroma({"odd": Weird()})
        self.assertEqual(result["odd"], "weird_value")

    def test_all_scalar_types_together(self):
        raw = {
            "source_file": "Binder2.md",
            "chunk_index": 3,
            "language": "en",
            "page_numbers": [1, 2],
            "total_pages": 2,
            "some_flag": True,
            "nothing": None,
        }
        result = _sanitize_metadata_for_chroma(raw)
        self.assertEqual(result["source_file"], "Binder2.md")
        self.assertEqual(result["chunk_index"], 3)
        self.assertEqual(result["page_numbers"], "1,2")
        self.assertNotIn("nothing", result)

    def test_real_binder2_style_metadata_survives_sanitization(self):
        """Exact shape produced by metadata_enricher.py for a real chunk."""
        raw = {
            "source_file": "outputs/Binder2/Binder2.md",
            "breadcrumb": "### SERVICE: Great circle > ### Operator actions",
            "section_header": "Operator actions",
            "chunk_type": "table",
            "chunk_index": 2,
            "language": "en",
            "source_pdf": "Binder2.pdf",
            "total_pages": 2,
            "page_numbers": [1],
        }
        result = _sanitize_metadata_for_chroma(raw)
        for key in raw:
            if key != "page_numbers":
                self.assertIn(key, result)
        self.assertEqual(result["page_numbers"], "1")
        # Every value must now be a scalar ChromaDB-acceptable type
        for v in result.values():
            self.assertIsInstance(v, (str, int, float, bool))


class TestGetField(unittest.TestCase):

    def test_dict_access(self):
        self.assertEqual(_get_field({"chunk_id": "abc"}, "chunk_id"), "abc")

    def test_dict_missing_key_returns_default(self):
        self.assertEqual(_get_field({}, "chunk_id", "fallback"), "fallback")

    def test_object_attribute_access(self):
        class Obj:
            chunk_id = "xyz"
        self.assertEqual(_get_field(Obj(), "chunk_id"), "xyz")

    def test_object_missing_attribute_returns_default(self):
        class Obj:
            pass
        self.assertEqual(_get_field(Obj(), "chunk_id", "fallback"), "fallback")


class TestChromaIngestorBasics(unittest.TestCase):

    def _make_ingestor(self, count_return=0):
        ingestor = ChromaIngestor()
        ingestor._collection = make_mock_collection(count_return)
        return ingestor

    def test_ingest_calls_upsert(self):
        ingestor = self._make_ingestor(count_return=2)
        chunks = [
            {"chunk_id": "a", "normalized_text": "Text A", "content": "Text A",
             "contextualized_text": "Ctx A", "metadata": {"chunk_type": "text"}},
            {"chunk_id": "b", "normalized_text": "Text B", "content": "Text B",
             "contextualized_text": "Ctx B", "metadata": {"chunk_type": "text"}},
        ]
        results = [
            EmbeddingResult(chunk_id="a", vector=[0.1, 0.2], dim=2),
            EmbeddingResult(chunk_id="b", vector=[0.3, 0.4], dim=2),
        ]
        summary = ingestor.ingest_chunks(chunks, results)
        ingestor._collection.upsert.assert_called_once()
        self.assertEqual(summary.ingested_count, 2)

    def test_upsert_called_with_correct_ids(self):
        ingestor = self._make_ingestor()
        chunks = [{"chunk_id": "id1", "normalized_text": "T", "metadata": {}}]
        results = [EmbeddingResult(chunk_id="id1", vector=[0.1], dim=1)]
        ingestor.ingest_chunks(chunks, results)
        call_kwargs = ingestor._collection.upsert.call_args.kwargs
        self.assertEqual(call_kwargs["ids"], ["id1"])

    def test_upsert_documents_use_normalized_text_not_raw_html(self):
        ingestor = self._make_ingestor()
        chunks = [{
            "chunk_id": "t1",
            "normalized_text": "Create SOI: clean prose.",
            "content": "<table><tr><td>raw</td></tr></table>",
            "metadata": {},
        }]
        results = [EmbeddingResult(chunk_id="t1", vector=[0.1], dim=1)]
        ingestor.ingest_chunks(chunks, results)
        call_kwargs = ingestor._collection.upsert.call_args.kwargs
        self.assertEqual(call_kwargs["documents"], ["Create SOI: clean prose."])
        self.assertNotIn("<table", call_kwargs["documents"][0])

    def test_content_preserved_in_metadata(self):
        ingestor = self._make_ingestor()
        chunks = [{
            "chunk_id": "t1",
            "normalized_text": "Clean.",
            "content": "<table>raw html</table>",
            "metadata": {},
        }]
        results = [EmbeddingResult(chunk_id="t1", vector=[0.1], dim=1)]
        ingestor.ingest_chunks(chunks, results)
        call_kwargs = ingestor._collection.upsert.call_args.kwargs
        self.assertEqual(call_kwargs["metadatas"][0]["content"], "<table>raw html</table>")

    def test_metadata_merged_with_content_fields(self):
        ingestor = self._make_ingestor()
        chunks = [{
            "chunk_id": "t1",
            "normalized_text": "Text.",
            "content": "Text.",
            "contextualized_text": "Ctx: Text.",
            "metadata": {"chunk_type": "text", "language": "en"},
        }]
        results = [EmbeddingResult(chunk_id="t1", vector=[0.1], dim=1)]
        ingestor.ingest_chunks(chunks, results)
        meta = ingestor._collection.upsert.call_args.kwargs["metadatas"][0]
        self.assertEqual(meta["chunk_type"], "text")
        self.assertEqual(meta["language"], "en")
        self.assertEqual(meta["contextualized_text"], "Ctx: Text.")

    def test_page_numbers_list_sanitized_in_real_ingestion(self):
        """The exact real-world scenario this module was built to prevent
        crashing on: page_numbers is a list, ChromaDB needs a scalar."""
        ingestor = self._make_ingestor()
        chunks = [{
            "chunk_id": "t1",
            "normalized_text": "Text.",
            "content": "Text.",
            "metadata": {"page_numbers": [1, 2], "total_pages": 2},
        }]
        results = [EmbeddingResult(chunk_id="t1", vector=[0.1], dim=1)]
        ingestor.ingest_chunks(chunks, results)
        meta = ingestor._collection.upsert.call_args.kwargs["metadatas"][0]
        self.assertEqual(meta["page_numbers"], "1,2")
        self.assertIsInstance(meta["page_numbers"], str)

    def test_chunk_without_vector_is_skipped(self):
        ingestor = self._make_ingestor()
        chunks = [
            {"chunk_id": "has_vector", "normalized_text": "A", "metadata": {}},
            {"chunk_id": "no_vector", "normalized_text": "B", "metadata": {}},
        ]
        results = [EmbeddingResult(chunk_id="has_vector", vector=[0.1], dim=1)]
        summary = ingestor.ingest_chunks(chunks, results)
        self.assertEqual(summary.ingested_count, 1)
        self.assertEqual(summary.skipped_no_vector, 1)

    def test_matches_by_chunk_id_not_position(self):
        """Embedding results in a DIFFERENT order than chunks must still
        match correctly — matching is by chunk_id, not list position."""
        ingestor = self._make_ingestor()
        chunks = [
            {"chunk_id": "second", "normalized_text": "Second content", "metadata": {}},
            {"chunk_id": "first", "normalized_text": "First content", "metadata": {}},
        ]
        # Results provided in the OPPOSITE order of chunks
        results = [
            EmbeddingResult(chunk_id="first", vector=[1.0], dim=1),
            EmbeddingResult(chunk_id="second", vector=[2.0], dim=1),
        ]
        ingestor.ingest_chunks(chunks, results)
        call_kwargs = ingestor._collection.upsert.call_args.kwargs
        # "second" chunk (first in `chunks` list) must get vector [2.0]
        self.assertEqual(call_kwargs["ids"][0], "second")
        self.assertEqual(call_kwargs["embeddings"][0], [2.0])
        self.assertEqual(call_kwargs["ids"][1], "first")
        self.assertEqual(call_kwargs["embeddings"][1], [1.0])

    def test_empty_chunks_list(self):
        ingestor = self._make_ingestor(count_return=0)
        summary = ingestor.ingest_chunks([], [])
        self.assertEqual(summary.ingested_count, 0)

    def test_summary_includes_collection_count_after(self):
        ingestor = self._make_ingestor(count_return=42)
        chunks = [{"chunk_id": "x", "normalized_text": "T", "metadata": {}}]
        results = [EmbeddingResult(chunk_id="x", vector=[0.1], dim=1)]
        summary = ingestor.ingest_chunks(chunks, results)
        self.assertEqual(summary.collection_count_after, 42)

    def test_enriched_chunk_objects_supported(self):
        """Must work with real EnrichedChunk objects, not just dicts."""
        from rag_pipeline.enrichers.metadata_enricher import EnrichedChunk
        ingestor = self._make_ingestor()
        chunk = EnrichedChunk(
            content="raw",
            chunk_id="obj1",
            normalized_text="Clean text.",
            metadata={"chunk_type": "text"},
        )
        results = [EmbeddingResult(chunk_id="obj1", vector=[0.5], dim=1)]
        summary = ingestor.ingest_chunks([chunk], results)
        self.assertEqual(summary.ingested_count, 1)


class TestChromaIngestorBatching(unittest.TestCase):

    def test_upsert_called_once_per_batch(self):
        cfg = ChromaIngestionConfig(batch_size=2)
        ingestor = ChromaIngestor(cfg)
        ingestor._collection = make_mock_collection(count_return=5)

        chunks = [
            {"chunk_id": str(i), "normalized_text": f"Text {i}", "metadata": {}}
            for i in range(5)
        ]
        results = [
            EmbeddingResult(chunk_id=str(i), vector=[float(i)], dim=1)
            for i in range(5)
        ]
        ingestor.ingest_chunks(chunks, results)
        # 5 items, batch_size=2 → 3 calls (2, 2, 1)
        self.assertEqual(ingestor._collection.upsert.call_count, 3)

    def test_batch_sizes_correct(self):
        cfg = ChromaIngestionConfig(batch_size=2)
        ingestor = ChromaIngestor(cfg)
        ingestor._collection = make_mock_collection()

        chunks = [
            {"chunk_id": str(i), "normalized_text": f"T{i}", "metadata": {}}
            for i in range(5)
        ]
        results = [
            EmbeddingResult(chunk_id=str(i), vector=[float(i)], dim=1)
            for i in range(5)
        ]
        ingestor.ingest_chunks(chunks, results)
        call_sizes = [len(c.kwargs["ids"]) for c in ingestor._collection.upsert.call_args_list]
        self.assertEqual(call_sizes, [2, 2, 1])

    def test_all_items_ingested_across_batches(self):
        cfg = ChromaIngestionConfig(batch_size=3)
        ingestor = ChromaIngestor(cfg)
        ingestor._collection = make_mock_collection(count_return=7)

        chunks = [
            {"chunk_id": str(i), "normalized_text": f"T{i}", "metadata": {}}
            for i in range(7)
        ]
        results = [
            EmbeddingResult(chunk_id=str(i), vector=[float(i)], dim=1)
            for i in range(7)
        ]
        summary = ingestor.ingest_chunks(chunks, results)
        self.assertEqual(summary.ingested_count, 7)


class TestChromaIngestorLazyLoading(unittest.TestCase):

    def test_import_error_when_chromadb_missing(self):
        ingestor = ChromaIngestor()
        with patch.dict(sys.modules, {"chromadb": None}):
            with self.assertRaises(ImportError):
                ingestor._get_client()

    def test_persistent_client_created_with_configured_path(self):
        cfg = ChromaIngestionConfig(persist_directory="/tmp/my_chroma")
        ingestor = ChromaIngestor(cfg)
        mock_chromadb = MagicMock()

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            ingestor._get_client()

        mock_chromadb.PersistentClient.assert_called_once_with(path="/tmp/my_chroma")

    def test_client_created_lazily_once(self):
        ingestor = ChromaIngestor()
        mock_chromadb = MagicMock()

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            ingestor._get_client()
            ingestor._get_client()

        mock_chromadb.PersistentClient.assert_called_once()

    def test_collection_created_with_configured_name(self):
        cfg = ChromaIngestionConfig(collection_name="my_collection")
        ingestor = ChromaIngestor(cfg)
        mock_client = MagicMock()
        ingestor._client = mock_client

        ingestor._get_collection()

        mock_client.get_or_create_collection.assert_called_once_with(
            name="my_collection"
        )

    def test_collection_created_lazily_once(self):
        ingestor = ChromaIngestor()
        mock_client = MagicMock()
        ingestor._client = mock_client

        ingestor._get_collection()
        ingestor._get_collection()

        mock_client.get_or_create_collection.assert_called_once()


class TestChromaIngestorQuery(unittest.TestCase):

    def test_query_calls_collection_query(self):
        ingestor = ChromaIngestor()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [["result"]]}
        ingestor._collection = mock_collection

        result = ingestor.query([0.1, 0.2, 0.3], n_results=5)

        mock_collection.query.assert_called_once()
        call_kwargs = mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["query_embeddings"], [[0.1, 0.2, 0.3]])
        self.assertEqual(call_kwargs["n_results"], 5)
        self.assertEqual(result, {"documents": [["result"]]})

    def test_query_with_where_filter(self):
        ingestor = ChromaIngestor()
        mock_collection = MagicMock()
        ingestor._collection = mock_collection

        ingestor.query([0.1], where={"language": "en"})

        call_kwargs = mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"language": "en"})

    def test_query_without_where_omits_it(self):
        ingestor = ChromaIngestor()
        mock_collection = MagicMock()
        ingestor._collection = mock_collection

        ingestor.query([0.1])

        call_kwargs = mock_collection.query.call_args.kwargs
        self.assertNotIn("where", call_kwargs)

    def test_count_returns_collection_count(self):
        ingestor = ChromaIngestor()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 99
        ingestor._collection = mock_collection

        self.assertEqual(ingestor.count(), 99)


class TestChromaIngestorGetByIds(unittest.TestCase):

    def test_calls_collection_get_with_ids(self):
        ingestor = ChromaIngestor()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["a", "b"],
            "documents": ["doc a", "doc b"],
            "metadatas": [{"x": 1}, {"x": 2}],
        }
        ingestor._collection = mock_collection

        result = ingestor.get_by_ids(["a", "b"])

        mock_collection.get.assert_called_once_with(
            ids=["a", "b"], include=["documents", "metadatas"]
        )
        self.assertEqual(result["ids"], ["a", "b"])

    def test_empty_ids_returns_empty_without_calling_collection(self):
        ingestor = ChromaIngestor()
        mock_collection = MagicMock()
        ingestor._collection = mock_collection

        result = ingestor.get_by_ids([])

        mock_collection.get.assert_not_called()
        self.assertEqual(result["ids"], [])

    def test_returns_flat_shape_not_nested(self):
        """get_by_ids must return .get()'s FLAT shape, not .query()'s
        nested list-of-lists shape — this is the exact API mismatch
        the module docstring warns about."""
        ingestor = ChromaIngestor()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["a"],
            "documents": ["doc a"],
            "metadatas": [{"x": 1}],
        }
        ingestor._collection = mock_collection

        result = ingestor.get_by_ids(["a"])

        # Flat: result["ids"][0] == "a", NOT result["ids"][0][0] == "a"
        self.assertEqual(result["ids"][0], "a")
        self.assertIsInstance(result["ids"][0], str)


class TestEndToEndWithMockedChroma(unittest.TestCase):
    """Full flow: chunks + embeddings → ingest → query, all mocked."""

    def test_full_ingest_and_query_flow(self):
        mock_chromadb = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "ids": [["a", "b"]],
            "documents": [["Create SOI content", "System reports content"]],
            "distances": [[0.1, 0.3]],
        }
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            ingestor = ChromaIngestor(ChromaIngestionConfig())

            chunks = [
                {"chunk_id": "a", "normalized_text": "Create SOI content",
                 "content": "Create SOI content", "metadata": {"chunk_type": "table",
                 "page_numbers": [1]}},
                {"chunk_id": "b", "normalized_text": "System reports content",
                 "content": "System reports content", "metadata": {"chunk_type": "table"}},
            ]
            results = [
                EmbeddingResult(chunk_id="a", vector=[0.1, 0.2], dim=2),
                EmbeddingResult(chunk_id="b", vector=[0.3, 0.4], dim=2),
            ]

            summary = ingestor.ingest_chunks(chunks, results)
            self.assertEqual(summary.ingested_count, 2)

            query_result = ingestor.query([0.15, 0.25], n_results=2)
            self.assertEqual(query_result["ids"], [["a", "b"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
