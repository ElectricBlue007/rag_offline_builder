"""
tests/test_embed_generator.py

Unit tests for EmbedGenerator and its file-loading helpers, using a
mocked Ollama client — no live Ollama service or model pull required.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.config import EmbedGeneratorConfig
from Chunker.embedding.embed_generator import (
    EmbedGenerator,
    EmbeddingResult,
    discover_chunk_files,
    load_chunks_from_json,
    load_all_chunks,
)


def make_mock_client(vector_dim: int = 4):
    """
    Return a mock Ollama client whose .embed(model=..., input=[...])
    yields deterministic vectors based on each input string's length,
    for easy assertions — mirrors the real client.embed() response
    shape: {"embeddings": [[...], [...]]}.
    """
    def _embed(model, input):
        return {"embeddings": [[float(len(t))] * vector_dim for t in input]}
    mock = MagicMock()
    mock.embed.side_effect = _embed
    return mock


class TestEmbedGeneratorBasics(unittest.TestCase):

    def setUp(self):
        self.gen = EmbedGenerator()
        self.gen._client = make_mock_client(vector_dim=4)

    def test_embed_texts_returns_list_of_lists(self):
        result = self.gen.embed_texts(["passage: hello", "passage: world"])
        self.assertEqual(len(result), 2)
        for vec in result:
            self.assertIsInstance(vec, list)
            self.assertEqual(len(vec), 4)

    def test_embed_texts_preserves_order(self):
        result = self.gen.embed_texts(["passage: a", "passage: bb", "passage: ccc"])
        lengths = [v[0] for v in result]  # our mock encodes len(text) into vector
        # "passage: a"=10 chars, "passage: bb"=11 chars, "passage: ccc"=12 chars
        self.assertEqual(lengths, [10.0, 11.0, 12.0])

    def test_empty_texts_returns_empty_without_loading_model(self):
        gen = EmbedGenerator()
        with patch.object(gen, "_get_client") as mock_get_client:
            result = gen.embed_texts([])
            self.assertEqual(result, [])
            mock_get_client.assert_not_called()

    def test_vectors_are_plain_floats_json_serializable(self):
        result = self.gen.embed_texts(["passage: test"])
        json.dumps(result)  # must not raise


class TestEmbedGeneratorChunks(unittest.TestCase):

    def setUp(self):
        self.gen = EmbedGenerator()
        self.gen._client = make_mock_client(vector_dim=3)

    def test_embed_chunks_from_dicts(self):
        chunks = [
            {"chunk_id": "abc123", "embed_text": "passage: content one"},
            {"chunk_id": "def456", "embed_text": "passage: content two"},
        ]
        results = self.gen.embed_chunks(chunks)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], EmbeddingResult)
        self.assertEqual(results[0].chunk_id, "abc123")
        self.assertEqual(results[1].chunk_id, "def456")

    def test_embed_chunks_from_enriched_chunk_objects(self):
        from Chunker.enrichers.metadata_enricher import EnrichedChunk
        chunks = [
            EnrichedChunk(content="c1", chunk_id="id1", embed_text="passage: c1"),
            EnrichedChunk(content="c2", chunk_id="id2", embed_text="passage: c2"),
        ]
        results = self.gen.embed_chunks(chunks)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "id1")
        self.assertEqual(results[1].chunk_id, "id2")

    def test_dim_matches_vector_length(self):
        chunks = [{"chunk_id": "x", "embed_text": "passage: test"}]
        results = self.gen.embed_chunks(chunks)
        self.assertEqual(results[0].dim, len(results[0].vector))
        self.assertEqual(results[0].dim, 3)

    def test_empty_chunks_list(self):
        results = self.gen.embed_chunks([])
        self.assertEqual(results, [])

    def test_missing_fields_default_to_empty_string(self):
        """A malformed chunk dict shouldn't crash — just embeds empty text."""
        chunks = [{"chunk_id": "x"}]  # no embed_text key
        results = self.gen.embed_chunks(chunks)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "x")

    def test_order_preserved_across_mixed_batch(self):
        chunks = [
            {"chunk_id": "first", "embed_text": "passage: a"},
            {"chunk_id": "second", "embed_text": "passage: bb"},
            {"chunk_id": "third", "embed_text": "passage: ccc"},
        ]
        results = self.gen.embed_chunks(chunks)
        ids = [r.chunk_id for r in results]
        self.assertEqual(ids, ["first", "second", "third"])


class TestEmbedGeneratorClientLoading(unittest.TestCase):

    def test_import_error_when_ollama_missing(self):
        gen = EmbedGenerator()
        with patch.dict(sys.modules, {"ollama": None}):
            with self.assertRaises(ImportError):
                gen.embed_texts(["passage: test"])

    def test_client_created_lazily_once(self):
        gen = EmbedGenerator()
        mock_ollama = MagicMock()
        mock_client_cls = MagicMock()
        mock_client_instance = make_mock_client()
        mock_client_cls.return_value = mock_client_instance
        mock_ollama.Client = mock_client_cls

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            gen.embed_texts(["passage: one"])
            gen.embed_texts(["passage: two"])

        # Client() constructor should only be called once, even across
        # two separate embed_texts() calls.
        mock_client_cls.assert_called_once()

    def test_host_passed_to_client_constructor(self):
        cfg = EmbedGeneratorConfig(host="http://example-host:11434")
        gen = EmbedGenerator(cfg)
        mock_ollama = MagicMock()
        mock_client_cls = MagicMock()
        mock_client_cls.return_value = make_mock_client()
        mock_ollama.Client = mock_client_cls

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            gen.embed_texts(["passage: test"])

        call_kwargs = mock_client_cls.call_args.kwargs
        self.assertEqual(call_kwargs["host"], "http://example-host:11434")

    def test_model_name_passed_to_embed_call(self):
        cfg = EmbedGeneratorConfig(model_name="qwen3-embedding:4b")
        gen = EmbedGenerator(cfg)
        gen._client = make_mock_client()

        gen.embed_texts(["passage: test"])

        call_kwargs = gen._client.embed.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "qwen3-embedding:4b")

    def test_batch_size_splits_into_multiple_embed_calls(self):
        cfg = EmbedGeneratorConfig(batch_size=2)
        gen = EmbedGenerator(cfg)
        gen._client = make_mock_client()

        gen.embed_texts(["a", "b", "c", "d", "e"])  # 5 texts, batch_size=2 → 3 calls

        self.assertEqual(gen._client.embed.call_count, 3)
        first_call_input = gen._client.embed.call_args_list[0].kwargs["input"]
        self.assertEqual(first_call_input, ["a", "b"])
        last_call_input = gen._client.embed.call_args_list[-1].kwargs["input"]
        self.assertEqual(last_call_input, ["e"])

    def test_single_batch_when_texts_fit_within_batch_size(self):
        cfg = EmbedGeneratorConfig(batch_size=16)
        gen = EmbedGenerator(cfg)
        gen._client = make_mock_client()

        gen.embed_texts(["passage: a", "passage: b"])

        gen._client.embed.assert_called_once()


class TestExtractEmbeddings(unittest.TestCase):
    """
    Tests for the defensive dict-vs-attribute response parsing in
    EmbedGenerator._extract_embeddings — ollama-python's response shape
    has varied across versions, so both must work (mirrors the same
    pattern tested for LlmClient's response parsing).
    """

    def test_dict_style_response(self):
        response = {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}
        result = EmbedGenerator._extract_embeddings(response)
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_attribute_style_response(self):
        class FakeResponse:
            embeddings = [[1.0, 2.0], [3.0, 4.0]]
        result = EmbedGenerator._extract_embeddings(FakeResponse())
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_tuple_vectors_converted_to_lists(self):
        response = {"embeddings": [(1.0, 2.0), (3.0, 4.0)]}
        result = EmbedGenerator._extract_embeddings(response)
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
        for vec in result:
            self.assertIsInstance(vec, list)

    def test_empty_embeddings_list(self):
        response = {"embeddings": []}
        result = EmbedGenerator._extract_embeddings(response)
        self.assertEqual(result, [])


class TestDiscoverChunkFiles(unittest.TestCase):

    def test_finds_chunks_json_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "a.chunks.json").write_text('{"chunks": []}')
            (p / "b.chunks.json").write_text('{"chunks": []}')
            (p / "not_a_chunk_file.json").write_text('{}')
            found = discover_chunk_files(p)
            self.assertEqual(len(found), 2)

    def test_finds_nested_chunk_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            sub = p / "vol1" / "chap2"
            sub.mkdir(parents=True)
            (sub / "file.chunks.json").write_text('{"chunks": []}')
            found = discover_chunk_files(p)
            self.assertEqual(len(found), 1)

    def test_empty_directory_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            found = discover_chunk_files(Path(tmp))
            self.assertEqual(found, [])

    def test_returns_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in ["c.chunks.json", "a.chunks.json", "b.chunks.json"]:
                (p / name).write_text('{"chunks": []}')
            found = discover_chunk_files(p)
            names = [f.name for f in found]
            self.assertEqual(names, sorted(names))


class TestLoadChunksFromJson(unittest.TestCase):

    def test_loads_chunks_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.chunks.json"
            payload = {
                "source_file": "test.md",
                "total_chunks": 2,
                "chunks": [
                    {"chunk_id": "a", "embed_text": "passage: one"},
                    {"chunk_id": "b", "embed_text": "passage: two"},
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = load_chunks_from_json(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["chunk_id"], "a")

    def test_missing_chunks_key_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.chunks.json"
            path.write_text('{"source_file": "test.md"}', encoding="utf-8")
            result = load_chunks_from_json(path)
            self.assertEqual(result, [])


class TestLoadAllChunks(unittest.TestCase):

    def test_concatenates_across_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "a.chunks.json").write_text(
                json.dumps({"chunks": [{"chunk_id": "1", "embed_text": "x"}]})
            )
            (p / "b.chunks.json").write_text(
                json.dumps({"chunks": [{"chunk_id": "2", "embed_text": "y"},
                                        {"chunk_id": "3", "embed_text": "z"}]})
            )
            result = load_all_chunks(p)
            self.assertEqual(len(result), 3)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = load_all_chunks(Path(tmp))
            self.assertEqual(result, [])


class TestEndToEndWithMockedClient(unittest.TestCase):
    """Full flow: write chunks.json → load → embed, using a mocked Ollama client."""

    def test_full_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            payload = {
                "source_file": "Binder2.md",
                "chunks": [
                    {"chunk_id": "aaa111", "embed_text": "passage: Create SOI content."},
                    {"chunk_id": "bbb222", "embed_text": "passage: System reports content."},
                ],
            }
            (p / "Binder2.chunks.json").write_text(json.dumps(payload), encoding="utf-8")

            chunks = load_all_chunks(p)
            gen = EmbedGenerator()
            gen._client = make_mock_client(vector_dim=4)
            results = gen.embed_chunks(chunks)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].chunk_id, "aaa111")
            self.assertEqual(results[1].chunk_id, "bbb222")
            for r in results:
                self.assertEqual(len(r.vector), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
