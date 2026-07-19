"""
tests/test_rag_answerer.py

Unit tests for RagAnswerer — using fake retriever/reranker/llm_client
(dependency injection makes this trivial to test in full isolation),
plus a real-fixture-data integration test.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.config import ContextBuilderConfig
from Chunker.generation.rag_answerer import RagAnswerer, RagAnswer
from Chunker.retrieval.hybrid_retriever import RetrievedChunk
from Chunker.retrieval.reranker import RerankedChunk


def make_chunk(chunk_id="a", text="Some content.", metadata=None, rrf_score=0.01):
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, metadata=metadata or {}, rrf_score=rrf_score,
    )


class FakeRetriever:
    def __init__(self, results=None):
        self._results = results if results is not None else [make_chunk()]
        self.last_query = None

    def retrieve(self, query):
        self.last_query = query
        return self._results


class FakeReranker:
    def __init__(self, results=None):
        self._results = results
        self.last_query = None
        self.last_candidates = None

    def rerank(self, query, candidates):
        self.last_query = query
        self.last_candidates = candidates
        return self._results if self._results is not None else candidates


class FakeLlmClient:
    def __init__(self, response="This is the generated answer."):
        self._response = response
        self.last_system_prompt = None
        self.last_user_prompt = None

    def generate(self, system_prompt, user_prompt):
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self._response


class TestRagAnswererBasics(unittest.TestCase):

    def test_returns_rag_answer_instance(self):
        answerer = RagAnswerer(FakeRetriever(), FakeLlmClient())
        result = answerer.answer("What is a CMI?")
        self.assertIsInstance(result, RagAnswer)

    def test_answer_text_comes_from_llm(self):
        llm = FakeLlmClient(response="The CMI is a track correction mechanism.")
        answerer = RagAnswerer(FakeRetriever(), llm)
        result = answerer.answer("What is a CMI?")
        self.assertEqual(result.answer, "The CMI is a track correction mechanism.")

    def test_empty_query_returns_no_results_message_without_calling_retriever(self):
        retriever = FakeRetriever()
        answerer = RagAnswerer(retriever, FakeLlmClient())
        result = answerer.answer("")
        self.assertIn("couldn't find", result.answer.lower())
        self.assertIsNone(retriever.last_query)

    def test_whitespace_query_returns_no_results_message(self):
        answerer = RagAnswerer(FakeRetriever(), FakeLlmClient())
        result = answerer.answer("   ")
        self.assertIn("couldn't find", result.answer.lower())

    def test_no_retrieval_results_returns_no_results_message_without_calling_llm(self):
        retriever = FakeRetriever(results=[])
        llm = FakeLlmClient()
        answerer = RagAnswerer(retriever, llm)
        result = answerer.answer("obscure query with no matches")
        self.assertIn("couldn't find", result.answer.lower())
        self.assertIsNone(llm.last_system_prompt)

    def test_sources_include_chunk_metadata(self):
        chunk = make_chunk(
            chunk_id="abc123",
            metadata={"source_file": "Binder2.md", "breadcrumb": "### System reports"},
        )
        answerer = RagAnswerer(FakeRetriever(results=[chunk]), FakeLlmClient())
        result = answerer.answer("query")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0]["chunk_id"], "abc123")
        self.assertEqual(result.sources[0]["source_file"], "Binder2.md")
        self.assertEqual(result.sources[0]["breadcrumb"], "### System reports")

    def test_context_used_is_populated(self):
        chunk = make_chunk(text="Create SOI content.")
        answerer = RagAnswerer(FakeRetriever(results=[chunk]), FakeLlmClient())
        result = answerer.answer("query")
        self.assertIn("Create SOI content.", result.context_used)

    def test_context_used_empty_when_no_results(self):
        answerer = RagAnswerer(FakeRetriever(results=[]), FakeLlmClient())
        result = answerer.answer("query")
        self.assertEqual(result.context_used, "")


class TestRagAnswererReranking(unittest.TestCase):

    def test_reranker_called_when_provided(self):
        retrieved = [make_chunk(chunk_id="a"), make_chunk(chunk_id="b")]
        reranker = FakeReranker(results=[make_chunk(chunk_id="b")])
        answerer = RagAnswerer(FakeRetriever(results=retrieved), FakeLlmClient(), reranker=reranker)

        answerer.answer("query")

        self.assertEqual(reranker.last_query, "query")
        self.assertEqual(reranker.last_candidates, retrieved)

    def test_reranked_results_used_for_context_not_original(self):
        retrieved = [make_chunk(chunk_id="a", text="Original content.")]
        reranked = [make_chunk(chunk_id="b", text="Reranked content.")]
        reranker = FakeReranker(results=reranked)
        answerer = RagAnswerer(FakeRetriever(results=retrieved), FakeLlmClient(), reranker=reranker)

        result = answerer.answer("query")

        self.assertIn("Reranked content.", result.context_used)
        self.assertNotIn("Original content.", result.context_used)

    def test_no_reranker_skips_reranking_entirely(self):
        retrieved = [make_chunk(text="Direct retrieval content.")]
        answerer = RagAnswerer(FakeRetriever(results=retrieved), FakeLlmClient())  # no reranker
        result = answerer.answer("query")
        self.assertIn("Direct retrieval content.", result.context_used)

    def test_rerank_empty_result_returns_no_results_message(self):
        retrieved = [make_chunk()]
        reranker = FakeReranker(results=[])
        answerer = RagAnswerer(FakeRetriever(results=retrieved), FakeLlmClient(), reranker=reranker)
        result = answerer.answer("query")
        self.assertIn("couldn't find", result.answer.lower())


class TestRagAnswererPromptConstruction(unittest.TestCase):

    def test_user_prompt_includes_query_and_context(self):
        chunk = make_chunk(text="Relevant fact.")
        llm = FakeLlmClient()
        answerer = RagAnswerer(FakeRetriever(results=[chunk]), llm)
        answerer.answer("What is the fact?")
        self.assertIn("What is the fact?", llm.last_user_prompt)
        self.assertIn("Relevant fact.", llm.last_user_prompt)

    def test_system_prompt_mentions_context_only_constraint(self):
        llm = FakeLlmClient()
        answerer = RagAnswerer(FakeRetriever(), llm)
        answerer.answer("query")
        self.assertIn("ONLY the", llm.last_system_prompt)

    def test_system_prompt_english_for_english_query(self):
        llm = FakeLlmClient()
        answerer = RagAnswerer(FakeRetriever(), llm)
        answerer.answer("What is the operator action for creating a CMI track?")
        self.assertIn("Answer in English", llm.last_system_prompt)

    def test_system_prompt_spanish_for_spanish_query(self):
        llm = FakeLlmClient()
        answerer = RagAnswerer(FakeRetriever(), llm)
        answerer.answer("Definición del botón de conjunto de trabajo nueva posición")
        self.assertIn("Responde en español", llm.last_system_prompt)

    def test_system_prompt_mentions_citation_format(self):
        llm = FakeLlmClient()
        answerer = RagAnswerer(FakeRetriever(), llm)
        answerer.answer("query")
        self.assertIn("[1]", llm.last_system_prompt)


class TestRagAnswererContextConfig(unittest.TestCase):

    def test_custom_context_config_respected(self):
        chunks = [make_chunk(chunk_id=str(i), text=f"Content {i}") for i in range(5)]
        cfg = ContextBuilderConfig(max_chunks=2)
        llm = FakeLlmClient()
        answerer = RagAnswerer(FakeRetriever(results=chunks), llm, context_config=cfg)
        result = answerer.answer("query")
        self.assertIn("Content 0", result.context_used)
        self.assertIn("Content 1", result.context_used)
        self.assertNotIn("Content 4", result.context_used)


class TestRagAnswererWorksWithRerankedChunks(unittest.TestCase):
    """RagAnswerer must handle RerankedChunk objects, not just RetrievedChunk."""

    def test_reranked_chunk_sources_extracted_correctly(self):
        reranked = RerankedChunk(
            chunk_id="xyz", text="Content.", metadata={"source_file": "f.md"},
            rerank_score=0.9, original_rrf_score=0.01, original_rank=0,
        )
        reranker = FakeReranker(results=[reranked])
        answerer = RagAnswerer(FakeRetriever(results=[make_chunk()]), FakeLlmClient(), reranker=reranker)
        result = answerer.answer("query")
        self.assertEqual(result.sources[0]["chunk_id"], "xyz")
        self.assertEqual(result.sources[0]["source_file"], "f.md")


class TestRagAnswererRealFixtures(unittest.TestCase):
    """Integration test using real chunk data from the pipeline."""

    def _get_real_chunks(self) -> list[RetrievedChunk]:
        from Chunker.tests.fixtures_path import FIXTURES_DIR
        from Chunker.config import PipelineConfig
        from Chunker.pipeline import MarkdownChunkingPipeline

        cfg = PipelineConfig(input_dir=FIXTURES_DIR, verbose=False)
        pipeline = MarkdownChunkingPipeline(cfg)
        chunks = pipeline.process_file(FIXTURES_DIR / "Binder2.md")
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id, text=c.normalized_text, metadata=c.metadata,
                rrf_score=0.01,
            )
            for c in chunks
        ]

    def test_real_data_produces_full_answer_flow(self):
        real_chunks = self._get_real_chunks()
        retriever = FakeRetriever(results=real_chunks[:3])
        llm = FakeLlmClient(response="Based on the context, a CMI is a course/manoeuvre indicator.")

        answerer = RagAnswerer(retriever, llm)
        result = answerer.answer("What is a CMI?")

        self.assertEqual(result.answer, "Based on the context, a CMI is a course/manoeuvre indicator.")
        self.assertEqual(len(result.sources), 3)
        self.assertIn("Binder2", result.context_used)

    def test_real_data_no_html_reaches_llm_prompt(self):
        real_chunks = self._get_real_chunks()
        table_chunks = [c for c in real_chunks if c.metadata.get("chunk_type") == "table"]
        llm = FakeLlmClient()
        answerer = RagAnswerer(FakeRetriever(results=table_chunks), llm)

        answerer.answer("What are the operator actions?")

        self.assertNotIn("<table", llm.last_user_prompt)
        self.assertNotIn("<td", llm.last_user_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
