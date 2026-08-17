"""
tests/test_query_trace.py -- tests for observability/query_trace.py

These tests use the REAL HybridRetriever, the REAL RRF fusion and the REAL
RagAnswerer, with fakes only at the edges (embedding model, ChromaDB, BM25
index, LLM). So they verify the tracer against the actual pipeline logic
without needing Ollama, Chroma or any model download.

The most important guarantees, in order:
  1. Tracing NEVER changes the answer.
  2. Tracing ALWAYS puts the wrapped methods back, even when the call fails.
  3. What the panel shows about the prompt is the literal prompt that was sent.
"""

import unittest
from types import SimpleNamespace

from rag_pipeline.config import ContextBuilderConfig, HybridRetrieverConfig
from rag_pipeline.generation.rag_answerer import RagAnswerer
from rag_pipeline.observability.query_trace import (
    QueryTracer,
    demo_trace,
    trace_answer,
)
from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever


# --------------------------------------------------------------------------
# Fakes at the edges
# --------------------------------------------------------------------------

class FakeEmbedder:
    def __init__(self):
        self.cfg = SimpleNamespace(model_name="fake-embed", host=None)
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3]]


class FakeChroma:
    """Mimics chroma_ingestion.query()'s nested list-of-lists response."""

    def __init__(self, docs):
        # docs: list of (chunk_id, text, metadata)
        self.docs = docs
        self.query_calls = []

    def query(self, query_vector, n_results=10, where=None):
        self.query_calls.append({"n_results": n_results, "where": where})
        picked = self.docs[:n_results]
        return {
            "ids": [[d[0] for d in picked]],
            "documents": [[d[1] for d in picked]],
            "metadatas": [[d[2] for d in picked]],
            "distances": [[0.1 * (i + 1) for i in range(len(picked))]],
        }

    def get_by_ids(self, ids):
        by_id = {d[0]: d for d in self.docs}
        found = [by_id[i] for i in ids if i in by_id]
        return {
            "ids": [d[0] for d in found],
            "documents": [d[1] for d in found],
            "metadatas": [d[2] for d in found],
        }


class FakeBM25:
    def __init__(self, hits):
        # hits: list of (chunk_id, score)
        self.hits = hits
        self.search_calls = []

    def search(self, query, n_results=10):
        self.search_calls.append({"query": query, "n_results": n_results})
        return [
            SimpleNamespace(chunk_id=cid, score=score)
            for cid, score in self.hits[:n_results]
        ]


class FakeLlm:
    def __init__(self, answer="An answer citing [1].", raises=None):
        self.cfg = SimpleNamespace(
            model="fake-llm", num_ctx=8192, temperature=0.1, host=None
        )
        self.answer_text = answer
        self.raises = raises
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.raises:
            raise self.raises
        return self.answer_text


class FakeReranker:
    """Reverses the candidate order, so rank movement is easy to assert."""

    def __init__(self):
        self.cfg = SimpleNamespace(
            backend="ollama", ollama_model="fake-rerank", model_name=None, host=None
        )
        self.calls = []

    def rerank(self, query, candidates):
        self.calls.append((query, list(candidates)))
        out = []
        for position, chunk in enumerate(reversed(list(candidates))):
            out.append(SimpleNamespace(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                rerank_score=1.0 - (position * 0.1),
                original_rrf_score=chunk.rrf_score,
                original_rank=position,
            ))
        return out


def _metadata(source, page, crumb, language="en"):
    return {
        "source_file": source,
        "page_numbers": page,
        "breadcrumb": crumb,
        "language": language,
        "chunk_type": "text",
    }


DOCS = [
    ("c1", "One-shot prompting gives the model a single worked example.",
     _metadata("Testing.md", "3", "Prompting > One-shot")),
    ("c2", "Zero-shot prompting uses no examples at all.",
     _metadata("Testing.md", "2", "Prompting > Zero-shot")),
    ("c3", "Temperature controls randomness of the output.",
     _metadata("Binder2.md", "11", "Generation > Temperature")),
]


def build_answerer(
    docs=DOCS,
    bm25_hits=(("c2", 5.5), ("c1", 4.25), ("c3", 0.5)),
    llm=None,
    reranker=None,
    max_chunks=2,
    max_chars_per_chunk=1500,
    language_filter=False,
):
    embedder = FakeEmbedder()
    chroma = FakeChroma(list(docs))
    bm25 = FakeBM25(list(bm25_hits))
    retriever = HybridRetriever(
        chroma, bm25, embedder,
        HybridRetrieverConfig(
            dense_n_results=3,
            sparse_n_results=3,
            final_n_results=3,
            language_filter=language_filter,
        ),
    )
    llm = llm or FakeLlm()
    answerer = RagAnswerer(
        retriever=retriever,
        llm_client=llm,
        reranker=reranker,
        context_config=ContextBuilderConfig(
            max_chunks=max_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
        ),
    )
    return answerer, embedder, chroma, bm25, llm


# --------------------------------------------------------------------------

class TestTracingDoesNotChangeBehaviour(unittest.TestCase):
    def test_answer_text_is_identical_with_and_without_tracing(self):
        plain_answerer, *_ = build_answerer()
        plain = plain_answerer.answer("What is one-shot prompting?")

        traced_answerer, *_ = build_answerer()
        traced, trace = trace_answer(traced_answerer, "What is one-shot prompting?")

        self.assertEqual(plain.answer, traced.answer)
        self.assertEqual(plain.context_used, traced.context_used)
        self.assertEqual(
            [s["chunk_id"] for s in plain.sources],
            [s["chunk_id"] for s in traced.sources],
        )
        self.assertIsNotNone(trace)

    def test_methods_are_restored_after_a_successful_trace(self):
        answerer, embedder, chroma, bm25, llm = build_answerer()
        trace_answer(answerer, "What is one-shot prompting?")

        # No leftover instance attributes shadowing the real methods.
        self.assertNotIn("embed_texts", embedder.__dict__)
        self.assertNotIn("query", chroma.__dict__)
        self.assertNotIn("search", bm25.__dict__)
        self.assertNotIn("generate", llm.__dict__)
        self.assertNotIn("retrieve", answerer.retriever.__dict__)

    def test_methods_are_restored_even_when_the_answer_fails(self):
        boom = RuntimeError("ollama is down")
        answerer, embedder, chroma, bm25, llm = build_answerer(llm=FakeLlm(raises=boom))

        with self.assertRaises(RuntimeError):
            trace_answer(answerer, "What is one-shot prompting?")

        self.assertNotIn("embed_texts", embedder.__dict__)
        self.assertNotIn("query", chroma.__dict__)
        self.assertNotIn("search", bm25.__dict__)
        self.assertNotIn("generate", llm.__dict__)
        self.assertNotIn("retrieve", answerer.retriever.__dict__)

        # And the pipeline still works normally afterwards.
        recovered, *_ = build_answerer()
        self.assertTrue(recovered.answer("anything").answer)


class TestRetrievalIsRecorded(unittest.TestCase):
    def setUp(self):
        self.answerer, self.embedder, self.chroma, self.bm25, self.llm = build_answerer()
        self.result, self.trace = trace_answer(
            self.answerer, "What is one-shot prompting?"
        )

    def test_dense_search_is_recorded(self):
        dense = self.trace["dense"]
        self.assertEqual(dense["requested"], 3)
        self.assertEqual(dense["returned"], 3)
        self.assertEqual(dense["ids"], ["c1", "c2", "c3"])
        self.assertEqual(len(dense["distances"]), 3)

    def test_sparse_search_is_recorded_with_scores(self):
        sparse = self.trace["sparse"]
        self.assertEqual(sparse["ids"], ["c2", "c1", "c3"])
        self.assertEqual(sparse["scores"][0], 5.5)

    def test_embedded_query_differs_from_the_bm25_query(self):
        # The dense side embeds a prefixed instruction string; BM25 gets the
        # raw words. Showing both is the point -- users are surprised by this.
        texts = self.trace["query_texts"]
        self.assertEqual(texts["raw"], "What is one-shot prompting?")
        self.assertEqual(texts["bm25"], "What is one-shot prompting?")
        self.assertIn("one-shot prompting", texts["embedded"])

    def test_every_stage_is_present(self):
        names = [s["name"] for s in self.trace["stages"]]
        for expected in ("embed_query", "dense_search", "sparse_search",
                         "fuse_rrf", "generate"):
            self.assertIn(expected, names)

    def test_stages_carry_bilingual_labels_and_notes(self):
        for stage in self.trace["stages"]:
            self.assertTrue(stage["label"]["en"])
            self.assertTrue(stage["label"]["fa"])
            self.assertIn("note", stage)

    def test_funnel_counts_are_present(self):
        counts = {step["key"]: step["count"] for step in self.trace["funnel"]}
        self.assertEqual(counts["dense"], 3)
        self.assertEqual(counts["sparse"], 3)
        self.assertEqual(counts["used"], 2)


class TestWhatReachedTheModel(unittest.TestCase):
    def test_only_max_chunks_are_marked_as_used(self):
        answerer, *_ = build_answerer(max_chunks=2)
        _, trace = trace_answer(answerer, "What is one-shot prompting?")

        used = [c for c in trace["candidates"] if c["used_in_prompt"]]
        dropped = [c for c in trace["candidates"] if not c["used_in_prompt"]]
        self.assertEqual(len(used), 2)
        self.assertEqual(len(dropped), 1)
        self.assertEqual([c["citation_number"] for c in used], [1, 2])
        self.assertIsNone(dropped[0]["citation_number"])
        self.assertEqual(dropped[0]["chars_sent"], 0)

    def test_dropped_chunks_produce_a_warning(self):
        answerer, *_ = build_answerer(max_chunks=1)
        _, trace = trace_answer(answerer, "What is one-shot prompting?")
        codes = {w["code"] for w in trace["warnings"]}
        self.assertIn("dropped", codes)

    def test_the_recorded_prompt_is_the_prompt_that_was_sent(self):
        answerer, _, _, _, llm = build_answerer()
        _, trace = trace_answer(answerer, "What is one-shot prompting?")

        sent_system, sent_user = llm.calls[0]
        self.assertEqual(trace["prompt"]["system"], sent_system)
        self.assertEqual(trace["prompt"]["user"], sent_user)
        self.assertIn(trace["prompt"]["context_block"], sent_user)

    def test_truncated_chunks_are_flagged(self):
        answerer, *_ = build_answerer(max_chars_per_chunk=20)
        _, trace = trace_answer(answerer, "What is one-shot prompting?")

        used = [c for c in trace["candidates"] if c["used_in_prompt"]]
        self.assertTrue(all(c["truncated_in_prompt"] for c in used))
        self.assertTrue(all(c["chars_sent"] == 20 for c in used))
        self.assertIn("truncated", {w["code"] for w in trace["warnings"]})

    def test_context_window_usage_is_reported(self):
        answerer, *_ = build_answerer()
        _, trace = trace_answer(answerer, "What is one-shot prompting?")
        self.assertEqual(trace["prompt"]["num_ctx"], 8192)
        self.assertIsNotNone(trace["prompt"]["context_window_used_pct"])
        self.assertGreater(trace["prompt"]["tokens_estimate"], 0)


class TestCitations(unittest.TestCase):
    def test_cited_chunks_are_marked(self):
        answerer, *_ = build_answerer(llm=FakeLlm(answer="Because of [1]."))
        _, trace = trace_answer(answerer, "What is one-shot prompting?")

        used = [c for c in trace["candidates"] if c["used_in_prompt"]]
        self.assertTrue(used[0]["cited_in_answer"])
        self.assertFalse(used[1]["cited_in_answer"])
        self.assertEqual(trace["answer"]["citations"], [1])
        self.assertIn("uncited", {w["code"] for w in trace["warnings"]})

    def test_a_citation_that_did_not_exist_is_called_out(self):
        answerer, *_ = build_answerer(llm=FakeLlm(answer="As shown in [9]."))
        _, trace = trace_answer(answerer, "What is one-shot prompting?")
        self.assertIn("invalid_citation", {w["code"] for w in trace["warnings"]})


class TestReranking(unittest.TestCase):
    def test_rerank_stage_and_rank_movement_are_recorded(self):
        answerer, *_ = build_answerer(reranker=FakeReranker(), max_chunks=3)
        _, trace = trace_answer(answerer, "What is one-shot prompting?")

        names = [s["name"] for s in trace["stages"]]
        self.assertIn("rerank", names)

        # The fake reverses the order, so the first fused chunk must have
        # fallen and the last must have climbed.
        movements = {c["chunk_id"]: c["rank_change"] for c in trace["candidates"]}
        self.assertTrue(any(v and v > 0 for v in movements.values()))
        self.assertTrue(any(v and v < 0 for v in movements.values()))
        self.assertTrue(all(
            c["rerank_score"] is not None for c in trace["candidates"]
        ))


class TestEmptyAndFilteredCases(unittest.TestCase):
    def test_no_results_is_reported_clearly(self):
        answerer, *_ = build_answerer(docs=[], bm25_hits=[])
        result, trace = trace_answer(answerer, "something absent")

        self.assertEqual(trace["candidates"], [])
        codes = {w["code"] for w in trace["warnings"]}
        self.assertIn("nothing_sent", codes)
        self.assertIn("dense_empty", codes)
        self.assertIn("sparse_empty", codes)
        self.assertTrue(result.answer)

    def test_language_filter_is_recorded_when_enabled(self):
        answerer, _, chroma, _, _ = build_answerer(language_filter=True)
        _, trace = trace_answer(answerer, "What is one-shot prompting?")

        applied = chroma.query_calls[0]["where"]
        self.assertEqual(trace["dense"]["where"], applied)
        self.assertEqual(trace["settings"]["where_filter"], applied)

    def test_language_filter_off_records_no_filter(self):
        answerer, *_ = build_answerer(language_filter=False)
        _, trace = trace_answer(answerer, "What is one-shot prompting?")
        self.assertIsNone(trace["dense"]["where"])


class TestSettingsSnapshot(unittest.TestCase):
    def test_settings_describe_the_run(self):
        answerer, *_ = build_answerer(max_chunks=2, max_chars_per_chunk=1500)
        _, trace = trace_answer(answerer, "What is one-shot prompting?")

        settings = trace["settings"]
        self.assertEqual(settings["embedding_model"], "fake-embed")
        self.assertEqual(settings["generation_model"], "fake-llm")
        self.assertEqual(settings["max_chunks"], 2)
        self.assertEqual(settings["max_chars_per_chunk"], 1500)
        self.assertEqual(settings["dense_n_results"], 3)
        self.assertFalse(settings["rerank_enabled"])

    def test_tracer_can_be_reused_via_the_class_api(self):
        answerer, *_ = build_answerer()
        tracer = QueryTracer(answerer)
        result, trace = tracer.run("What is one-shot prompting?")
        self.assertTrue(result.answer)
        self.assertEqual(trace["question"], "What is one-shot prompting?")


class TestDemoTrace(unittest.TestCase):
    """The demo trace is what users see before Ollama is running, so its
    shape must match a real trace exactly -- otherwise the UI would work in
    the demo and break on the first real question."""

    def setUp(self):
        self.demo = demo_trace()

    def test_demo_is_labelled_as_a_demo(self):
        self.assertTrue(self.demo["demo"])

    def test_demo_has_the_same_keys_as_a_real_trace(self):
        answerer, *_ = build_answerer()
        _, real = trace_answer(answerer, "What is one-shot prompting?")
        for key in real:
            self.assertIn(key, self.demo, f"demo trace is missing '{key}'")

    def test_demo_candidates_have_the_same_fields_as_real_ones(self):
        answerer, *_ = build_answerer()
        _, real = trace_answer(answerer, "What is one-shot prompting?")
        real_fields = set(real["candidates"][0])
        for candidate in self.demo["candidates"]:
            missing = real_fields - set(candidate)
            self.assertEqual(missing, set(), f"demo candidate missing {missing}")

    def test_demo_is_internally_consistent(self):
        used = [c for c in self.demo["candidates"] if c["used_in_prompt"]]
        counts = {step["key"]: step["count"] for step in self.demo["funnel"]}
        self.assertEqual(counts["used"], len(used))
        self.assertEqual(
            counts["cited"], len([c for c in used if c["cited_in_answer"]])
        )
        for candidate in used:
            self.assertIn(f"[{candidate['citation_number']}]",
                          self.demo["prompt"]["context_block"])


if __name__ == "__main__":
    unittest.main()
