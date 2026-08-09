"""
tests/run_tests.py — Unified test runner.

Run all test suites and print a grouped summary.

Usage:
    python rag_pipeline/tests/run_tests.py
    python rag_pipeline/tests/run_tests.py -v          # verbose
    python rag_pipeline/tests/run_tests.py --fast      # skip real-file tests
"""

import sys
import argparse
import unittest
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SUITES = [
    ("Preprocessor",       "rag_pipeline.tests.test_md_preprocessor"),
    ("Language Detector",  "rag_pipeline.tests.test_language_detector"),
    ("Structure Splitter", "rag_pipeline.tests.test_structure_splitter"),
    ("Token Counter",      "rag_pipeline.tests.test_token_counter"),
    ("Semantic Chunker",   "rag_pipeline.tests.test_semantic_chunker"),
    ("LLM Chunk Refiner",  "rag_pipeline.tests.test_llm_chunk_refiner"),
    ("Table Normalizer",   "rag_pipeline.tests.test_table_normalizer"),
    ("Chunk Normalizer",   "rag_pipeline.tests.test_chunk_normalizer"),
    ("Contextual Enricher","rag_pipeline.tests.test_contextual_enricher"),
    ("Embed Text Builder", "rag_pipeline.tests.test_embed_text_builder"),
    ("Embed Generator",    "rag_pipeline.tests.test_embed_generator"),
    ("Chroma Ingestion",   "rag_pipeline.tests.test_chroma_ingestion"),
    ("Run Ingestion CLI",  "rag_pipeline.tests.test_run_ingestion"),
    ("BM25 Index",         "rag_pipeline.tests.test_bm25_index"),
    ("Run BM25 CLI",       "rag_pipeline.tests.test_run_bm25_index"),
    ("Hybrid Retriever",   "rag_pipeline.tests.test_hybrid_retriever"),
    ("Run Query CLI",      "rag_pipeline.tests.test_run_query"),
    ("Reranker",           "rag_pipeline.tests.test_reranker"),
    ("Context Builder",    "rag_pipeline.tests.test_context_builder"),
    ("LLM Client",         "rag_pipeline.tests.test_llm_client"),
    ("RAG Answerer",       "rag_pipeline.tests.test_rag_answerer"),
    ("Run Answer CLI",     "rag_pipeline.tests.test_run_answer"),
    ("Eval Dataset",       "rag_pipeline.tests.test_eval_dataset"),
    ("RAGAS Evaluator",    "rag_pipeline.tests.test_ragas_evaluator"),
    ("Run Evaluation CLI", "rag_pipeline.tests.test_run_evaluation"),
    ("Metadata Enricher",  "rag_pipeline.tests.test_metadata_enricher"),
    ("Pipeline",           "rag_pipeline.tests.test_pipeline"),
]

BAR = "═" * 65


def run_suite(label: str, module: str, verbosity: int) -> unittest.TestResult:
    print(f"\n{BAR}")
    print(f"  {label}")
    print(BAR)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromName(module)
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    return runner.run(suite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    verbosity = 2 if args.verbose else 1

    t0 = time.time()
    results = []

    for label, module in SUITES:
        result = run_suite(label, module, verbosity)
        results.append((label, result))

    # ── Grand summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    total_run = sum(r.testsRun for _, r in results)
    total_fail = sum(len(r.failures) for _, r in results)
    total_err  = sum(len(r.errors)   for _, r in results)
    total_skip = sum(len(r.skipped)  for _, r in results)

    print(f"\n\n{BAR}")
    print("  GRAND SUMMARY")
    print(BAR)
    for label, r in results:
        status = "✅ PASS" if not r.failures and not r.errors else "❌ FAIL"
        print(f"  {status}  {label:25s}  "
              f"run={r.testsRun:3d}  "
              f"fail={len(r.failures)}  "
              f"err={len(r.errors)}  "
              f"skip={len(r.skipped)}")

    print(BAR)
    print(f"  Total: {total_run} tests | "
          f"{total_fail} failures | {total_err} errors | "
          f"{total_skip} skipped | {elapsed:.1f}s")
    print(BAR)

    sys.exit(0 if total_fail == 0 and total_err == 0 else 1)


if __name__ == "__main__":
    main()
