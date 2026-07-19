"""
tests/run_tests.py — Unified test runner.

Run all test suites and print a grouped summary.

Usage:
    python Chunker/tests/run_tests.py
    python Chunker/tests/run_tests.py -v          # verbose
    python Chunker/tests/run_tests.py --fast      # skip real-file tests
"""

import sys
import argparse
import unittest
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SUITES = [
    ("Preprocessor",       "Chunker.tests.test_md_preprocessor"),
    ("Language Detector",  "Chunker.tests.test_language_detector"),
    ("Structure Splitter", "Chunker.tests.test_structure_splitter"),
    ("Token Counter",      "Chunker.tests.test_token_counter"),
    ("Semantic Chunker",   "Chunker.tests.test_semantic_chunker"),
    ("LLM Chunk Refiner",  "Chunker.tests.test_llm_chunk_refiner"),
    ("Table Normalizer",   "Chunker.tests.test_table_normalizer"),
    ("Chunk Normalizer",   "Chunker.tests.test_chunk_normalizer"),
    ("Contextual Enricher","Chunker.tests.test_contextual_enricher"),
    ("Embed Text Builder", "Chunker.tests.test_embed_text_builder"),
    ("Embed Generator",    "Chunker.tests.test_embed_generator"),
    ("Chroma Ingestion",   "Chunker.tests.test_chroma_ingestion"),
    ("Run Ingestion CLI",  "Chunker.tests.test_run_ingestion"),
    ("BM25 Index",         "Chunker.tests.test_bm25_index"),
    ("Run BM25 CLI",       "Chunker.tests.test_run_bm25_index"),
    ("Hybrid Retriever",   "Chunker.tests.test_hybrid_retriever"),
    ("Run Query CLI",      "Chunker.tests.test_run_query"),
    ("Reranker",           "Chunker.tests.test_reranker"),
    ("Context Builder",    "Chunker.tests.test_context_builder"),
    ("LLM Client",         "Chunker.tests.test_llm_client"),
    ("RAG Answerer",       "Chunker.tests.test_rag_answerer"),
    ("Run Answer CLI",     "Chunker.tests.test_run_answer"),
    ("Eval Dataset",       "Chunker.tests.test_eval_dataset"),
    ("RAGAS Evaluator",    "Chunker.tests.test_ragas_evaluator"),
    ("Run Evaluation CLI", "Chunker.tests.test_run_evaluation"),
    ("Metadata Enricher",  "Chunker.tests.test_metadata_enricher"),
    ("Pipeline",           "Chunker.tests.test_pipeline"),
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
