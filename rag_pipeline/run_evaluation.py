#!/usr/bin/env python3
"""
run_evaluation.py — Measure whether the whole RAG pipeline (retrieval
+ generation) is actually working, using RAGAS's four core metrics.

Run AFTER run_pipeline.py, run_ingestion.py, and run_bm25_index.py have
all been run at least once, and Ollama is running locally.

    python run_evaluation.py

Uses evaluation/sample_eval_questions.json by default — a 12-question
starter set built from this project's own fixture documents. Expand it
(or point --questions at your own file) with real questions from your
actual documents for a meaningful evaluation; 12 questions is a
demonstration, not a robust sample size (the original evaluation plan
calls for 20-30).

Fully offline: uses Ollama as both RAGAS's judge LLM and its
embedding-based metrics — never OpenAI, which RAGAS defaults to.

Optional CLI overrides:
    python run_evaluation.py --questions ./my_eval_set.json \\
        --metrics faithfulness,answer_relevancy --rerank \\
        --save-results ./eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.settings import load_settings
from rag_pipeline.config import (
    EmbedGeneratorConfig,
    ChromaIngestionConfig,
    BM25IndexConfig,
    HybridRetrieverConfig,
    RerankerConfig,
    LlmClientConfig,
    ContextBuilderConfig,
    RagasEvaluatorConfig,
)
from rag_pipeline.embedding.embed_generator import EmbedGenerator
from rag_pipeline.storage.chroma_ingestion import ChromaIngestor
from rag_pipeline.storage.bm25_index import BM25Index
from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever
from rag_pipeline.retrieval.reranker import CrossEncoderReranker
from rag_pipeline.generation.llm_client import LlmClient
from rag_pipeline.generation.rag_answerer import RagAnswerer
from rag_pipeline.evaluation.eval_dataset import load_eval_questions
from rag_pipeline.evaluation.ragas_evaluator import RagasEvaluator

_S = load_settings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline's retrieval + generation quality with RAGAS."
    )
    parser.add_argument("--questions", default=_S.evaluation_questions_path,
                        help="Path to a JSON eval question set (see evaluation/eval_dataset.py). "
                             "Default: settings.yaml evaluation.questions_path.")

    # Retrieval sources (same as run_answer.py)
    parser.add_argument("--chroma-dir", default=_S.paths.chroma_dir)
    parser.add_argument("--collection", default=_S.paths.collection_name)
    parser.add_argument("--bm25-index", default=_S.paths.bm25_index_path)
    parser.add_argument("--model", default=_S.embed_generator.model_name)
    parser.add_argument("--n-results", type=int, default=_S.n_results)
    parser.add_argument("--rerank", action="store_true", default=_S.rerank_runtime.enabled)
    parser.add_argument("--rerank-model", default=_S.reranker.model_name)
    parser.add_argument("--candidate-pool", type=int, default=_S.rerank_runtime.candidate_pool)

    # Answer generation
    parser.add_argument("--llm-model", default=_S.llm_client.model)
    parser.add_argument("--llm-host", default=_S.llm_client.host)

    # RAGAS judge (can differ from the answer-generation LLM above)
    parser.add_argument("--judge-model", default=_S.ragas_evaluator.judge_llm_model)
    parser.add_argument("--judge-host", default=_S.ragas_evaluator.judge_llm_host)
    parser.add_argument("--metrics", default=",".join(_S.ragas_evaluator.metrics),
                        help="Comma-separated subset of the four RAGAS metrics to compute. "
                             "Default: settings.yaml evaluation.metrics.")

    parser.add_argument("--save-results", default=_S.evaluation_save_results_path,
                        help="Optional path to save detailed per-question results as JSON. "
                             "Default: settings.yaml evaluation.save_results_path.")
    parser.add_argument("--quiet", "-q", action="store_true")
    return parser.parse_args()


def log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


def build_answerer(args: argparse.Namespace) -> RagAnswerer:
    embedder = EmbedGenerator(EmbedGeneratorConfig(model_name=args.model))
    chroma = ChromaIngestor(ChromaIngestionConfig(
        persist_directory=args.chroma_dir, collection_name=args.collection
    ))
    bm25 = BM25Index(BM25IndexConfig(persist_path=args.bm25_index))
    bm25.load()

    final_n = args.candidate_pool if args.rerank else args.n_results
    retriever = HybridRetriever(chroma, bm25, embedder, HybridRetrieverConfig(final_n_results=final_n))

    reranker = None
    if args.rerank:
        reranker = CrossEncoderReranker(
            RerankerConfig(model_name=args.rerank_model, final_n_results=args.n_results)
        )

    llm_client = LlmClient(LlmClientConfig(model=args.llm_model, host=args.llm_host))
    context_cfg = ContextBuilderConfig(
        include_type_label=_S.context_builder.include_type_label,
        include_source=_S.context_builder.include_source,
        include_breadcrumb=_S.context_builder.include_breadcrumb,
        max_chunks=args.n_results,
        max_chars_per_chunk=_S.context_builder.max_chars_per_chunk,
    )
    return RagAnswerer(retriever, llm_client, reranker=reranker, context_config=context_cfg)


def main():
    args = parse_args()

    log(f"\n📂 Loading eval questions from: {args.questions}", args.quiet)
    try:
        questions = load_eval_questions(args.questions)
    except FileNotFoundError:
        print(f"⚠️  Eval question file not found: {args.questions}")
        sys.exit(1)

    if not questions:
        print("⚠️  No questions found in the eval file.")
        sys.exit(1)

    log(f"🔍 Loaded {len(questions)} question(s)\n", args.quiet)

    try:
        answerer = build_answerer(args)
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("Run run_bm25_index.py first to build the BM25 index.")
        sys.exit(1)

    ragas_cfg = RagasEvaluatorConfig(
        judge_llm_model=args.judge_model,
        judge_llm_host=args.judge_host,
        embedding_model=args.model,
        embedding_host=_S.ragas_evaluator.embedding_host,
        metrics=[m.strip() for m in args.metrics.split(",") if m.strip()],
        context_recall_warning_threshold=_S.ragas_evaluator.context_recall_warning_threshold,
        faithfulness_warning_threshold=_S.ragas_evaluator.faithfulness_warning_threshold,
        judge_max_workers=_S.ragas_evaluator.judge_max_workers,
        judge_timeout_seconds=_S.ragas_evaluator.judge_timeout_seconds,
    )
    evaluator = RagasEvaluator(answerer, ragas_cfg)

    log("🧠 Running pipeline on each question (retrieval + generation)...", args.quiet)
    records = evaluator.run_pipeline_on_questions(questions)
    log(f"   Done — {len(records)} question(s) answered\n", args.quiet)

    log("📊 Scoring with RAGAS (this calls the judge LLM — may take a while)...", args.quiet)
    try:
        summary = evaluator.score(records)
    except ImportError as e:
        print(f"⚠️  {e}")
        sys.exit(1)

    summary.print_summary(ragas_cfg)

    if args.save_results:
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"metric_means": summary.metric_means, "records": records},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"\n💾 Detailed results saved to: {out_path}", args.quiet)

    sys.exit(0)


if __name__ == "__main__":
    main()
