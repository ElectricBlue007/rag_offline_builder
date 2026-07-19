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

from Chunker.config import (
    EmbedGeneratorConfig,
    ChromaIngestionConfig,
    BM25IndexConfig,
    HybridRetrieverConfig,
    RerankerConfig,
    LlmClientConfig,
    ContextBuilderConfig,
    RagasEvaluatorConfig,
)
from Chunker.embedding.embed_generator import EmbedGenerator
from Chunker.storage.chroma_ingestion import ChromaIngestor
from Chunker.storage.bm25_index import BM25Index
from Chunker.retrieval.hybrid_retriever import HybridRetriever
from Chunker.retrieval.reranker import CrossEncoderReranker
from Chunker.generation.llm_client import LlmClient
from Chunker.generation.rag_answerer import RagAnswerer
from Chunker.evaluation.eval_dataset import load_eval_questions
from Chunker.evaluation.ragas_evaluator import RagasEvaluator


_DEFAULT_QUESTIONS_PATH = Path(__file__).parent / "evaluation" / "sample_eval_questions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline's retrieval + generation quality with RAGAS."
    )
    parser.add_argument("--questions", default=str(_DEFAULT_QUESTIONS_PATH),
                        help="Path to a JSON eval question set (see evaluation/eval_dataset.py).")

    # Retrieval sources (same as run_answer.py)
    parser.add_argument("--chroma-dir", default="./chroma_db")
    parser.add_argument("--collection", default="rag_docs")
    parser.add_argument("--bm25-index", default="./bm25_index.pkl")
    parser.add_argument("--model", default="qwen3-embedding:8b")
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--candidate-pool", type=int, default=10)

    # Answer generation
    parser.add_argument("--llm-model", default="llama3.2")
    parser.add_argument("--llm-host", default="http://localhost:11434")

    # RAGAS judge (can differ from the answer-generation LLM above)
    parser.add_argument("--judge-model", default="llama3.2")
    parser.add_argument("--judge-host", default="http://localhost:11434")
    parser.add_argument("--metrics", default="context_precision,context_recall,faithfulness,answer_relevancy",
                        help="Comma-separated subset of the four RAGAS metrics to compute.")

    parser.add_argument("--save-results", default=None,
                        help="Optional path to save detailed per-question results as JSON.")
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
    context_cfg = ContextBuilderConfig(max_chunks=args.n_results)
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
        metrics=[m.strip() for m in args.metrics.split(",") if m.strip()],
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
