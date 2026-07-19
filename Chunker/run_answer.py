#!/usr/bin/env python3
"""
run_answer.py — Full RAG question-answering: retrieve + (optionally
rerank) + generate a final answer via a local Ollama LLM.

Run AFTER run_pipeline.py, run_ingestion.py, and run_bm25_index.py have
all been run at least once, and Ollama is running locally with a model
pulled (e.g. `ollama pull llama3.2`).

One-shot mode:
    python run_answer.py "What is a CMI?"

Interactive mode:
    python run_answer.py

With cross-encoder reranking for higher precision (requires
`pip install sentence-transformers`):
    python run_answer.py "What is a CMI?" --rerank

Optional CLI overrides:
    python run_answer.py "query" --llm-model llama3.2 --llm-host http://localhost:11434 \\
        --n-results 5 --rerank --no-sources
"""

import argparse
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
)
from Chunker.embedding.embed_generator import EmbedGenerator
from Chunker.storage.chroma_ingestion import ChromaIngestor
from Chunker.storage.bm25_index import BM25Index
from Chunker.retrieval.hybrid_retriever import HybridRetriever
from Chunker.retrieval.reranker import CrossEncoderReranker
from Chunker.generation.llm_client import LlmClient
from Chunker.generation.rag_answerer import RagAnswerer, RagAnswer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question and get a generated answer from your documents."
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Question text. Omit for an interactive prompt loop.",
    )
    # Retrieval sources
    parser.add_argument("--chroma-dir", default="./chroma_db")
    parser.add_argument("--collection", default="rag_docs")
    parser.add_argument("--bm25-index", default="./bm25_index.pkl")
    parser.add_argument("--model", default="qwen3-embedding:8b",
                        help="Ollama embedding model tag (must match what was used for ingestion).")
    parser.add_argument("--n-results", type=int, default=5,
                        help="How many chunks to feed into the LLM's context.")

    # Reranking (opt-in)
    parser.add_argument("--rerank", action="store_true",
                        help="Apply cross-encoder reranking before generation "
                             "(requires sentence-transformers; heavier, more precise).")
    parser.add_argument("--rerank-model",
                        default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--candidate-pool", type=int, default=10,
                        help="How many candidates to retrieve before reranking "
                             "narrows down to --n-results. Only used with --rerank.")

    # LLM generation
    parser.add_argument("--llm-model", default="llama3.2")
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--no-sources", action="store_true",
                        help="Don't print the source list after the answer.")
    return parser.parse_args()


def build_retriever(args: argparse.Namespace) -> HybridRetriever:
    embedder = EmbedGenerator(EmbedGeneratorConfig(model_name=args.model))
    chroma = ChromaIngestor(ChromaIngestionConfig(
        persist_directory=args.chroma_dir, collection_name=args.collection
    ))
    bm25 = BM25Index(BM25IndexConfig(persist_path=args.bm25_index))
    bm25.load()
    final_n = args.candidate_pool if args.rerank else args.n_results
    retriever_cfg = HybridRetrieverConfig(final_n_results=final_n)
    return HybridRetriever(chroma, bm25, embedder, retriever_cfg)


def build_reranker(args: argparse.Namespace) -> CrossEncoderReranker:
    cfg = RerankerConfig(model_name=args.rerank_model, final_n_results=args.n_results)
    return CrossEncoderReranker(cfg)


def build_answerer(args: argparse.Namespace) -> RagAnswerer:
    retriever = build_retriever(args)
    reranker = build_reranker(args) if args.rerank else None
    llm_client = LlmClient(LlmClientConfig(
        model=args.llm_model, host=args.llm_host, temperature=args.temperature,
    ))
    context_cfg = ContextBuilderConfig(max_chunks=args.n_results)
    return RagAnswerer(retriever, llm_client, reranker=reranker, context_config=context_cfg)


def print_answer(result: RagAnswer, show_sources: bool) -> None:
    print(f"\n{result.answer}")
    if show_sources and result.sources:
        print("\nSources:")
        # Deliberately NOT deduplicated, and numbered starting at 1 in the
        # exact same order context_builder.py assigned — this is what makes
        # a bracketed "[2]" in the answer text actually mean something: it
        # points at Sources line [2] below, not some collapsed/reordered
        # version of it.
        for i, s in enumerate(result.sources, start=1):
            label = s["source_file"]
            if s.get("breadcrumb") and s["breadcrumb"] != "(root)":
                label += f" — {s['breadcrumb']}"
            pages = s.get("page_numbers") or []
            if pages:
                page_str = ",".join(str(p) for p in pages)
                label += f" (page {page_str})"
            print(f"  [{i}] {label}")


def main():
    args = parse_args()

    try:
        answerer = build_answerer(args)
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("Run run_bm25_index.py first to build the BM25 index.")
        sys.exit(1)

    if args.query:
        result = answerer.answer(args.query)
        print_answer(result, show_sources=not args.no_sources)
    else:
        print("Interactive mode — type a question, or 'exit'/'quit' to stop.")
        print("(Make sure Ollama is running locally with your chosen model pulled.)")
        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue
            result = answerer.answer(query)
            print_answer(result, show_sources=not args.no_sources)


if __name__ == "__main__":
    main()
