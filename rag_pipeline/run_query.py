#!/usr/bin/env python3
"""
run_query.py — Query the hybrid retriever (dense + sparse + RRF),
optionally with a cross-encoder reranking pass.

Run AFTER run_pipeline.py, run_ingestion.py, and run_bm25_index.py have
all been run at least once. One-shot mode:

    python run_query.py "What is a CMI?"

Or omit the query for an interactive prompt loop:

    python run_query.py

With cross-encoder reranking (requires `pip install sentence-transformers`,
which pulls in torch — a heavier dependency, hence opt-in rather than
default):

    python run_query.py "What is a CMI?" --rerank

Optional CLI overrides:
    python run_query.py "query" --chroma-dir ./chroma_db --bm25-index ./bm25_index.pkl \\
        --model qwen3-embedding8b:latest --n-results 5 \\
        --rerank --candidate-pool 15
"""

import argparse
from dataclasses import replace
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
)
from rag_pipeline.embedding.embed_generator import EmbedGenerator
from rag_pipeline import model_manager
from rag_pipeline.storage.chroma_ingestion import ChromaIngestor
from rag_pipeline.storage.bm25_index import BM25Index
from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from rag_pipeline.retrieval.reranker import CrossEncoderReranker, RerankedChunk
from rag_pipeline.document_reference import format_reference

_S = load_settings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the hybrid (dense + sparse) retriever, optionally reranked."
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Query text. Omit for an interactive prompt loop.",
    )
    parser.add_argument("--chroma-dir", default=_S.paths.chroma_dir)
    parser.add_argument("--collection", default=_S.paths.collection_name)
    parser.add_argument("--bm25-index", default=_S.paths.bm25_index_path)
    parser.add_argument("--model", default=_S.embed_generator.model_name)
    parser.add_argument("--n-results", type=int, default=_S.n_results,
                        help="Final number of results to display.")
    parser.add_argument("--rerank", action="store_true", default=_S.rerank_runtime.enabled,
                        help="Apply cross-encoder reranking after hybrid retrieval "
                             "(requires sentence-transformers; heavier, more precise).")
    parser.add_argument("--rerank-model",
                        default=_S.reranker.model_name)
    parser.add_argument("--candidate-pool", type=int, default=_S.rerank_runtime.candidate_pool,
                        help="How many candidates hybrid retrieval fetches before "
                             "reranking narrows down to --n-results. Only used "
                             "with --rerank.")
    return parser.parse_args()


def build_retriever(args: argparse.Namespace) -> HybridRetriever:
    embedder = EmbedGenerator(
        replace(_S.embed_generator, model_name=args.model),  # FIX(B2b): keeps batch_size
        runtime=_S.runtime,
    )
    chroma = ChromaIngestor(ChromaIngestionConfig(
        persist_directory=args.chroma_dir, collection_name=args.collection
    ))
    bm25 = BM25Index(BM25IndexConfig(persist_path=args.bm25_index))
    bm25.load()
    # When reranking, retrieval should return a WIDER pool than the
    # final display count — the reranker is what narrows it back down
    # to --n-results, using a more precise (but slower) judgment than
    # RRF alone.
    final_n = args.candidate_pool if args.rerank else args.n_results
    # FIX(B2): same bug class as B1 -- this used to drop dense_n_results,
    # sparse_n_results, rrf_k and language_filter from settings.yaml.
    retriever_cfg = replace(_S.hybrid_retriever, final_n_results=final_n)
    return HybridRetriever(chroma, bm25, embedder, retriever_cfg)


def build_reranker(args: argparse.Namespace) -> CrossEncoderReranker:
    # FIX(B1): start from settings.yaml's reranker block and override ONLY
    # what the CLI actually set. Rebuilding RerankerConfig() from scratch
    # silently dropped backend / ollama_model / ollama_host / task_instruction
    # / max_length / device / model_kwargs / huggingface_unload_after_use back
    # to their dataclass defaults -- so `backend: "ollama"` in settings.yaml
    # was ignored and a HuggingFace CrossEncoder was loaded instead.
    cfg = replace(
        _S.reranker,
        model_name=args.rerank_model,
        final_n_results=args.n_results,
    )
    return CrossEncoderReranker(cfg, runtime=_S.runtime)


def format_results(results: list) -> str:
    """
    Handles both RetrievedChunk (no reranking) and RerankedChunk
    (post-reranking) via duck-typing on rerank_score's presence, so
    the same formatter serves either code path without the caller
    needing to track which one is active.
    """
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, start=1):
        # Full document location (file, page, Chapter > Section) rather
        # than the bare breadcrumb, so a retrieval check can be verified
        # against the source document without opening the chunk JSON.
        # Derived at display time, so it also works for a chroma_db that
        # was ingested before these metadata fields existed.
        location = (
            r.metadata.get("reference")
            or format_reference(r.metadata)
            or r.metadata.get("source_file", "?")
        )
        preview = r.text[:200].replace("\n", " ")

        if hasattr(r, "rerank_score"):
            score_info = (
                f"rerank_score={r.rerank_score:.4f}  "
                f"(was rrf_score={r.original_rrf_score:.4f}, "
                f"original rank #{r.original_rank + 1})"
            )
        else:
            score_info = (
                f"score={r.rrf_score:.4f}  "
                f"(dense_rank={r.dense_rank}, sparse_rank={r.sparse_rank})"
            )

        lines.append(
            f"\n[{i}] {score_info}\n"
            f"    {location}\n"
            f"    {preview}..."
        )
    return "".join(lines)


def main():
    args = parse_args()

    try:
        retriever = build_retriever(args)
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("Run run_bm25_index.py first to build the BM25 index.")
        sys.exit(1)

    reranker = build_reranker(args) if args.rerank else None

    def run_query(query: str):
        results = retriever.retrieve(query)
        if reranker is not None:
            results = reranker.rerank(query, results)
        print(format_results(results))
        # No RagAnswerer here to do this automatically (there's no
        # generation step in this script) — explicitly release the
        # embedding model (and ollama-backend reranker model, if any)
        # after each query so nothing lingers in VRAM between queries
        # in interactive mode.
        model_manager.release_all(retriever.embedder.cfg.host, _S.runtime)
        if reranker is not None and getattr(reranker.cfg, "ollama_host", None):
            model_manager.release_all(reranker.cfg.ollama_host, _S.runtime)

    if args.query:
        run_query(args.query)
    else:
        print("Interactive mode — type a query, or 'exit'/'quit' to stop.")
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
            run_query(query)


if __name__ == "__main__":
    main()
