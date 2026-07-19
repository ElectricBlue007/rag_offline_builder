#!/usr/bin/env python3
"""
run_ingestion.py — Entry point for embedding generation + ChromaDB ingestion.

Run AFTER run_pipeline.py has produced chunks/*.chunks.json files:

    (venv) python run_ingestion.py

This reads every *.chunks.json under --chunks-dir, embeds each chunk's
embed_text field (which already carries the embedding model's
instruction-prefix contract, if any — see EmbedTextConfig — and the
contextual situating blurb from earlier pipeline stages), and upserts
the results into a local persistent ChromaDB collection at --chroma-dir.

Safe to re-run: chunk_ids are deterministic (see
enrichers/metadata_enricher.py compute_chunk_id), so re-running after
adding new PDFs updates only what changed rather than duplicating
existing entries.

Optional CLI overrides:
    python run_ingestion.py --chunks-dir ./chunks --chroma-dir ./chroma_db \\
        --collection rag_docs --model qwen3-embedding:8b
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Chunker.config import EmbedGeneratorConfig, ChromaIngestionConfig
from Chunker.embedding.embed_generator import EmbedGenerator, load_all_chunks
from Chunker.storage.chroma_ingestion import ChromaIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunks.json files and ingest them into ChromaDB."
    )
    parser.add_argument(
        "--chunks-dir", "-c",
        default="./chunks",
        help="Directory containing *.chunks.json files (output of run_pipeline.py).",
    )
    parser.add_argument(
        "--chroma-dir", "-d",
        default="./chroma_db",
        help="Directory where ChromaDB persists its data.",
    )
    parser.add_argument(
        "--collection",
        default="rag_docs",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--model", "-m",
        default="qwen3-embedding:8b",
        help="Ollama embedding model tag — should match SemanticChunkerConfig.embedding_model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding generation (lower for tighter VRAM).",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama service address serving --model.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


def main():
    args = parse_args()
    chunks_dir = Path(args.chunks_dir)

    log(f"\n📂 Loading chunks from: {chunks_dir}", args.quiet)
    chunks = load_all_chunks(chunks_dir)

    if not chunks:
        print(f"⚠️  No chunks found under {chunks_dir}. Run run_pipeline.py first.")
        sys.exit(1)

    log(f"🔍 Found {len(chunks)} chunk(s) across all files\n", args.quiet)

    # ── Embed ────────────────────────────────────────────────────────────────
    t0 = time.time()
    embed_cfg = EmbedGeneratorConfig(
        model_name=args.model, batch_size=args.batch_size, host=args.host
    )
    generator = EmbedGenerator(embed_cfg)

    log(f"🧠 Embedding {len(chunks)} chunk(s) with {args.model}...", args.quiet)
    embedding_results = generator.embed_chunks(chunks)
    embed_elapsed = time.time() - t0
    log(f"   Done in {embed_elapsed:.1f}s\n", args.quiet)

    # ── Ingest ───────────────────────────────────────────────────────────────
    t0 = time.time()
    chroma_cfg = ChromaIngestionConfig(
        persist_directory=args.chroma_dir,
        collection_name=args.collection,
    )
    ingestor = ChromaIngestor(chroma_cfg)

    log(f"💾 Ingesting into ChromaDB at: {args.chroma_dir} "
        f"(collection: {args.collection})", args.quiet)
    summary = ingestor.ingest_chunks(chunks, embedding_results)
    ingest_elapsed = time.time() - t0

    log(f"\n{'='*55}", args.quiet)
    log(f"  INGESTION COMPLETE", args.quiet)
    log(f"  Chunks ingested       : {summary.ingested_count}", args.quiet)
    if summary.skipped_no_vector:
        log(f"  Skipped (no vector)   : {summary.skipped_no_vector}", args.quiet)
    log(f"  Collection size now   : {summary.collection_count_after}", args.quiet)
    log(f"  Embedding time        : {embed_elapsed:.1f}s", args.quiet)
    log(f"  Ingestion time        : {ingest_elapsed:.1f}s", args.quiet)
    log(f"{'='*55}\n", args.quiet)

    sys.exit(0)


if __name__ == "__main__":
    main()
