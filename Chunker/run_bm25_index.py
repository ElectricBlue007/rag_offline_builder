#!/usr/bin/env python3
"""
run_bm25_index.py — Entry point for building the BM25 keyword index.

Run AFTER run_pipeline.py has produced chunks/*.chunks.json files:

    (venv) python run_bm25_index.py

Complements ChromaDB's dense vector search (run_ingestion.py) with
exact keyword matching — essential for technical documents full of
abbreviations (CMI, MAC, GCI...) that embedding models under-match on.
The future hybrid retriever (roadmap step 6) combines both via
Reciprocal Rank Fusion.

Rebuilding is fast (seconds even for hundreds of documents), so it's
fine to re-run this after any chunking/config change rather than
trying to incrementally update the index.

Optional CLI overrides:
    python run_bm25_index.py --chunks-dir ./chunks --output ./bm25_index.pkl
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Chunker.config import BM25IndexConfig
from Chunker.embedding.embed_generator import load_all_chunks
from Chunker.storage.bm25_index import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a BM25 keyword index from chunks.json files."
    )
    parser.add_argument(
        "--chunks-dir", "-c",
        default="./chunks",
        help="Directory containing *.chunks.json files (output of run_pipeline.py).",
    )
    parser.add_argument(
        "--output", "-o",
        default="./bm25_index.pkl",
        help="Where to write the persisted BM25 index.",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 hyperparameter (term frequency saturation).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b hyperparameter (document length normalization).",
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

    t0 = time.time()
    cfg = BM25IndexConfig(persist_path=args.output, k1=args.k1, b=args.b)
    index = BM25Index(cfg)

    log(f"🔨 Building BM25 index (k1={args.k1}, b={args.b})...", args.quiet)
    index.build(chunks)
    build_elapsed = time.time() - t0

    index.save()

    log(f"\n{'='*55}", args.quiet)
    log(f"  BM25 INDEX BUILT", args.quiet)
    log(f"  Chunks indexed  : {index.size}", args.quiet)
    log(f"  Saved to        : {args.output}", args.quiet)
    log(f"  Build time      : {build_elapsed:.2f}s", args.quiet)
    log(f"{'='*55}\n", args.quiet)

    sys.exit(0)


if __name__ == "__main__":
    main()
