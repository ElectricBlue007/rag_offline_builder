#!/usr/bin/env python3
"""
run_pipeline.py — Entry point.

Place this file alongside your  input/  (or outputs/) directory and run:

    (venv) python run_pipeline.py

All configuration lives in Chunker/config.py — edit there, not here.

Optional CLI overrides:
    python run_pipeline.py --input ./outputs --output ./chunks --verbose
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
# Path(__file__).parent      = .../Chunker/          (the package dir)
# Path(__file__).parent.parent = .../project_root/        (what we need)
sys.path.insert(0, str(Path(__file__).parent.parent))

from Chunker.config import PipelineConfig
from Chunker.pipeline import MarkdownChunkingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic chunking pipeline for Chandra OCR markdown output."
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Root directory containing .md files (default: config value).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Directory where chunk JSON files will be written (default: config value).",
    )
    parser.add_argument(
        "--embedding-model", "-e",
        default=None,
        help="FastEmbed model name (default: BAAI/bge-m3).",
    )
    parser.add_argument(
        "--threshold-type", "-t",
        choices=["percentile", "standard_deviation", "interquartile"],
        default=None,
        help="Semantic breakpoint strategy (default: percentile).",
    )
    parser.add_argument(
        "--threshold-amount", "-a",
        type=float,
        default=None,
        help="Numeric value for the threshold (default: 85.0 for percentile).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Start from defaults and apply any CLI overrides
    cfg = PipelineConfig()

    if args.input:
        cfg.input_dir = Path(args.input)
    if args.output:
        cfg.output_dir = Path(args.output)
    if args.embedding_model:
        cfg.semantic.embedding_model = args.embedding_model
    if args.threshold_type:
        cfg.semantic.breakpoint_threshold_type = args.threshold_type
    if args.threshold_amount is not None:
        cfg.semantic.breakpoint_threshold_amount = args.threshold_amount
    if args.quiet:
        cfg.verbose = False

    pipeline = MarkdownChunkingPipeline(cfg)
    summary = pipeline.run()

    sys.exit(0 if not summary["failed"] else 1)


if __name__ == "__main__":
    main()