"""Central CLI router for the pipeline tools."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence


def _load(module_name: str):
    return importlib.import_module(f"scripts.cli.{module_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-offline-builder")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("health")
    subparsers.add_parser("extract-metadata")
    subparsers.add_parser("extract-content")
    subparsers.add_parser("extract-tables")
    subparsers.add_parser("extract-reading-order")
    subparsers.add_parser("extract-ocr")
    subparsers.add_parser("extract-images")  
    subparsers.add_parser("generate-markdown")
    subparsers.add_parser("enrich-markdown")
    subparsers.add_parser("chunk-markdown")
    subparsers.add_parser("run-pipeline")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed_argv = list(argv) if argv is not None else None
    args, remaining = parser.parse_known_args(parsed_argv)

    handlers = {
        "health": _load("health_check").main,
        "extract-metadata": _load("extract_metadata").main,
        "extract-content": _load("extract_content").main,
        "extract-tables": _load("extract_tables").main,
        "extract-reading-order": _load("extract_reading_order").main,
        "extract-ocr": _load("extract_ocr").main,
        "extract-images": _load("extract_images").main,  
        "generate-markdown": _load("generate_markdown").main,
        "enrich-markdown": _load("enrich_markdown").main,
        "chunk-markdown": _load("chunk_markdown").main,
        "run-pipeline": _load("run_pipeline").main,
    }

    if args.command not in handlers:
        parser.print_help()
        return 2

    if args.command == "health":
        return handlers[args.command]()
    return handlers[args.command](remaining)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
