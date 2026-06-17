"""Minimal deterministic CLI for the compiler."""

from __future__ import annotations

import argparse
import json
import sys

from rag_doc_compiler.errors import CompilerError
from rag_doc_compiler.pipeline import DocumentCompiler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("input")
    run.add_argument("output_dir")

    validate = subparsers.add_parser("validate")
    validate.add_argument("input")

    debug = subparsers.add_parser("debug")
    debug.add_argument("input")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compiler = DocumentCompiler()
    try:
        if args.command == "run":
            result = compiler.run_file(args.input, args.output_dir)
            print(json.dumps({"clean_markdown": str(result["markdown_path"]), "json_index": str(result["index_path"])}, sort_keys=True))
            return 0
        if args.command == "validate":
            print(json.dumps(compiler.validate_file(args.input), sort_keys=True))
            return 0
        if args.command == "debug":
            markdown = compiler.loader.load_text(args.input)
            result = compiler.compile_text(markdown)
            print(json.dumps({
                "structure_tree": result["analysis"].tree.to_dict(),
                "chunk_candidates": [chunk.to_dict() for chunk in result["chunks"]],
                "semantic_labels": [
                    {"chunk_id": chunk.chunk_id, "intent": chunk.intent, "content_type": chunk.content_type}
                    for chunk in result["chunks"]
                ],
            }, indent=2, sort_keys=True))
            return 0
    except CompilerError as error:
        print(json.dumps({"valid": False, "error": type(error).__name__, "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
