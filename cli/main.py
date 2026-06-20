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
    run.add_argument("--embeddings", action="store_true", help="Enable embedding generation for chunks")

    validate = subparsers.add_parser("validate")
    validate.add_argument("input")

    debug = subparsers.add_parser("debug")
    debug.add_argument("input")
    debug.add_argument("--embeddings", action="store_true", help="Include embeddings in debug output")
    
    download_model = subparsers.add_parser("download-model")
    download_model.add_argument("--force", action="store_true", help="Force redownload even if model exists")
    download_model.add_argument("--output-dir", type=str, default=None, help="Custom directory to store the model")
    
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    
    if args.command == "download-model":
        from pathlib import Path
        compiler = DocumentCompiler()
        try:
            model_path = compiler.download_embedding_model(force=args.force)
            print(json.dumps({
                "status": "success",
                "model_path": str(model_path),
                "message": f"Model cached at {model_path}"
            }, sort_keys=True))
            return 0
        except Exception as e:
            print(json.dumps({
                "status": "error",
                "message": str(e)
            }, sort_keys=True), file=sys.stderr)
            return 2
    
    compiler = DocumentCompiler(enable_embeddings=getattr(args, 'embeddings', False))
    try:
        if args.command == "run":
            result = compiler.run_file(args.input, args.output_dir)
            output = {"clean_markdown": str(result["markdown_path"]), "json_index": str(result["index_path"])}
            if "embedded_chunks" in result:
                output["embedded_chunks_count"] = len(result["embedded_chunks"])
            print(json.dumps(output, sort_keys=True))
            return 0
        if args.command == "validate":
            print(json.dumps(compiler.validate_file(args.input), sort_keys=True))
            return 0
        if args.command == "debug":
            markdown = compiler.loader.load_text(args.input)
            result = compiler.compile_text(markdown)
            debug_output = {
                "structure_tree": result["analysis"].tree.to_dict(),
                "chunk_candidates": [chunk.to_dict() for chunk in result["chunks"]],
                "semantic_labels": [
                    {"chunk_id": chunk.chunk_id, "intent": chunk.intent, "content_type": chunk.content_type}
                    for chunk in result["chunks"]
                ],
            }
            if "embedded_chunks" in result:
                debug_output["embedded_chunks"] = [ec.to_dict() for ec in result["embedded_chunks"]]
            print(json.dumps(debug_output, indent=2, sort_keys=True))
            return 0
    except CompilerError as error:
        print(json.dumps({"valid": False, "error": type(error).__name__, "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
