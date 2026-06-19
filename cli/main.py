"""Minimal deterministic CLI for the compiler."""

from __future__ import annotations

import argparse
import json
import sys

from rag_doc_compiler.errors import CompilerError
from rag_doc_compiler.pipeline import DocumentCompiler
from api.rag_service import RAGService
from api.batch_pipeline import BatchPipeline
from config.settings import load_model_settings


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

    query = subparsers.add_parser("query")
    query.add_argument("chunks_json")
    query.add_argument("query")
    query.add_argument("--profile", default="generic")
    query.add_argument("--store", default="faiss", choices=["faiss", "qdrant"])
    query.add_argument("--no-llm", action="store_true")
    query.add_argument("--json", action="store_true")

    full_run = subparsers.add_parser("full-run")
    full_run.add_argument("input_dir")
    full_run.add_argument("workspace_dir")
    full_run.add_argument("--query")
    full_run.add_argument("--skip-ocr", action="store_true")
    full_run.add_argument("--store", default="faiss", choices=["faiss", "qdrant"])
    full_run.add_argument("--no-llm", action="store_true")
    full_run.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compiler = DocumentCompiler()
    try:
        if args.command == "run":
            result = compiler.run_file(args.input, args.output_dir)
            print(json.dumps({
                "clean_markdown": str(result["markdown_path"]),
                "chunks": str(result["chunks_path"]),
                "json_index": str(result["index_path"]),
            }, sort_keys=True))
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
                "final_chunks": result["final_chunks"].to_dict(),
                "semantic_labels": [
                    {"chunk_id": chunk.chunk_id, "intent": chunk.intent, "content_type": chunk.content_type}
                    for chunk in result["chunks"]
                ],
            }, indent=2, sort_keys=True))
            return 0
        if args.command == "query":
            service = RAGService(
                profile=args.profile,
                store_type=args.store,
                settings=load_model_settings(),
                use_llm=not args.no_llm,
            )
            service.load_chunks(args.chunks_json)
            response = service.ask(args.query)
            payload = {
                "answer": response["answer"],
                "sources": response["sources"],
                "plan": response["plan"],
            }
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
            else:
                print(_render_query_response(payload))
            return 0
        if args.command == "full-run":
            result = BatchPipeline(settings=load_model_settings()).run(
                input_dir=args.input_dir,
                workspace_dir=args.workspace_dir,
                query=args.query,
                skip_ocr=args.skip_ocr,
                store_type=args.store,
                use_llm=not args.no_llm,
            )
            printable = {
                "raw_markdown_dir": result["raw_markdown_dir"],
                "compiled_outputs": result["compiled_outputs"],
                "corpus_chunks": result["corpus_chunks"],
                "chandra_processed": result["chandra_processed"],
            }
            if result["response"]:
                printable["answer"] = result["response"]["answer"]
                printable["sources"] = result["response"]["sources"]
            if args.json:
                print(json.dumps(printable, indent=2, ensure_ascii=False, sort_keys=True))
            else:
                print(_render_full_run_response(printable))
            return 0
    except (CompilerError, ValueError, RuntimeError, FileNotFoundError) as error:
        print(json.dumps({"valid": False, "error": type(error).__name__, "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    return 1

def _render_query_response(payload: dict) -> str:
    lines = ["ANSWER", payload["answer"], "", "SOURCES"]
    if not payload["sources"]:
        lines.append("- No sources returned.")
    for index, source in enumerate(payload["sources"], start=1):
        heading = " > ".join(source.get("heading_path", [])) or "Document"
        line_range = source.get("line_range")
        line_text = f" lines {line_range[0]}-{line_range[1]}" if line_range else ""
        lines.append(f"{index}. {heading}{line_text}")
        lines.append(f"   chunk: {source['chunk_id']}")
    return "\n".join(lines)

def _render_full_run_response(payload: dict) -> str:
    lines = ["PIPELINE COMPLETE", ""]
    lines.append(f"Raw OCR Markdown: {payload['raw_markdown_dir']}")
    lines.append(f"Corpus chunks: {payload['corpus_chunks']}")
    lines.append("")
    lines.append("Compiled documents:")
    for item in payload["compiled_outputs"]:
        lines.append(f"- Input: {item['input_markdown']}")
        lines.append(f"  Clean Markdown: {item['clean_markdown']}")
        lines.append(f"  Chunks: {item['chunks']}")
        lines.append(f"  Index: {item['index']}")
    if payload.get("answer"):
        lines.append("")
        lines.append(_render_query_response({
            "answer": payload["answer"],
            "sources": payload.get("sources", []),
            "plan": {},
        }))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
