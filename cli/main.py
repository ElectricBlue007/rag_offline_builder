"""Comprehensive CLI for RAG Document Compiler with Phase 1 & Phase 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_doc_compiler.errors import CompilerError
from rag_doc_compiler.pipeline import DocumentCompiler
from agents.orchestrator import DocumentQualificationEngine as QualificationOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-compiler",
        description="RAG Document Compiler: Phase 1 (Compilation) + Phase 2 (Qualification)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Phase 1: Run full compilation pipeline
    run = subparsers.add_parser("run", help="Run full Phase 1 compilation pipeline")
    run.add_argument("input", type=Path, help="Input markdown file")
    run.add_argument("output_dir", type=Path, help="Output directory")
    run.add_argument("--repair-chunks", action="store_true", help="Use an LLM to repair reviewed chunks before final output")
    run.add_argument("--repair-model", default="qwen2.5:7b", help="Local Ollama model used when --repair-chunks is enabled")
    run.add_argument("--repair-decisions", default="REPAIR,RECHUNK", help="Comma-separated decisions to repair")
    run.add_argument("--repair-max-attempts", type=int, default=2, help="Maximum LLM attempts per chunk repair")
    run.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    # Phase 1: Validate only
    validate = subparsers.add_parser("validate", help="Validate markdown structure (Phase 1)")
    validate.add_argument("input", type=Path, help="Input markdown file")
    validate.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    # Phase 1: Debug mode with full details
    debug = subparsers.add_parser("debug", help="Debug mode with detailed structure analysis")
    debug.add_argument("input", type=Path, help="Input markdown file")
    debug.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    # Phase 2: Qualify document for chunking/embedding/retrieval
    qualify = subparsers.add_parser("qualify", help="Run Phase 2 qualification engine")
    qualify.add_argument("input", type=Path, help="Input markdown file")
    qualify.add_argument("--output", type=Path, help="Optional output file for results")
    qualify.add_argument("--detailed", action="store_true", help="Include detailed agent reports")
    qualify.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    # Run both Phase 1 and Phase 2
    full = subparsers.add_parser("full", help="Run both Phase 1 and Phase 2 pipelines")
    full.add_argument("input", type=Path, help="Input markdown file")
    full.add_argument("output_dir", type=Path, help="Output directory")
    full.add_argument("--detailed", action="store_true", help="Include detailed qualification reports")
    full.add_argument("--repair-chunks", action="store_true", help="Use an LLM to repair reviewed chunks before final output")
    full.add_argument("--repair-model", default="qwen2.5:7b", help="Local Ollama model used when --repair-chunks is enabled")
    full.add_argument("--repair-decisions", default="REPAIR,RECHUNK", help="Comma-separated decisions to repair")
    full.add_argument("--repair-max-attempts", type=int, default=2, help="Maximum LLM attempts per chunk repair")
    full.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    # Run tests
    test = subparsers.add_parser("test", help="Run test suite")
    test.add_argument("--phase", choices=["1", "2", "all"], default="all", help="Which phase tests to run")
    test.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return parser


def run_phase1(compiler: DocumentCompiler, input_path: Path, output_dir: Path | None = None) -> dict:
    """Execute Phase 1 compilation pipeline."""
    if output_dir:
        result = compiler.run_file(input_path, output_dir)
        output = {
            "phase": 1,
            "status": "success",
            "markdown_path": str(result["markdown_path"]),
            "index_path": str(result["index_path"]),
            "chunk_review_path": str(result["chunk_review_path"]),
            "chunk_count": len(result["chunks"]),
        }
        if "chunk_repair_path" in result:
            output["chunk_repair_path"] = str(result["chunk_repair_path"])
        return output
    else:
        result = compiler.compile_text(compiler.loader.load_text(input_path))
        return {
            "phase": 1,
            "status": "success",
            "chunk_count": len(result["chunks"]),
            "block_count": len(result["analysis"].blocks),
        }


def run_phase2(input_path: Path, detailed: bool = False) -> dict:
    """Execute Phase 2 qualification engine."""
    compiler = DocumentCompiler()
    compiled = compiler.compile_text(compiler.loader.load_text(input_path))
    
    orchestrator = QualificationOrchestrator()
    qualification = orchestrator.qualify(compiled["analysis"])
    
    result = {
        "phase": 2,
        "status": "success",
        "is_chunkable": qualification.is_chunkable,
        "is_embeddable": qualification.is_embeddable,
        "is_retrievable": qualification.is_retrievable,
        "overall_score": qualification.overall_score,
        "issue_count": len(qualification.issues),
        "warning_count": len(qualification.warnings),
        "recommendation_count": len(qualification.recommendations),
        "chunking_strategy": qualification.chunking_strategy.to_dict() if qualification.chunking_strategy else None,
    }
    
    if detailed:
        result["structural_analysis"] = qualification.structural_analysis
        result["semantic_analysis"] = qualification.semantic_analysis
        result["retrieval_analysis"] = qualification.retrieval_analysis
        result["embedding_analysis"] = qualification.embedding_analysis
    
    return result


def build_compiler_from_args(args: argparse.Namespace) -> DocumentCompiler:
    """Create a compiler with optional chunk repair settings from CLI args."""
    repair_decisions = [
        decision.strip().upper()
        for decision in getattr(args, "repair_decisions", "").split(",")
        if decision.strip()
    ]
    return DocumentCompiler(
        repair_chunks=getattr(args, "repair_chunks", False),
        repair_model=getattr(args, "repair_model", "qwen2.5:7b"),
        repair_decisions=repair_decisions or None,
        repair_max_attempts=getattr(args, "repair_max_attempts", 2),
    )


def run_tests(phase: str = "all", verbose: bool = False) -> dict:
    """Run test suite."""
    import subprocess
    
    test_args = ["python", "-m", "pytest", "tests/"]
    
    if phase == "1":
        test_args = ["python", "-m", "pytest", "tests/", "-k", "not phase2"]
    elif phase == "2":
        test_args = ["python", "-m", "pytest", "tests/test_phase2_qualification.py"]
    
    if verbose:
        test_args.append("-v")
    
    result = subprocess.run(test_args, capture_output=True, text=True, cwd="/workspace")
    
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    
    indent = 2 if getattr(args, "pretty", False) else None
    
    try:
        if args.command == "run":
            compiler = build_compiler_from_args(args)
            result = run_phase1(compiler, args.input, args.output_dir)
            print(json.dumps(result, indent=indent, sort_keys=True))
            return 0
            
        elif args.command == "validate":
            compiler = DocumentCompiler()
            result = compiler.validate_file(args.input)
            result["phase"] = 1
            print(json.dumps(result, indent=indent, sort_keys=True))
            return 0
            
        elif args.command == "debug":
            compiler = DocumentCompiler()
            markdown = compiler.loader.load_text(args.input)
            result = compiler.compile_text(markdown)
            output = {
                "phase": 1,
                "structure_tree": result["analysis"].tree.to_dict(),
                "block_count": len(result["analysis"].blocks),
                "chunk_candidates": [chunk.to_dict() for chunk in result["chunks"]],
                "semantic_labels": [
                    {"chunk_id": chunk.chunk_id, "intent": chunk.intent, "content_type": chunk.content_type}
                    for chunk in result["chunks"]
                ],
            }
            print(json.dumps(output, indent=indent, sort_keys=True))
            return 0
            
        elif args.command == "qualify":
            result = run_phase2(args.input, detailed=args.detailed)
            
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=indent, sort_keys=True)
                print(f"Results saved to {args.output}", file=sys.stderr)
            
            print(json.dumps(result, indent=indent, sort_keys=True))
            return 0
            
        elif args.command == "full":
            compiler = build_compiler_from_args(args)
            phase1_result = run_phase1(compiler, args.input, args.output_dir)
            phase2_result = run_phase2(args.input, detailed=args.detailed)
            
            result = {
                "phases_completed": [1, 2],
                "status": "success",
                "phase1": phase1_result,
                "phase2": phase2_result,
            }
            
            print(json.dumps(result, indent=indent, sort_keys=True))
            return 0
            
        elif args.command == "test":
            result = run_tests(phase=args.phase, verbose=args.verbose)
            print(json.dumps(result, indent=indent, sort_keys=True))
            return result["exit_code"]
            
    except CompilerError as error:
        print(
            json.dumps(
                {"valid": False, "error": type(error).__name__, "message": str(error)}, 
                sort_keys=True
            ), 
            file=sys.stderr
        )
        return 2
    except Exception as error:
        print(
            json.dumps(
                {"valid": False, "error": type(error).__name__, "message": str(error)}, 
                sort_keys=True
            ), 
            file=sys.stderr
        )
        return 1
    
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
