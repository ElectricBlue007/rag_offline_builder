#!/usr/bin/env python3
"""
RAG Document Compiler CLI

A command-line interface for the RAG ingestion pipeline.
Supports Phase 1 (Structuring), Phase 2 (Qualification), and Self-Healing.

Usage:
    python -m rag_doc_compiler.cli process input.md --output result.json
    python -m rag_doc_compiler.cli validate input.md
    python -m rag_doc_compiler.cli stats input.md
    python -m rag_doc_compiler.cli heal input.md issues.json --output fixed.md
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Optional

# Ensure the project root is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.markdown_normalizer import MarkdownNormalizer
    from core.structure_analyzer import StructureAnalyzer
    from agents.orchestrator import DocumentQualificationEngine
    from agents.models import DocumentQualificationResult
    from agents.self_healing_agent import SelfHealingAgent
    
    # Create global instances for convenience
    _normalizer = MarkdownNormalizer()
    _analyzer = StructureAnalyzer()
    _qualification_engine = DocumentQualificationEngine()
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you are running this from the project root or have installed the package.")
    sys.exit(1)


def normalize_markdown(content: str) -> str:
    """Helper function to normalize markdown content."""
    result = _normalizer.normalize(content)
    return result.markdown


def analyze_structure(content: str):
    """Helper function to analyze markdown structure."""
    return _analyzer.analyze(content)


def qualify_document(document):
    """Helper function to qualify a document."""
    return _qualification_engine.qualify(document)


def load_markdown(file_path: str) -> str:
    """Load markdown content from a file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text(encoding="utf-8")


def save_json(data: dict, output_path: str):
    """Save data to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to: {output_path}")


def run_phase_1(content: str, verbose: bool = False) -> tuple[dict, any]:
    """Execute Phase 1: Normalization and Structuring.
    
    Returns:
        Tuple of (doc_dict, document_object) for use in Phase 2
    """
    if verbose:
        print("🚀 Starting Phase 1: Normalization & Structuring...")
    
    # Step 1: Normalize
    normalized_content = normalize_markdown(content)
    if verbose:
        print(f"   ✅ Normalized content length: {len(normalized_content)} chars")
    
    # Step 2: Analyze Structure
    analysis = analyze_structure(normalized_content)
    if verbose:
        print(f"   ✅ Detected {len(analysis.ordered_blocks)} blocks")
        block_types = {}
        for block in analysis.ordered_blocks:
            bt = block.node_type if hasattr(block, 'node_type') else 'unknown'
            block_types[bt] = block_types.get(bt, 0) + 1
        print(f"   📊 Block distribution: {block_types}")
    
    # Convert document to serializable dict
    doc_dict = {
        "metadata": {
            "total_blocks": len(analysis.ordered_blocks),
            "has_front_matter": hasattr(analysis, 'tree') and getattr(analysis.tree, 'front_matter', None) is not None,
        },
        "blocks": [
            {
                "block_id": b.node_id,
                "order": b.order if hasattr(b, 'order') else idx,
                "block_type": b.node_type if hasattr(b, 'node_type') else 'unknown',
                "content": b.content,
                "metadata": b.metadata if hasattr(b, 'metadata') else {},
                "level": b.level if hasattr(b, 'level') else None,
                "heading_path": b.heading_path if hasattr(b, 'heading_path') else None,
                "flags": []
            }
            for idx, b in enumerate(analysis.ordered_blocks)
        ]
    }
    return doc_dict, analysis


def run_phase_2(doc_dict: dict, verbose: bool = False) -> dict:
    """Execute Phase 2: Qualification and Strategy."""
    if verbose:
        print("🔍 Starting Phase 2: Qualification Engine...")
    
    # Reconstruct a simple object for the orchestrator if needed, 
    # but here we can pass the dict or a simplified model.
    # For this CLI, we will instantiate the orchestrator and pass the structured data.
    # Note: The orchestrator expects a Document object or similar. 
    # We'll adapt the call to match our existing implementation.
    
    orchestrator = QualificationOrchestrator()
    
    # Since our orchestrator expects a 'Document' object from Phase 1, 
    # and we serialized it, we need to either deserialize or modify the orchestrator.
    # To keep it simple without changing core logic too much, let's assume 
    # the user runs the pipeline in one go where the object is still in memory.
    # However, for CLI modularity, let's create a minimal mock or re-run analysis 
    # if only validation is requested.
    
    # *Correction*: The best way for CLI is to run Phase 1 then pass the object directly.
    # But since we serialized for JSON output, let's assume the 'run_full_pipeline' 
    # keeps the object in memory.
    
    # For the purpose of this CLI function being standalone, 
    # we will assume the caller (run_full_pipeline) passes the live object.
    # This function is designed to be called with the live Document object in the full flow.
    # But to support 'validate' command on a JSON file, we'd need a deserializer.
    # Let's simplify: This function will be called by run_full_pipeline with the live object.
    # If called standalone via CLI 'validate', we need a different approach.
    
    # Let's refactor slightly: The CLI 'process' command keeps the object in memory.
    # The 'validate' command would need to reconstruct it. 
    # For now, let's focus on the 'process' flow which is the primary use case.
    
    # Actually, let's make the Orchestrator accept the dict representation too 
    # or just re-instantiate the Document from the dict. 
    # To avoid over-engineering in the CLI, let's assume the 'process' command 
    # does everything in one memory session.
    
    # Wait, the function signature above takes doc_dict. 
    # Let's implement a quick reconstruction or pass the live object in the main flow.
    # I will adjust the main flow to pass the live object instead of the dict to this function.
    # And for 'validate' command, we will just say it requires the source file to re-process 
    # OR we implement a lightweight loader.
    
    # Let's stick to the strongest path: 
    # run_phase_2 should take the live 'document' object.
    # I will adjust the call in run_full_pipeline.
    # For now, this placeholder raises a reminder.
    raise NotImplementedError("Use run_full_pipeline for end-to-end processing.")


def run_full_pipeline(file_path: str, output: Optional[str] = None, verbose: bool = False):
    """Run Phase 1 and Phase 2 sequentially."""
    try:
        content = load_markdown(file_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Phase 1
    if verbose:
        print("\n" + "="*40)
    doc_dict, document_obj = run_phase_1(content, verbose=verbose)
    
    # Phase 2
    result = qualify_document(document_obj)
    
    if verbose:
        print(f"   ✅ Qualification Score: {result.overall_score:.2f}")
        print(f"   ✅ Chunkable: {result.is_chunkable}")
        print(f"   ✅ Embeddable: {result.is_embeddable}")
        print(f"   ✅ Retrievable: {result.is_retrievable}")
        
        if result.issues:
            print(f"   ⚠️  Issues found: {len(result.issues)}")
            for issue in result.issues[:5]: # Show first 5
                print(f"      - [{issue.severity}] {issue.rule_id}: {issue.message}")
                if verbose and hasattr(issue, 'recommendation') and issue.recommendation:
                    print(f"        💡 Fix: {issue.recommendation}")
        
        if result.recommendations:
            print(f"   💡 Recommendations: {len(result.recommendations)}")
            for rec in result.recommendations:
                print(f"      - {rec}")
        print("="*40 + "\n")

    # Combine results
    final_output = {
        "source_file": str(file_path),
        "phase_1_structure": doc_dict,
        "phase_2_qualification": {
            "is_chunkable": result.is_chunkable,
            "is_embeddable": result.is_embeddable,
            "is_retrievable": result.is_retrievable,
            "overall_score": result.overall_score,
            "structural_score": result.structural_score,
            "semantic_score": result.semantic_score,
            "retrieval_score": result.retrieval_score,
            "embedding_readiness_score": result.embedding_readiness_score,
            "chunking_strategy": result.chunking_strategy.to_dict() if result.chunking_strategy else None,
            "issues": [i.to_dict() for i in result.issues],
            "warnings": [w.to_dict() if hasattr(w, 'to_dict') else w for w in result.warnings],
            "recommendations": result.recommendations
        }
    }

    if output:
        save_json(final_output, output)
    else:
        # Print summary to stdout if no output file specified
        print(json.dumps({
            "file": str(file_path),
            "score": result.overall_score,
            "status": "PASS" if result.is_chunkable else "NEEDS_REVIEW",
            "strategy": result.chunking_strategy.strategy_type if result.chunking_strategy else "none"
        }, indent=2))


def cmd_heal(args):
    """Handle the 'heal' command - Auto-fix issues using local LLM."""
    input_file = Path(args.input)
    issues_file = Path(args.issues)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    if not issues_file.exists():
        print(f"❌ Issues file not found: {issues_file}")
        sys.exit(1)
        
    output_file = args.output
    if not output_file:
        base, ext = input_file.stem, input_file.suffix
        output_file = f"{base}_fixed{ext}"
        
    print(f"🩺 Starting Self-Healing Process...")
    agent = SelfHealingAgent(model_name=args.model)
    agent.heal(str(input_file), str(issues_file), output_file)


def cmd_process(args):
    """Handle the 'process' command."""
    files = []
    input_path = Path(args.input)
    
    if input_path.is_file():
        files.append(input_path)
    elif input_path.is_dir():
        files = list(input_path.glob("*.md"))
        if not files:
            print(f"No markdown files found in {input_path}")
            sys.exit(0)
    else:
        print(f"Input path does not exist: {input_path}")
        sys.exit(1)

    print(f"📂 Processing {len(files)} file(s)...")
    
    for i, file in enumerate(files, 1):
        if len(files) > 1:
            print(f"\n[{i}/{len(files)}] Processing: {file.name}")
        
        output_file = None
        if args.output:
            # If output is a directory, construct filename
            out_path = Path(args.output)
            if out_path.is_dir() or args.output.endswith('/'):
                out_path.mkdir(parents=True, exist_ok=True)
                output_file = str(out_path / f"{file.stem}_qualified.json")
            else:
                # If single file output for multiple inputs, warn or append?
                # Better to enforce directory for multiple files
                if len(files) > 1:
                    print("⚠️  When processing multiple files, --output must be a directory.")
                    sys.exit(1)
                output_file = args.output
        
        run_full_pipeline(str(file), output=output_file, verbose=args.verbose)


def cmd_stats(args):
    """Handle the 'stats' command."""
    try:
        content = load_markdown(args.input)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    normalized = normalize_markdown(content)
    doc = analyze_structure(normalized)
    
    stats = {
        "file": args.input,
        "original_length": len(content),
        "normalized_length": len(normalized),
        "total_blocks": len(doc.blocks),
        "has_front_matter": doc.has_front_matter,
        "block_counts": {}
    }
    
    for block in doc.blocks:
        bt = block.block_type.value
        stats["block_counts"][bt] = stats["block_counts"].get(bt, 0) + 1
    
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="rag-doc-compiler",
        description="RAG Document Compiler: Phase 1 (Structure) & Phase 2 (Qualification)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process Command
    p_process = subparsers.add_parser("process", help="Run full pipeline (Phase 1 + 2)")
    p_process.add_argument("input", help="Input markdown file or directory")
    p_process.add_argument("-o", "--output", help="Output JSON file or directory")
    p_process.add_argument("-v", "--verbose", action="store_true", help="Show detailed logs")
    p_process.set_defaults(func=cmd_process)

    # Stats Command
    p_stats = subparsers.add_parser("stats", help="Show document statistics")
    p_stats.add_argument("input", help="Input markdown file")
    p_stats.set_defaults(func=cmd_stats)

    # Heal Command
    p_heal = subparsers.add_parser("heal", help="Auto-fix issues using local LLM")
    p_heal.add_argument("input", help="Input markdown file to fix")
    p_heal.add_argument("issues", help="JSON file with issues from 'process' command")
    p_heal.add_argument("-o", "--output", help="Output fixed markdown file")
    p_heal.add_argument("-m", "--model", default="qwen2.5:7b", help="Ollama model to use (default: qwen2.5:7b)")
    p_heal.set_defaults(func=cmd_heal)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
