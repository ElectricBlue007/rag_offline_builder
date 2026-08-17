#!/usr/bin/env python3
"""
run_pipeline.py — Entry point (step 1 of 3).

    (venv) python rag_pipeline/run_pipeline.py

What this command does, in one sentence: it reads your OCR markdown files and
turns them into chunks — small, self-contained pieces of text that a search
engine and a language model can actually work with — one JSON file per input
document.

All configuration lives in settings.yaml (the project's ONE file to
edit — see its header comment) — edit there, not here.

Optional CLI overrides (always win over settings.yaml, for one-off runs):
    python run_pipeline.py --input ./outputs --output ./chunks --verbose

WHAT HAPPENED AND WHY (new)
---------------------------
Unless you pass --no-report, this command also writes a human-readable
account of the run:

    reports/run_<timestamp>/00_SUMMARY.md    one table: every step, in/out, time
    reports/run_<timestamp>/01_*.md ...      one file per step: what it did,
                                             why it matters, which source file
                                             implements it, and a real excerpt
                                             of YOUR data at that point
    reports/run_<timestamp>/DATA_HEALTH.md   concrete problems found in the
                                             chunks, each with a fix
    reports/run_<timestamp>/RAW_TRACE.json   the same thing for machines

Read 00_SUMMARY.md first. If a number looks wrong (1 document became 800
chunks, or 3), the per-step file tells you which stage produced it.

Before your first run, check the machine is ready:
    python rag_pipeline/run_doctor.py
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
# Path(__file__).parent      = .../rag_pipeline/          (the package dir)
# Path(__file__).parent.parent = .../project_root/        (what we need)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.settings import load_settings
from rag_pipeline.pipeline import MarkdownChunkingPipeline
from rag_pipeline.observability import RunReport, analyze_chunks, load_chunks

_S = load_settings()


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
        help=f"Ollama embedding model tag (default: settings.yaml models.embedding = "
             f"{_S.pipeline.semantic.embedding_model}).",
    )
    parser.add_argument(
        "--threshold-type", "-t",
        choices=["percentile", "standard_deviation", "interquartile"],
        default=None,
        help=f"Semantic breakpoint strategy (default: settings.yaml = "
             f"{_S.pipeline.semantic.breakpoint_threshold_type}).",
    )
    parser.add_argument(
        "--threshold-amount", "-a",
        type=float,
        default=None,
        help=f"Numeric value for the threshold (default: settings.yaml = "
             f"{_S.pipeline.semantic.breakpoint_threshold_amount}).",
    )
    parser.add_argument(
        "--contextual-mode",
        choices=["deterministic", "llm", "off"],
        default=None,
        help=(
            "How chunks are situated within their document before "
            "embedding (see ContextualEnricherConfig). 'deterministic' "
            "(default) uses breadcrumb+filename only, zero cost. 'llm' "
            "asks a local Ollama model to write 1-2 situating sentences "
            "per chunk — meaningfully better retrieval precision (this "
            "is Anthropic's 'Contextual Retrieval' technique), at the "
            "cost of one extra LLM call per chunk during ingestion. "
            "Requires Ollama running locally."
        ),
    )
    parser.add_argument(
        "--contextual-model",
        default=None,
        help="Ollama model tag for --contextual-mode llm (default: llama3.2).",
    )
    parser.add_argument(
        "--llm-refine-chunks",
        action="store_true",
        help=(
            "Enable an extra LLM pass after semantic chunking that merges "
            "chunks judged to be split before a complete, independently "
            "answerable thought was finished. Off by default — adds real "
            "wall-clock time (roughly doubles chunking time on CPU-only "
            "local hardware). Requires Ollama running locally."
        ),
    )
    parser.add_argument(
        "--llm-refine-model",
        default=None,
        help="Ollama model tag for --llm-refine-chunks (default: llama3.2).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    # ── Observability ─────────────────────────────────────────────────────
    parser.add_argument(
        "--no-report",
        action="store_true",
        help=(
            "Do not write the human-readable run report. By default this command writes "
            "reports/run_<timestamp>/ containing one Markdown file per step (what it did, "
            "how many items went in and out, how long it took, and a real excerpt of your "
            "data), plus DATA_HEALTH.md listing concrete problems in the chunks produced."
        ),
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Where to write run reports (default: ./reports).",
    )
    parser.add_argument(
        "--report-lang",
        choices=["en", "fa", "both"],
        default="both",
        help="Language of the report text: English, Persian, or both side by side (default: both).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Start from settings.yaml and apply any CLI overrides
    cfg = load_settings().pipeline

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
    if args.contextual_mode:
        cfg.contextual.mode = args.contextual_mode
    if args.contextual_model:
        cfg.contextual.llm_model = args.contextual_model
    if args.llm_refine_chunks:
        cfg.llm_chunk_refiner.enabled = True
    if args.llm_refine_model:
        cfg.llm_chunk_refiner.llm_model = args.llm_refine_model
    if args.quiet:
        cfg.verbose = False

    # A disabled report costs nothing; an enabled one records every step and
    # writes it out as Markdown at the end. The pipeline itself behaves
    # identically either way — observability never alters your data.
    report = RunReport(
        out_dir=args.report_dir,
        title="Chunking pipeline run",
        lang=args.report_lang,
        enabled=not args.no_report,
        settings_path=str(Path(__file__).parent.parent / "settings.yaml"),
        command=" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
    )

    pipeline = MarkdownChunkingPipeline(cfg, runtime=_S.runtime, report=report)
    summary = pipeline.run()

    if not args.no_report:
        # Judge the chunks we just wrote, using THIS project's own thresholds
        # from settings.yaml rather than invented ones.
        try:
            chunks = load_chunks(Path(cfg.output_dir))
            report.add_findings(analyze_chunks(
                chunks,
                min_chunk_chars=cfg.semantic.min_chunk_chars,
                max_chunk_chars=cfg.semantic.max_chunk_chars,
                expect_embed_text=cfg.embed_text.enabled,
            ))
        except Exception as exc:                     # never fail a good run
            print(f"  [observability] WARNING: data health check skipped: {exc}")

        print(report.console_summary())
        folder = report.write()
        if folder:
            print(f"  📄 Step-by-step report : {folder}/00_SUMMARY.md")
            print(f"  🧪 Data health         : {folder}/DATA_HEALTH.md")
            print(f"  🧾 Machine-readable    : {folder}/RAW_TRACE.json\n")

    sys.exit(0 if not summary["failed"] else 1)


if __name__ == "__main__":
    main()
