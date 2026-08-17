"""
Observability layer: make the pipeline explain itself.

Nothing in this package changes how documents are processed. It only records
what happened and turns that into Markdown a non-expert can read:

  RunReport   -- one record per pipeline step (in, out, time, notes, real
                 excerpts of your data, and which source file implements it),
                 written to reports/run_<timestamp>/.
  analyze_chunks / Finding
              -- rules over the chunks on disk that name real data problems
                 and the fix for each one.
  i18n        -- tiny bilingual (English / فارسی) Markdown helpers.

Everything here is stdlib-only and safe to import even when Ollama, Chroma or
the ML dependencies are not installed.
"""
from . import i18n
from .health import (
    Finding,
    analyze_chunks,
    console_summary as health_console_summary,
    corpus_stats,
    findings_md,
    load_chunks,
)
from .run_report import STEP_DOCS, RunReport, Step, StepDoc, StepHandle, doc_for

__all__ = [
    "i18n",
    "RunReport", "Step", "StepHandle", "StepDoc", "STEP_DOCS", "doc_for",
    "Finding", "analyze_chunks", "corpus_stats", "findings_md", "load_chunks",
    "health_console_summary",
]
