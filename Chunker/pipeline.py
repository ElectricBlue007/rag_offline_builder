"""
pipeline.py — Main pipeline orchestrator.

Chains:
  MarkdownLoader
    → MarkdownPreprocessor
      → StructureSplitter
        → SemanticChunker
          → MetadataEnricher
            → JSON output (one file per .md input)

Each stage is independent and can be swapped or extended without touching
the others.
"""

import json
import time
from pathlib import Path
from typing import Iterator

from Chunker.config import PipelineConfig
from Chunker.preprocessors.md_preprocessor import MarkdownPreprocessor
from Chunker.splitters.structure_splitter import StructureSplitter
from Chunker.splitters.semantic_chunker import SemanticChunker
from Chunker.enrichers.metadata_enricher import MetadataEnricher, EnrichedChunk


# ── Markdown file loader ───────────────────────────────────────────────────────

def _discover_md_files(input_dir: Path, glob: str) -> list[Path]:
    """Recursively find all markdown files matching the glob pattern."""
    files = sorted(input_dir.glob(glob))
    # Exclude _metadata.json siblings (they're JSON, not markdown — but just in case)
    return [f for f in files if f.suffix.lower() == ".md"]


def _load_md(path: Path) -> str:
    """Read a markdown file as UTF-8 text."""
    return path.read_text(encoding="utf-8", errors="replace")


# ── Pipeline ──────────────────────────────────────────────────────────────────

class MarkdownChunkingPipeline:
    """
    Orchestrates the full chunking pipeline for a directory of .md files.

    Usage:
        cfg = PipelineConfig(input_dir=Path("./outputs"))
        pipeline = MarkdownChunkingPipeline(cfg)
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config

        # Instantiate each stage with its sub-config
        self.preprocessor  = MarkdownPreprocessor(config.preprocessor)
        self.struct_split  = StructureSplitter(config.structure)
        self.sem_chunker   = SemanticChunker(config.semantic)
        self.enricher      = MetadataEnricher(config.metadata)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Process every .md file found under input_dir and write chunk JSONs
        to output_dir.

        Returns a summary dict:
            {
              "files_processed": int,
              "total_chunks": int,
              "failed": [str, ...],
            }
        """
        input_dir  = Path(self.cfg.input_dir)
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        md_files = _discover_md_files(input_dir, self.cfg.md_glob)
        if not md_files:
            self._log(f"⚠️  No .md files found under: {input_dir}")
            return {"files_processed": 0, "total_chunks": 0, "failed": []}

        self._log(f"\n📂 Input  : {input_dir}")
        self._log(f"📂 Output : {output_dir}")
        self._log(f"🔍 Found  : {len(md_files)} markdown file(s)\n")

        total_chunks = 0
        failed: list[str] = []

        for i, md_path in enumerate(md_files, start=1):
            self._log(f"[{i}/{len(md_files)}] Processing: {md_path.relative_to(input_dir)}")
            t0 = time.time()

            try:
                chunks = self.process_file(md_path)
                self._save_chunks(chunks, md_path, output_dir, input_dir)
                elapsed = time.time() - t0
                self._log(
                    f"  ✅  {len(chunks)} chunks  "
                    f"({elapsed:.1f}s)  "
                    f"→ {self._output_path(md_path, output_dir, input_dir).name}"
                )
                total_chunks += len(chunks)
            except Exception as exc:
                self._log(f"  ❌  FAILED: {exc}")
                failed.append(str(md_path))

        self._log(f"\n{'='*55}")
        self._log(f"  DONE — {len(md_files) - len(failed)} files, {total_chunks} chunks")
        if failed:
            self._log(f"  Failed: {failed}")
        self._log(f"{'='*55}\n")

        return {
            "files_processed": len(md_files) - len(failed),
            "total_chunks": total_chunks,
            "failed": failed,
        }

    def process_file(self, md_path: Path) -> list[EnrichedChunk]:
        """
        Run the full pipeline for a single markdown file.
        Returns a list of EnrichedChunk objects.
        """
        # 1. Load
        raw_text = _load_md(md_path)

        # 2. Preprocess
        preprocessed = self.preprocessor.process(raw_text, source_path=str(md_path))

        # 3. Structure split
        sections = self.struct_split.split(
            preprocessed.cleaned_text, source_path=str(md_path)
        )

        # 3b. Handle image descriptions extracted by preprocessor
        #     Each placeholder in sections is already tagged as image_description.
        #     Replace the placeholder text with the actual alt content.
        sections = self._resolve_image_placeholders(sections, preprocessed.image_alts)

        # 4. Semantic chunking within each section
        sem_chunks = self.sem_chunker.chunk_sections(sections)

        # 5. Metadata enrichment
        enriched = self.enricher.enrich(sem_chunks, md_path)

        return enriched

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_image_placeholders(self, sections, image_alts):
        """
        Replace [IMAGE_DESC_PLACEHOLDER_N] tokens in section content
        with the actual alt text captured by the preprocessor.
        """
        import re
        _re = re.compile(r'\[IMAGE_DESC_PLACEHOLDER_(\d+)\]')

        resolved = []
        for sec in sections:
            if sec.section_type == "image_description":
                def _sub(m, alts=image_alts):
                    idx = int(m.group(1))
                    return alts[idx].alt_text if idx < len(alts) else ""
                sec.content = _re.sub(_sub, sec.content).strip()
                if sec.content:
                    resolved.append(sec)
            else:
                resolved.append(sec)

        return resolved

    def _output_path(self, md_path: Path, output_dir: Path, input_dir: Path) -> Path:
        """Mirror the input directory structure in the output directory."""
        try:
            rel = md_path.relative_to(input_dir)
        except ValueError:
            rel = Path(md_path.name)
        return output_dir / rel.with_suffix(".chunks.json")

    def _save_chunks(
        self,
        chunks: list[EnrichedChunk],
        md_path: Path,
        output_dir: Path,
        input_dir: Path,
    ) -> None:
        out_path = self._output_path(md_path, output_dir, input_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_file": str(md_path),
            "total_chunks": len(chunks),
            "chunks": [c.to_dict() for c in chunks],
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg)
