"""
pipeline.py — Main pipeline orchestrator.

Chains:
  MarkdownLoader
    → MarkdownPreprocessor
      → StructureSplitter
        → SemanticChunker
          → LlmChunkRefiner        (optional: LLM-judged merge of split-too-early chunks)
            → TableChunkNormalizer      (HTML tables → embeddable prose)
              → ContextualEnricher      (situates each chunk in its document)
                → EmbedTextBuilder      (applies the embedding model's prefix)
                  → MetadataEnricher    (adds chunk_id + descriptive metadata)
                    → JSON output (one file per .md input)

Each stage is independent and can be swapped or extended without touching
the others.
"""

import json
import time
from pathlib import Path
from typing import Iterator, Optional

from rag_pipeline.config import PipelineConfig, RuntimeConfig
from rag_pipeline import model_manager
from rag_pipeline.preprocessors.md_preprocessor import MarkdownPreprocessor
from rag_pipeline.splitters.structure_splitter import StructureSplitter
from rag_pipeline.splitters.semantic_chunker import SemanticChunker
from rag_pipeline.splitters.llm_chunk_refiner import LlmChunkRefiner
from rag_pipeline.normalizers.chunk_normalizer import TableChunkNormalizer
from rag_pipeline.enrichers.contextual_enricher import ContextualEnricher
from rag_pipeline.enrichers.embed_text_builder import EmbedTextBuilder
from rag_pipeline.enrichers.metadata_enricher import MetadataEnricher, EnrichedChunk


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

    def __init__(self, config: PipelineConfig, runtime: Optional[RuntimeConfig] = None):
        self.cfg = config
        self.runtime = runtime or RuntimeConfig()

        # Instantiate each stage with its sub-config. `runtime` is
        # threaded through to every stage that calls a model, so they
        # all share the same "only one model on the GPU at a time"
        # policy from settings.yaml's runtime: block.
        self.preprocessor     = MarkdownPreprocessor(config.preprocessor)
        self.struct_split     = StructureSplitter(config.structure)
        self.sem_chunker      = SemanticChunker(config.semantic, runtime=self.runtime)
        # semantic_config is passed through so the refiner's post-merge
        # resplit safety net uses the SAME chunk-size ceiling as the
        # initial semantic split, instead of a hardcoded value of its own.
        self.llm_refiner      = LlmChunkRefiner(
            config.llm_chunk_refiner, semantic_config=config.semantic, runtime=self.runtime
        )
        self.table_normalizer = TableChunkNormalizer(config.table_normalizer)
        self.contextual       = ContextualEnricher(config.contextual, runtime=self.runtime)
        self.embed_text_bldr  = EmbedTextBuilder(config.embed_text)
        self.enricher         = MetadataEnricher(config.metadata)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Process every .md file found under input_dir and write chunk JSONs
        to output_dir.

        Deliberately staged by MODEL rather than by file: every file's
        embedding-model work (steps 1-4) runs before ANY file's LLM
        refiner work (step 4b) starts, instead of interleaving embed →
        refine → embed → refine per file. On a GPU that can't hold the
        embedding model and the refiner LLM in VRAM at once, per-file
        interleaving forces Ollama to evict and reload whichever model
        isn't currently needed on EVERY file — for N files that's up to
        2N model loads. Staging by model means the embedding model loads
        once for the whole run and the refiner model loads once for the
        whole run, regardless of N. When the refiner is disabled (the
        default), stage 2 is a no-op and this behaves exactly as before.

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

        failed: list[str] = []
        total_chunks = 0

        # ── Stage 1 — embedding-model work for every file ───────────────
        # (load, preprocess, structure split, semantic chunk). Collects
        # each file's intermediate state instead of finishing it, so the
        # embedding model stays the only model in play until every file
        # has been through it.
        staged: list[dict] = []
        for i, md_path in enumerate(md_files, start=1):
            self._log(f"[{i}/{len(md_files)}] Chunking  : {md_path.relative_to(input_dir)}")
            t0 = time.time()
            try:
                sem_chunks, preprocessed = self._embed_stage(md_path)
                staged.append({
                    "md_path": md_path,
                    "sem_chunks": sem_chunks,
                    "preprocessed": preprocessed,
                    "elapsed": time.time() - t0,
                })
            except Exception as exc:
                self._log(f"  ❌  FAILED (chunking): {exc}")
                failed.append(str(md_path))

        # ── Stage 2 — LLM refiner work for every file that survived stage 1
        # A genuine no-op (refine() returns its input unchanged) when
        # llm_chunk_refiner.enabled is False, so this costs nothing and
        # triggers no model load at all in the default configuration.
        if self.cfg.llm_chunk_refiner.enabled and staged:
            self._log(f"\n[{len(staged)} file(s)] Refining chunks with LLM...")
            # Stage 1 is fully done for every file at this point — release
            # the embedding model NOW rather than waiting for the refiner
            # model's own ensure_active() call to evict it implicitly.
            # Same "load it, use it, explicitly stop it, then load the
            # next one" flow the pipeline module docstring promises,
            # made visible at the orchestration level too.
            model_manager.release(
                self.cfg.semantic.embedding_host, self.cfg.semantic.embedding_model, self.runtime
            )
        for entry in staged:
            t0 = time.time()
            try:
                entry["sem_chunks"] = self.llm_refiner.refine(entry["sem_chunks"])
                entry["elapsed"] += time.time() - t0
            except Exception as exc:
                rel = entry["md_path"].relative_to(input_dir)
                self._log(f"  ❌  FAILED (refine) {rel}: {exc}")
                failed.append(str(entry["md_path"]))
                entry["skip"] = True

        # ── Stage 3 — everything else (no model, or whatever model the
        # contextual enricher uses if --contextual-mode llm is set) —
        # run per file and save. No embedding/refiner calls happen here.
        if self.cfg.llm_chunk_refiner.enabled and staged:
            # Refiner stage is done for every file — release its model
            # before stage 3 potentially loads yet another one (the
            # contextual enricher's LLM, if --contextual-mode llm).
            model_manager.release(
                self.cfg.llm_chunk_refiner.llm_host, self.cfg.llm_chunk_refiner.llm_model, self.runtime
            )
        for entry in staged:
            if entry.get("skip"):
                continue
            md_path = entry["md_path"]
            t0 = time.time()
            try:
                enriched = self._finish_stage(
                    entry["sem_chunks"], md_path, entry["preprocessed"]
                )
                self._save_chunks(enriched, md_path, output_dir, input_dir)
                elapsed = entry["elapsed"] + (time.time() - t0)
                self._log(
                    f"  ✅  {len(enriched)} chunks  "
                    f"({elapsed:.1f}s)  "
                    f"→ {self._output_path(md_path, output_dir, input_dir).name}"
                )
                total_chunks += len(enriched)
            except Exception as exc:
                self._log(f"  ❌  FAILED: {exc}")
                failed.append(str(md_path))

        self._log(f"\n{'='*55}")
        self._log(f"  DONE — {len(md_files) - len(failed)} files, {total_chunks} chunks")
        if failed:
            self._log(f"  Failed: {failed}")
        self._log(f"{'='*55}\n")

        # Final safety net: whatever model was active last (embedding,
        # refiner, or the contextual enricher's LLM if mode="llm") is
        # explicitly released so nothing is left resident in VRAM once
        # the whole chunking run is done.
        if self.cfg.contextual.mode == "llm":
            model_manager.release(self.cfg.contextual.llm_host, self.cfg.contextual.llm_model, self.runtime)
        else:
            model_manager.release(
                self.cfg.semantic.embedding_host, self.cfg.semantic.embedding_model, self.runtime
            )

        return {
            "files_processed": len(md_files) - len(failed),
            "total_chunks": total_chunks,
            "failed": failed,
        }

    def process_file(self, md_path: Path) -> list[EnrichedChunk]:
        """
        Run the full pipeline for a SINGLE markdown file, start to
        finish, in one call. Used directly by callers/tests that want
        one file's result without going through run()'s multi-file
        staging — behaves identically to how this method always has,
        just now implemented as the two stages run() also uses.
        """
        sem_chunks, preprocessed = self._embed_stage(md_path)
        sem_chunks = self.llm_refiner.refine(sem_chunks)
        return self._finish_stage(sem_chunks, md_path, preprocessed)

    # ── Stage helpers (also used directly by run()'s batching) ─────────────

    def _embed_stage(self, md_path: Path):
        """Steps 1-4: load → preprocess → structure split → semantic chunk."""
        raw_text = _load_md(md_path)
        preprocessed = self.preprocessor.process(raw_text, source_path=str(md_path))
        sections = self.struct_split.split(
            preprocessed.cleaned_text, source_path=str(md_path)
        )
        sections = self._resolve_image_placeholders(sections, preprocessed.image_alts)
        sem_chunks = self.sem_chunker.chunk_sections(sections)
        return sem_chunks, preprocessed

    def _finish_stage(self, sem_chunks, md_path: Path, preprocessed) -> list[EnrichedChunk]:
        """Steps 5-8: table normalize → contextual enrich → embed_text → metadata."""
        # Merging in the refiner stage changes the chunk count, so
        # chunk_index must be reassigned — MetadataEnricher relies on it
        # being correctly sequential for both chunk_id hashing and
        # page-number estimation.
        for i, chunk in enumerate(sem_chunks):
            chunk.chunk_index = i

        sem_chunks = self.table_normalizer.normalize_chunks(sem_chunks)
        sem_chunks = self.contextual.enrich_chunks(
            sem_chunks,
            document_text=preprocessed.cleaned_text,
            source_filename=md_path.name,
        )
        sem_chunks = self.embed_text_bldr.build(sem_chunks)
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
