"""End-to-end PDF OCR -> Markdown compiler -> chunks -> retrieval pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from adapters.chandra_runner import ChandraRunner
from api.rag_service import RAGService
from config.settings import ModelSettings, DEFAULT_MODEL_SETTINGS
from rag_doc_compiler.pipeline import DocumentCompiler


class BatchPipeline:
    def __init__(self, settings: ModelSettings = DEFAULT_MODEL_SETTINGS) -> None:
        self.settings = settings
        self.compiler = DocumentCompiler()

    def run(
        self,
        input_dir: str | Path,
        workspace_dir: str | Path,
        query: str | None = None,
        skip_ocr: bool = False,
        use_llm: bool = True,
        store_type: str = "faiss",
    ) -> dict:
        workspace = Path(workspace_dir)
        raw_markdown_dir = workspace / "output_markdowns"
        compiled_dir = workspace / "compiled"
        corpus_dir = workspace / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        chandra_results = []
        if not skip_ocr:
            chandra_results = ChandraRunner(cache_dir=self.settings.chandra_cache_dir).run_directory(input_dir, raw_markdown_dir)

        markdown_files = sorted(raw_markdown_dir.rglob("*.md"))
        if not markdown_files:
            raise FileNotFoundError(f"No Markdown files found under: {raw_markdown_dir}")

        compiled_outputs = []
        corpus_chunks = []
        for markdown_path in markdown_files:
            relative = markdown_path.relative_to(raw_markdown_dir)
            output_dir = compiled_dir / relative.with_suffix("")
            result = self.compiler.run_file(markdown_path, output_dir)
            compiled_outputs.append({
                "input_markdown": str(markdown_path),
                "clean_markdown": str(result["markdown_path"]),
                "chunks": str(result["chunks_path"]),
                "index": str(result["index_path"]),
            })
            chunks_data = json.loads(result["chunks_path"].read_text(encoding="utf-8"))
            document_id = str(relative.with_suffix(""))
            for chunk in chunks_data["chunks"]:
                chunk["document_id"] = document_id
                chunk["chunk_id"] = f"{document_id}::{chunk['chunk_id']}"
                adjacency = chunk.get("adjacency", {})
                for key in ("previous_chunk_id", "next_chunk_id"):
                    if adjacency.get(key):
                        adjacency[key] = f"{document_id}::{adjacency[key]}"
                corpus_chunks.append(chunk)

        corpus_path = corpus_dir / "chunks.json"
        corpus_payload = {
            "strategy": {"name": "batch_corpus_chunks_v1", "source_count": len(compiled_outputs)},
            "chunks": corpus_chunks,
            "observability": {
                "document_count": len(compiled_outputs),
                "chunk_count": len(corpus_chunks),
            },
        }
        corpus_path.write_text(json.dumps(corpus_payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

        response = None
        if query:
            service = RAGService(settings=self.settings, store_type=store_type, use_llm=use_llm)
            service.load_chunks(str(corpus_path))
            response = service.ask(query)

        return {
            "chandra_processed": [str(item.pdf_path) for item in chandra_results],
            "raw_markdown_dir": str(raw_markdown_dir),
            "compiled_outputs": compiled_outputs,
            "corpus_chunks": str(corpus_path),
            "response": response,
        }
