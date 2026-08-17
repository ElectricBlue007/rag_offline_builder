# Ingestion pipeline stages

The orchestrator is `rag_pipeline/pipeline.py`. It owns wiring, model-lifecycle batching, progress logging, and output-path calculation. It does **not** own the parsing logic. Each transformation has one module with one documented `run()` method.

## Order and contracts

| # | Module | Input → output | Debug first |
|---:|---|---|---|
| 01 | `load_markdown.py` | `Path → str` | encoding/file errors |
| 02 | `preprocess_markdown.py` | raw text → `PreprocessedDocument` | cleaned text, image captures |
| 03 | `split_structure.py` | cleaned text → `StructuralSection[]` | breadcrumbs, tables, images |
| 04 | `resolve_images.py` | sections + alt text → resolved sections | empty placeholders |
| 05 | `split_semantic.py` | sections → `SemanticChunk[]` | effective semantic/fallback mode |
| 06 | `refine_chunks.py` | chunks → chunks | merge decisions and LLM fallback |
| 07 | `normalize_tables.py` | chunks → chunks | raw `content` vs `normalized_text` |
| 08 | `enrich_context.py` | chunks + document → contextual chunks | breadcrumb/source context |
| 09 | `build_embed_text.py` | contextual chunks → embed-ready chunks | prefix contract |
| 10 | `enrich_metadata.py` | chunks → `EnrichedChunk[]` | IDs, pages, language, hierarchy |
| 11 | `write_chunks.py` | enriched chunks → JSON | output path and serialization |

## Debugging procedure

1. Run `python rag_pipeline/test_dry_run.py` to test stages 01–03 without Ollama.
2. Inspect sections before semantic splitting; table and image sections must already be isolated.
3. Inspect `SemanticChunker.diagnostics` and any fallback warning.
4. Verify table `content` stays raw while `normalized_text` is prose.
5. Verify every chunk has a stable ID and source metadata.
6. Run BM25 independently before adding Chroma, reranking, or answer generation.

## Extension rule

Add a new stage module, document its input/output contract, add a focused test, then add one wiring line to `pipeline.py`. Do not put new business logic into the orchestrator.
