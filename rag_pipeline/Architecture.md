# Architecture

Offline, fully local RAG pipeline: PDF → OCR → chunking → embeddings → hybrid retrieval → LLM answers → evaluation. No API keys, no cloud dependency — every model (embedding, reranking, generation, evaluation judge) runs locally via Ollama, ChromaDB, and sentence-transformers.

## Pipeline Flow
```
PDF
 │
 ▼
Chandra OCR  ──────────────────────────►  .md + _metadata.json
 │
 ▼
┌─────────────────────── pipeline.py (chunking) ───────────────────────┐
│ MarkdownPreprocessor → StructureSplitter → SemanticChunker           │
│   → LlmChunkRefiner (optional) → TableChunkNormalizer                 │
│   → ContextualEnricher → EmbedTextBuilder → MetadataEnricher          │
└────────────────────────────────────────────────────────────────────┘
 │
 ▼
chunks/*.chunks.json
 │
 ├──────────────► EmbedGenerator ──► ChromaIngestor  ──► ChromaDB (dense)
 │
 └──────────────► BM25Index ──────────────────────────► bm25_index.pkl (sparse)
 │
 ▼
HybridRetriever (RRF fusion)  ──►  CrossEncoderReranker (optional)
 │
 ▼
ContextBuilder → RagAnswerer → LlmClient (Ollama)
 │
 ▼
Generated answer + citations
 │
 ▼
RagasEvaluator (measures the whole thing)
```

## Directory Structure

```
rag_pipeline/
├── config.py                  # single source of truth for every tunable parameter
├── pipeline.py                 # chunking pipeline orchestrator
├── run_pipeline.py              # CLI: PDF-derived .md → chunks.json
├── run_ingestion.py              # CLI: chunks.json → ChromaDB
├── run_bm25_index.py              # CLI: chunks.json → BM25 index
├── run_query.py                    # CLI: hybrid retrieval only
├── run_answer.py                    # CLI: full question-answering
├── run_evaluation.py                 # CLI: RAGAS evaluation
├── requirements.txt
│
├── preprocessors/
│   ├── md_preprocessor.py       # cleans raw Chandra markdown
│   └── language_detector.py      # offline EN/ES detection
│
├── splitters/
│   ├── structure_splitter.py    # header-hierarchy aware splitting
│   ├── token_counter.py          # real tokenizer or heuristic fallback
│   ├── semantic_chunker.py        # LangChain semantic splitting
│   └── llm_chunk_refiner.py        # optional LLM-judged merge of split-too-early chunks
│
├── normalizers/
│   ├── table_normalizer.py      # HTML tables → natural-language prose
│   └── chunk_normalizer.py       # pipeline stage wrapper for table_normalizer
│
├── enrichers/
│   ├── contextual_enricher.py   # Contextual Retrieval technique
│   ├── embed_text_builder.py     # embedding model prefix contract (config-driven)
│   └── metadata_enricher.py       # final metadata + chunk_id assignment
│
├── embedding/
│   └── embed_generator.py       # batch vector generation via Ollama
│
├── storage/
│   ├── chroma_ingestion.py      # ChromaDB persistent vector store
│   └── bm25_index.py             # BM25 keyword index
│
├── retrieval/
│   ├── hybrid_retriever.py      # dense + sparse fusion (RRF)
│   └── reranker.py               # cross-encoder precision pass
│
├── generation/
│   ├── context_builder.py       # formats chunks into LLM-ready context
│   ├── llm_client.py             # Ollama wrapper
│   └── rag_answerer.py            # full retrieve→generate orchestration
│
├── evaluation/
│   ├── eval_dataset.py          # eval question structure + JSON I/O
│   ├── sample_eval_questions.json
│   └── ragas_evaluator.py        # RAGAS wired to Ollama (judge LLM + embeddings)
│
└── tests/
    ├── fixtures/                # real sample OCR output (Binder2.md, Testing.md)
    ├── run_tests.py              # unified test runner, all suites
    └── test_*.py                  # one file per module, 825 tests total
```

---

## Module Reference

### Configuration

**`config.py`**
Every tunable parameter for every stage, as one dataclass per module (`PreprocessorConfig`, `SemanticChunkerConfig`, `ChromaIngestionConfig`, `RagasEvaluatorConfig`, etc.). Nothing is hardcoded elsewhere — change behavior here, not in the module logic.

---

### Stage 1 — Preprocessing

**`preprocessors/md_preprocessor.py`** — `MarkdownPreprocessor`
Cleans raw Chandra OCR markdown before chunking: separates adjacent HTML tables that got glued together, strips table-of-contents tables (dot-leader detection), extracts image alt-text into placeholders, removes superscript footnote markers, strips decorative OCR artifacts ("Continues next page...").

**`preprocessors/language_detector.py`** — `detect_language()`
Lightweight, fully offline EN/ES language detection using word-frequency heuristics — no model download, no network call. Used for query-language detection (hybrid retrieval filtering, LLM answer-language matching) and per-chunk language tagging.

---

### Stage 2 — Structural & Semantic Splitting

**`splitters/structure_splitter.py`** — `StructureSplitter`
Splits markdown by header hierarchy (`#`, `##`, `###`), building a breadcrumb trail per section. Tables and image descriptions are always kept as atomic, unsplittable units regardless of size.

**`splitters/token_counter.py`** — `TokenCounter`
Measures chunk size in actual tokens (via a real HuggingFace tokenizer when available) rather than characters, since the embedding model's 512-token limit is what actually matters. Falls back to a word-count heuristic when the real tokenizer can't be loaded — never blocks the pipeline.

**`splitters/semantic_chunker.py`** — `SemanticChunker`
Applies LangChain's semantic chunking within each structural section, using token-aware size limits. Tables and images bypass semantic splitting entirely (already atomic from the structure splitter).

**`splitters/llm_chunk_refiner.py`** — `LlmChunkRefiner` *(optional, off by default)*
Runs immediately after `SemanticChunker`. Embedding-similarity chunking is a purely statistical signal with no notion of whether a chunk actually reads as a complete, independently-answerable thought — it sometimes cuts a passage right before the sentence that would have made it self-contained. This stage sends each run of adjacent same-section chunks to a local Ollama model and asks it to judge, per chunk, whether it alone would fully answer a question about its content or feels like it's missing something from just before/after it — then merges accordingly. **Scope is deliberately narrow: the LLM decides merge boundaries only, never rewrites text.** Merged chunks are the original content concatenated verbatim; an LLM "helpfully" paraphrasing a procedure step or table value would corrupt the source of truth this pipeline retrieves from, which is a worse failure than an imperfect boundary. Falls back to the original (unmerged) boundaries on any failure — never blocks the pipeline. Enable via `python run_pipeline.py --llm-refine-chunks`.

---

### Stage 3 — Table Normalization

**`normalizers/table_normalizer.py`** — `normalize_table_html()`
Converts HTML tables into natural-language prose before embedding — critical because embedding models tokenize raw HTML as syntax noise, not semantics. Detects two table shapes automatically: header-row tables (standard `<thead>/<tbody>`) and key-value tables (bold-label rows with no header), each normalized with shape-appropriate logic.

**`normalizers/chunk_normalizer.py`** — `TableChunkNormalizer`
Pipeline-stage wrapper around `table_normalizer.py`: applies normalization to table chunks, falls back gracefully to raw HTML on any parsing error so one malformed table never crashes a whole pipeline run.

---

### Stage 4 — Contextual Enrichment & Embedding Prep

**`enrichers/contextual_enricher.py`** — `ContextualEnricher`
Implements Contextual Retrieval: prepends a short situating blurb to each chunk before embedding, so an isolated fragment doesn't lose the surrounding context that makes it meaningful. Two modes: `"deterministic"` (breadcrumb + filename, free, default) or `"llm"` (Ollama-generated richer context, opt-in).

**`enrichers/embed_text_builder.py`** — `EmbedTextBuilder`, `prepare_query_text()`
The single source of truth for the embedding model's instruction-prefix contract (see `EmbedTextConfig` in `config.py`) — currently no prefix for stored chunks and an instruction-style prefix for search queries at retrieval time (qwen3-embedding's contract; previously `"passage: "`/`"query: "` under E5). Centralizing both prevents the two from ever drifting out of sync.

**`enrichers/metadata_enricher.py`** — `MetadataEnricher`, `EnrichedChunk`
Final enrichment stage: attaches breadcrumb, detected language, chunk type, page numbers, and a stable deterministic `chunk_id` (hash of source file + index) used for idempotent vector-store upserts. Produces `EnrichedChunk` — the final chunk representation written to `chunks.json`.

---

### Stage 5 — Embedding Generation

**`embedding/embed_generator.py`** — `EmbedGenerator`
Batch-generates embedding vectors from chunk text via a local Ollama instance (default model: `qwen3-embedding8b:latest`). Decoupled from the chunking pipeline — can re-embed all chunks after switching models without reprocessing PDFs. Includes helpers for loading/discovering `chunks.json` files across a directory tree.

---

### Stage 6 — Storage

**`storage/chroma_ingestion.py`** — `ChromaIngestor`
Ingests embedded chunks into a local, persistent ChromaDB collection. Matches vectors to chunks by `chunk_id` (not list position, for robustness), sanitizes metadata into ChromaDB-compatible scalar types, and supports both similarity search (`query()`) and direct ID lookup (`get_by_ids()`).

**`storage/bm25_index.py`** — `BM25Index`
Keyword/exact-term search index, essential for technical documents full of domain abbreviations an embedding model has never seen. Built from `contextualized_text` (not the prefixed `embed_text`, since any instruction prefix is pure noise for keyword matching).

---

### Stage 7 — Retrieval

**`retrieval/hybrid_retriever.py`** — `HybridRetriever`, `reciprocal_rank_fusion()`
Combines dense (ChromaDB) and sparse (BM25) search via Reciprocal Rank Fusion — merging two ranked lists on incomparable score scales using rank position only. Detects query language for optional metadata filtering, applies the configured query-side prefix, and backfills display data for any chunk found only by BM25.

**`retrieval/reranker.py`** — `CrossEncoderReranker`
Optional precision pass: re-scores the hybrid retriever's top candidates with a cross-encoder that sees the query and each candidate together (full attention), more accurate than the bi-encoder's independent-embedding approach but too slow to run against the whole corpus. Default model is `Qwen/Qwen3-Reranker-4B` (instruction-aware, strong multilingual retrieval benchmarks). Opt-in via `--rerank` since it requires `sentence-transformers`/`torch` and, at 4B parameters, a GPU with a few GB of free VRAM.

---

### Stage 8 — Generation

**`generation/context_builder.py`** — `build_context()`
Formats retrieved chunks into a numbered, citable context block for the LLM prompt (`[1] Table — source.pdf, page 3 — breadcrumb`). Includes a defensive HTML-strip fallback, though table normalization upstream already guarantees clean prose reaches this stage.

**`generation/llm_client.py`** — `LlmClient`
Thin wrapper around Ollama for fully offline answer generation — no API key, no external network call beyond the local Ollama service.

**`generation/rag_answerer.py`** — `RagAnswerer`, `RagAnswer`
Orchestrates the full answer flow: retrieve → optionally rerank → build context → construct prompt → generate → return answer with citable sources. Detects query language and instructs the LLM to answer in kind.

---

### Stage 9 — Evaluation

**`evaluation/eval_dataset.py`** — `EvalQuestion`, `load_eval_questions()`, `save_eval_questions()`
Simple JSON-based structure for a hand-curated evaluation set (question, ground-truth answer, language, category tags).

**`evaluation/sample_eval_questions.json`**
A 12-question starter evaluation set built from this project's own fixture documents — mixes English/Spanish and table/prose/image-description question types.

**`evaluation/ragas_evaluator.py`** — `RagasEvaluator`, `EvaluationSummary`
Runs the full pipeline against eval questions, then scores results using RAGAS's four core metrics (context precision, context recall, faithfulness, answer relevancy). Wired to Ollama for both the judge LLM and the embedding-based metrics (matching the embedding model used everywhere else in the pipeline) instead of RAGAS's OpenAI defaults — keeping evaluation fully offline like everything else. Split into two explicit steps (`run_pipeline_on_questions()` then `score()`) so judge-LLM cost is only spent when you're ready to pay it.

---

## Entry Points (CLI Scripts)

| Script | Purpose |
|---|---|
| `run_pipeline.py` | PDF-derived markdown → `chunks/*.chunks.json` |
| `run_ingestion.py` | Embed chunks and ingest into ChromaDB |
| `run_bm25_index.py` | Build the BM25 keyword index |
| `run_query.py` | Query hybrid retrieval directly (no generation) |
| `run_answer.py` | Full question-answering with a generated, cited answer |
| `run_evaluation.py` | Score the whole pipeline with RAGAS metrics |

Each script is self-contained and independently runnable — no cross-script imports, matching the pipeline's modular design.

---

## Testing

`tests/run_tests.py` runs the complete suite (825 tests across every module) with a single command. Real fixture files (`tests/fixtures/Binder2.md`, `Testing.md`) are used throughout for integration-level tests, not just synthetic strings. Heavy external dependencies (ChromaDB, Ollama, sentence-transformers) are mocked in unit tests but exercised for real wherever lightweight enough to do so (BM25, table parsing, RAGAS object construction).
