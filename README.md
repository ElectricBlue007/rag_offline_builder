# Local-First RAG Document Intelligence PipeLine

This project runs a complete local-first RAG ingestion and querying workflow:

```text
PDFs
→ Chandra OCR Markdown
→ clean preserved Markdown
→ structured JSON index
→ finalized chunks
→ embeddings
→ local vector retrieval
→ graph expansion
→ Ollama LLM answer with citations
```

The system preserves original content. Cleaning is limited to traceable structural OCR fixes such as glued headings or glued HTML tables.

## Default Models

- Embedding model: `BAAI/bge-m3`
- Embedding provider: `sentence_transformers`
- Embedding cache directory: `models/embeddings`
- LLM provider: `ollama`
- LLM model: `gemma4`
- Chandra/Hugging Face cache: `models/chandra`

`BAAI/bge-m3` is the default because it is a strong stable multilingual retrieval model. The first real embedding run downloads it automatically into `models/embeddings`.

## Setup

Install Python dependencies:

```bash
pip install -r requirements-dev.txt
pip install -r requirements-ml.txt
```

Make sure Chandra OCR is installed and available as:

```bash
chandra --help
```

Make sure Ollama is running and your Gemma model is available:

```bash
ollama serve
ollama list
```

If your local model name is different from `gemma4`, set it:

```bash
export LLM_MODEL=your-gemma-model-name
```

Optional environment config:

```bash
export EMBEDDING_PROVIDER=sentence_transformers
export EMBEDDING_MODEL=BAAI/bge-m3
export EMBEDDING_CACHE_DIR=models/embeddings
export CHANDRA_CACHE_DIR=models/chandra
export LLM_PROVIDER=ollama
export LLM_MODEL=gemma4
export OLLAMA_BASE_URL=http://localhost:11434
```

For offline development without downloading the embedding model:

```bash
export EMBEDDING_PROVIDER=local_hash
```

## One Command Full Pipeline

Put PDFs under an input directory:

```text
input/
  book.pdf
  folder/another.pdf
```

Run everything and ask a question:

```bash
python -m cli.main full-run input workspace --query "What does this document say?"
```

Default output is human-readable. If you need the old machine-readable JSON output:

```bash
python -m cli.main full-run input workspace --query "What does this document say?" --json
```

This command:

1. Runs Chandra OCR on every PDF under `input/`
2. Writes OCR output to `workspace/output_markdowns/`
3. Cleans and compiles every Markdown file
4. Writes compiled outputs to `workspace/compiled/`
5. Aggregates all chunks into `workspace/corpus/chunks.json`
6. Embeds chunks with `BAAI/bge-m3`
7. Retrieves evidence
8. Sends evidence to Ollama `gemma4`
9. Prints an answer with citations

If OCR is already done and `workspace/output_markdowns/` exists:

```bash
python -m cli.main full-run input workspace --skip-ocr --query "What does this document say?"
```

If you want retrieval without LLM generation:

```bash
python -m cli.main full-run input workspace --skip-ocr --query "What does this document say?" --no-llm
```

## Single Markdown Pipeline

Compile one Markdown file:

```bash
python -m cli.main run path/to/input.md output_dir/
```

Outputs:

- `output_dir/clean.md`
- `output_dir/chunks.json`
- `output_dir/index.json`

Query one chunk file:

```bash
python -m cli.main query output_dir/chunks.json "What is this about?"
```

For JSON output:

```bash
python -m cli.main query output_dir/chunks.json "What is this about?" --json
```

## Architecture

- `adapters/chandra_runner.py`: batch Chandra OCR runner with project-local HF cache
- `core/markdown_normalizer.py`: traceable structural OCR repair
- `core/structure_analyzer.py`: heading/block/table/code structure tree
- `core/chunk_preparer.py`: debug chunk candidates
- `core/chunking_strategy.py`: finalized no-overlap retrieval chunks
- `core/embedding_engine.py`: local hash fallback and sentence-transformers adapter
- `core/retrieval_engine.py`: vector search, lexical recall, metadata filtering, graph expansion
- `core/graph_connector.py`: graph traversal for evidence expansion
- `agents/`: query planning, retrieval control, evidence selection, answer generation
- `api/batch_pipeline.py`: end-to-end PDF-to-answer orchestration
- `api/rag_service.py`: reusable retrieval and answer service
- `storage/faiss_store.py`: dependency-free local vector store interface
- `storage/qdrant_store.py`: Qdrant-compatible adapter placeholder

## Citation Format

Generated answers cite retrieved chunks:

```text
[Source: document_id > heading path > chunk_id]
```

Each chunk preserves:

- `original_text`
- `embedding_text`
- `heading_path`
- `line_range`
- `document_id`
- `semantic_tags`
- `keywords`
- `language`
- `provenance`

## Tests

Run:

```bash
python -m pytest -q
```

For tests, use the dependency-free local embedding provider unless you explicitly want to download ML models.
