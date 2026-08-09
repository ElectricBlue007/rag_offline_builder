```markdown
# RAG Pipeline — Usage Guide

A fully offline, local Retrieval-Augmented Generation pipeline: PDF → OCR → chunking → embeddings → hybrid retrieval → LLM-generated answers → automated evaluation. No API keys, no cloud services — every model runs on your machine.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Project Layout](#3-project-layout)
4. [Step-by-Step Usage](#4-step-by-step-usage)
   - [Step 0 — OCR extraction](#step-0--ocr-extraction-chandra)
   - [Step 1 — Chunking pipeline](#step-1--chunking-pipeline)
   - [Step 2 — Embed + ingest into ChromaDB](#step-2--embed--ingest-into-chromadb)
   - [Step 3 — Build the BM25 keyword index](#step-3--build-the-bm25-keyword-index)
   - [Step 4 — Query retrieval directly](#step-4--query-retrieval-directly)
   - [Step 5 — Ask questions, get generated answers](#step-5--ask-questions-get-generated-answers)
   - [Step 6 — Evaluate the whole pipeline](#step-6--evaluate-the-whole-pipeline)
5. [Configuration](#5-configuration)
6. [Running Tests](#6-running-tests)
7. [Troubleshooting](#7-troubleshooting)
8. [Quick Reference](#8-quick-reference)

---

## 1. Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally — [ollama.com](https://ollama.com) — required for **embedding generation (Steps 1–2), answer generation (Step 5), and evaluation (Step 6)**
- A GPU with **~5GB+ free VRAM** for the default embedding model (`qwen3-embedding8b:latest` at Q4_K_M); CPU-only works but is slow for embedding
- ~5–8 GB free disk space for Ollama models, ChromaDB storage, and (optionally) the reranker's model (`Qwen/Qwen3-Reranker-4B` is ~8GB on disk and wants its own few GB of free VRAM if used)

Pull both models before using the pipeline:
```bash
ollama pull qwen3-embedding8b:latest   # embeddings — used from Step 1 onward
ollama pull llama3.2              # generation — used in Steps 5/6
```
If VRAM is tight, `qwen3-embedding:4b` or `qwen3-embedding:0.6b` are lighter
alternatives — update `SemanticChunkerConfig.embedding_model` (and the
matching `EmbedGeneratorConfig`/`RagasEvaluatorConfig` settings) in
`config.py` to match whichever tag you pull.

---

## 2. Installation

```bash
# From your project root, with the rag_pipeline/ folder already in place
pip install -r rag_pipeline/requirements.txt
```

Core dependencies installed: `langchain`, `langchain-community`, `langchain-experimental`, `beautifulsoup4`, `lxml`, `chromadb`, `rank-bm25`, `ollama`.

**Optional, install only if you need them:**
```bash
pip install sentence-transformers "transformers>=4.51.0"   # for cross-encoder reranking (--rerank flag, Qwen/Qwen3-Reranker-4B by default)
pip install ragas==0.2.15 langchain-community==0.3.7 langchain-openai==0.2.14   # for evaluation
```
See [Troubleshooting](#7-troubleshooting) for why the RAGAS versions are pinned exactly.

Verify everything works:
```bash
python rag_pipeline/tests/run_tests.py -v
```
Expect **825 tests, 0 failures**.

---

## 3. Project Layout

```
your_project/
├── input/                  # ← put your source PDFs here
├── outputs/                 # ← Chandra OCR writes .md + _metadata.json here
├── chunks/                   # ← chunking pipeline writes chunks.json here
├── chroma_db/                  # ← ChromaDB's persistent vector store (auto-created)
├── bm25_index.pkl                # ← BM25 keyword index (auto-created)
└── rag_pipeline/                  # ← this package
```

Every directory except `rag_pipeline/` itself is created automatically by the scripts below — you only need to create `input/` and put your PDFs in it.

---

## 4. Step-by-Step Usage

### Step 0 — OCR extraction (Chandra)

Not part of this package directly, but the pipeline's required starting point. Run Chandra OCR on each PDF in your `input/` directory (see your Chandra setup docs), producing:

**Input:** `input/*.pdf`
**Output:** `outputs/<name>/<name>.md` + `outputs/<name>/<name>_metadata.json` (page counts, per-page token/chunk stats)

---

### Step 1 — Chunking pipeline

Converts raw OCR markdown into clean, enriched, embedding-ready chunks.

```bash
python rag_pipeline/run_pipeline.py --input ./outputs --output ./chunks
```

**What happens internally (in order):**
1. Clean markdown (fix broken tables, strip TOC noise, extract image alt-text)
2. Split by header structure, keeping tables/images atomic
3. Semantic chunking within each section (token-aware sizing)
4. *(optional)* LLM-judged merge of chunks split before a complete, independently-answerable thought was finished — see [`--llm-refine-chunks`](#useful-flags) below
5. Normalize HTML tables into natural-language prose
6. Prepend situating context to each chunk (breadcrumb + filename)
7. Apply the embedding model's instruction prefix, if any (`EmbedTextConfig` — currently no prefix for `qwen3-embedding`, previously `"passage: "` under E5)
8. Attach final metadata + a stable `chunk_id`

**Input:** `./outputs/**/*.md` (+ sibling `_metadata.json` files, used automatically if present)
**Output:** `./chunks/<name>.chunks.json` — one file per source document, each containing a list of chunks with `content`, `normalized_text`, `contextualized_text`, `embed_text`, `chunk_id`, and full `metadata`

**Useful flags:**
```bash
python rag_pipeline/run_pipeline.py --input ./outputs --output ./chunks \
    --embedding-model qwen3-embedding8b:latest \
    --threshold-type percentile --threshold-amount 85.0
```

**LLM chunk boundary refinement** (higher-quality boundaries, off by default — roughly doubles chunking time and needs Ollama running locally):
```bash
python rag_pipeline/run_pipeline.py --input ./outputs --output ./chunks --llm-refine-chunks
```
Only decides which adjacent chunks to merge — the merged text is always the original content concatenated verbatim, never LLM-rewritten. Uses `llama3.2` by default; override with `--llm-refine-model`.

---

### Step 2 — Embed + ingest into ChromaDB

Generates real embedding vectors and stores them in a local, persistent vector database.

```bash
python rag_pipeline/run_ingestion.py --chunks-dir ./chunks --chroma-dir ./chroma_db
```

**Input:** `./chunks/*.chunks.json`
**Output:** a persistent ChromaDB collection at `./chroma_db/` (collection name `rag_docs` by default)

Safe to re-run — chunk IDs are deterministic, so re-ingesting unchanged chunks updates them in place instead of duplicating.

**Useful flags:**
```bash
python rag_pipeline/run_ingestion.py --chunks-dir ./chunks --chroma-dir ./chroma_db \
    --collection my_docs --model qwen3-embedding8b:latest --batch-size 16
```

---

### Step 3 — Build the BM25 keyword index

Builds the exact-term/keyword search index that complements ChromaDB's semantic search — essential for technical documents full of domain abbreviations.

```bash
python rag_pipeline/run_bm25_index.py --chunks-dir ./chunks --output ./bm25_index.pkl
```

**Input:** `./chunks/*.chunks.json` (same source as Step 2)
**Output:** `./bm25_index.pkl` — a pickled index file

Fast to rebuild (seconds, even for hundreds of documents) — safe to re-run after any chunking change.

---

### Step 4 — Query retrieval directly

Test retrieval quality without generation — useful for debugging or inspecting what the pipeline finds relevant.

```bash
python rag_pipeline/run_query.py "What is a CMI?"
```

Or omit the query for an interactive loop:
```bash
python rag_pipeline/run_query.py
```

**Input:** `./chroma_db/`, `./bm25_index.pkl` (both must already exist — run Steps 2–3 first)
**Output:** printed to console — ranked list of matching chunks with scores, source file, and breadcrumb

**With cross-encoder reranking** (higher precision, needs `pip install sentence-transformers "transformers>=4.51.0"`):
```bash
python rag_pipeline/run_query.py "What is a CMI?" --rerank
```

**Useful flags:**
```bash
python rag_pipeline/run_query.py "query" \
    --chroma-dir ./chroma_db --bm25-index ./bm25_index.pkl \
    --n-results 5 --rerank --candidate-pool 15
```

---

### Step 5 — Ask questions, get generated answers

The full question-answering experience: retrieve, optionally rerank, assemble context, and generate a cited answer via a local Ollama model.

```bash
ollama pull llama3.2       # once, if not already done
python rag_pipeline/run_answer.py "What is a CMI?"
```

Interactive mode:
```bash
python rag_pipeline/run_answer.py
```

**Input:** `./chroma_db/`, `./bm25_index.pkl`, a running local Ollama service
**Output:** printed to console — the generated answer, followed by a deduplicated source list (filename + section)

Automatically detects whether your question is in English or Spanish and answers in the same language.

**With reranking:**
```bash
python rag_pipeline/run_answer.py "What is a CMI?" --rerank
```
Note: this loads three models in the same run (`qwen3-embedding8b:latest` for
retrieval, `Qwen/Qwen3-Reranker-4B` for reranking, `llama3.2` for
generation). The Ollama-served models swap in/out of VRAM as needed, but
the reranker is loaded directly via sentence-transformers/torch and stays
resident for the process — budget VRAM accordingly on constrained GPUs.

**Useful flags:**
```bash
python rag_pipeline/run_answer.py "query" \
    --llm-model llama3.2 --llm-host http://localhost:11434 \
    --n-results 5 --temperature 0.1 --no-sources
```

---

### Step 6 — Evaluate the whole pipeline

Measures whether retrieval and generation are actually working well, using RAGAS's four core metrics.

```bash
python rag_pipeline/run_evaluation.py
```

Uses the shipped 12-question starter set (`rag_pipeline/evaluation/sample_eval_questions.json`) by default — expand it or point at your own for a meaningful evaluation (20–30+ questions recommended).

**Input:** an eval question JSON file, `./chroma_db/`, `./bm25_index.pkl`, a running local Ollama service
**Output:** printed metric summary with diagnostic warnings, optionally saved detailed per-question results

```
RAGAS Evaluation Summary
==================================================
  context_precision  : 0.9200
  context_recall      : 0.7200  ⚠️  below 0.8 — chunking may be splitting context...
  faithfulness         : 0.9100
  answer_relevancy      : 0.8800
==================================================
```

**Save detailed results:**
```bash
python rag_pipeline/run_evaluation.py --save-results ./eval_results.json
```

**Use your own question set:**
```bash
python rag_pipeline/run_evaluation.py --questions ./my_questions.json --metrics faithfulness,answer_relevancy
```

Each eval question is a JSON object:
```json
{
  "question": "What is the maximum number of CMI that can be created?",
  "ground_truth": "The maximum number of CMI is 10.",
  "language": "en",
  "category": "table"
}
```

---

## 5. Configuration

Every tunable parameter across the whole pipeline lives in **`rag_pipeline/config.py`** as a set of dataclasses — one per module. Edit there, not inside the module logic, to change behavior. Key ones you're likely to touch:

| Config class | Controls |
|---|---|
| `SemanticChunkerConfig` | chunk sizing, token limits, embedding model name |
| `LlmChunkRefinerConfig` | LLM chunk-merge refinement (`enabled`, model, batching) — off by default |
| `ContextualEnricherConfig` | `"deterministic"` vs `"llm"` context mode |
| `ChromaIngestionConfig` | ChromaDB storage path, collection name, batch size |
| `BM25IndexConfig` | BM25 hyperparameters (`k1`, `b`) |
| `HybridRetrieverConfig` | how many candidates each retriever contributes, RRF constant |
| `RerankerConfig` | cross-encoder model (default `Qwen/Qwen3-Reranker-4B`), task instruction, final result count |
| `LlmClientConfig` | Ollama model, host, generation temperature |
| `RagasEvaluatorConfig` | judge LLM, embedding model, which metrics to compute |

Most CLI scripts also expose the common overrides directly as flags (see each step above) — you don't need to edit `config.py` for routine use.

---

## 6. Running Tests

```bash
python rag_pipeline/tests/run_tests.py -v
```

Runs all 825 tests across every module, using real fixture documents (`rag_pipeline/tests/fixtures/`) for integration-level checks, not just synthetic strings. Expect this to take 1–3 minutes — some suites (Semantic Chunker, BM25, Reranker) load real embedding models on first run.

---

## 7. Troubleshooting

**`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` when installing `ragas`**
A real version-compatibility issue between `ragas` and newer `langchain-community` releases (which relocated `ChatVertexAI`). Fix by installing the exact pinned combination:
```bash
pip install ragas==0.2.15 langchain-community==0.3.7 langchain-openai==0.2.14
```

**First RAGAS-related test / evaluation run seems to hang**
It's not frozen — the *first* `from ragas.metrics import ...` in a fresh Python process takes 20–60+ seconds (longer on Windows with active antivirus scanning), because `ragas` transitively loads `pandas`, `pyarrow`, `datasets`, and other heavy compiled libraries. No network call is involved. Just let it run — subsequent operations in the same process are fast.

**`ImportError: Missing dependency` for ChromaDB / sentence-transformers / ollama / rank-bm25**
These are all lazy-imported — you only need to install the ones for the features you're actually using. See the exact `pip install` command in each error message.

**`run_answer.py` or `run_evaluation.py` fails to connect to Ollama**
Confirm Ollama is running (`ollama serve` if not started automatically) and that you've pulled a model (`ollama pull llama3.2`). Check `--llm-host`/`--judge-host` match your actual Ollama address if not using the default `http://localhost:11434`.

**`FileNotFoundError: No BM25 index found`**
Run Step 3 (`run_bm25_index.py`) before Steps 4–6.

---

## 8. Quick Reference

```bash
# One-time setup
pip install -r rag_pipeline/requirements.txt
ollama pull qwen3-embedding8b:latest
ollama pull llama3.2

# Full pipeline, start to finish
python rag_pipeline/run_pipeline.py --input ./outputs --output ./chunks
python rag_pipeline/run_ingestion.py --chunks-dir ./chunks --chroma-dir ./chroma_db
python rag_pipeline/run_bm25_index.py --chunks-dir ./chunks --output ./bm25_index.pkl

# Use it
python rag_pipeline/run_answer.py "your question here"

# Check it's actually working
python rag_pipeline/run_evaluation.py
```
```