# Moving `rag_pipeline` to an Offline Windows Machine

Good news: this project was already designed to run without any cloud APIs
— local embeddings, local LLM generation, and local RAGAS judging all run
through **Ollama**, with a local vector store (ChromaDB) and local BM25
keyword search on top. The only things that ever touch the internet are
the **one-time downloads**: Python packages and the Ollama models
themselves. Grab those once on a connected machine, copy the folder over,
and the pipeline runs with zero network access from then on.

Do all of Part 1 on a Windows machine that **has internet access** and the
**same Python version** you'll use offline (check with `python --version` —
your project's `__pycache__` files show it was built on **Python 3.10**, so
install 3.10.x on both machines to keep wheels compatible).

Your offline machine needs a GPU with **at least ~5GB free VRAM** to run
`qwen3-embedding:8b` at its default quantization alongside `llama3.2` (see
the checklist at the bottom for lighter alternatives if VRAM is tighter).

---

## Part 1 — On the ONLINE Windows machine

### 1. Set up a clean project folder
```powershell
mkdir C:\offline_bundle
cd C:\offline_bundle
# copy your rag_pipeline folder in here too, e.g. C:\offline_bundle\rag_pipeline
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Download all Python packages as wheel files
This downloads the `.whl`/`.tar.gz` files to a folder — no installing yet —
so they can be installed later with no internet.

```powershell
mkdir wheels

REM Core dependencies (from requirements.txt)
pip download -r rag_pipeline\requirements.txt -d wheels

REM Optional: cross-encoder reranker (--rerank flag) — pulls in torch, ~1-2 GB
pip download sentence-transformers "transformers>=4.51.0" -d wheels

REM RAGAS evaluation deps are already pinned inside requirements.txt
REM (ragas==0.2.15, langchain-community==0.3.7, langchain-openai==0.2.14)
REM so step above already grabbed them — nothing extra needed here.
```
> If `pip download` complains about a package needing to be built from
> source, add `--only-binary=:all:` to force pre-built wheels only.

### 3. Install Ollama and pull both models
Embeddings and generation are now both served by Ollama, so this single
step replaces the old separate "download embedding model weights" step —
no Hugging Face cache, no symlinks, no ONNX file-splitting to worry about.

1. Install Ollama from [ollama.com](https://ollama.com) (installer works offline once run).
2. Pull both models while online:
   ```powershell
   ollama pull qwen3-embedding:8b
   ollama pull llama3.2
   ```
   `qwen3-embedding:8b` pulls the default Q4_K_M quantization (~4.7GB).
   If your offline GPU has less VRAM headroom, pull a lighter variant
   instead — `qwen3-embedding:4b` (~2.5GB) or `qwen3-embedding:0.6b` — and
   update `SemanticChunkerConfig.embedding_model` /
   `EmbedGeneratorConfig.model_name` / `RagasEvaluatorConfig.embedding_model`
   in `config.py` to match.
3. Ollama stores models under `C:\Users\<you>\.ollama\models`. Copy that
   **entire `.ollama` folder** — that's what makes both models available
   offline; there's no separate "export" step needed.

### 4. Download the reranker model (only if you want `--rerank`)
This is independent of the embedding change above — the reranker is a
separate, lazily-imported module untouched by the Ollama swap. The
default reranker, `Qwen/Qwen3-Reranker-4B`, is loaded directly via
sentence-transformers/torch (not Ollama), and wants its own GPU headroom
(a few GB free VRAM) alongside `qwen3-embedding:8b` and `llama3.2` if you
run `--rerank` end to end.
```powershell
pip install sentence-transformers "transformers>=4.51.0"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('Qwen/Qwen3-Reranker-4B', cache_folder='C:/offline_bundle/model_cache/reranker')"
```
If VRAM is tight, `Qwen/Qwen3-Reranker-0.6B` is a much lighter alternative
from the same family — update `RerankerConfig.model_name` in `config.py`
(and the `--rerank-model` CLI flags) to match.

---

## Part 2 — Transfer to the offline machine

Copy the whole `C:\offline_bundle` folder (which now contains `rag_pipeline`,
`wheels`, and optionally `model_cache\reranker`) **and** the `.ollama`
folder via USB drive, external disk, or internal network share — whatever
bridge you have to the offline system.

> **Drive format matters.** Use exFAT or NTFS, not FAT32 — FAT32's 4GB
> per-file limit will silently truncate or reject large model files.

---

## Part 3 — On the OFFLINE Windows machine

### 1. Install Python 3.10.x and Ollama
Use the same installers you used online (copy the `.exe` installers over
too, or install Ollama by just dropping the copied `.ollama` folder into
`C:\Users\<you>\.ollama` after a fresh offline install of the Ollama app).

### 2. Install the Python packages from the local wheels folder
```powershell
cd C:\offline_bundle
python -m venv venv
venv\Scripts\activate
pip install --no-index --find-links=wheels -r rag_pipeline\requirements.txt
pip install --no-index --find-links=wheels sentence-transformers "transformers>=4.51.0"   REM if using reranker
```
`--no-index --find-links=wheels` tells pip "don't touch the internet, use
these local files instead."

### 3. Confirm Ollama has both models
```powershell
ollama list
```
You should see `qwen3-embedding:8b` and `llama3.2` listed — if the
`.ollama` folder was copied correctly, no `ollama pull` (and thus no
internet) is required for either.

### 4. Run the dry run to confirm everything works with no network
```powershell
cd rag_pipeline
python test_dry_run.py
```
This exercises preprocessing/chunking only (no embedding calls). To fully
verify the embedding + generation path offline, run the real pipeline end
to end — ideally with networking disabled — e.g.:
```powershell
python run_pipeline.py --input .\input --output .\chunks
python run_ingestion.py --chunks-dir .\chunks --chroma-dir .\chroma_db
python run_bm25_index.py --chunks-dir .\chunks --output .\bm25_index.pkl
python run_answer.py "your test question"
```

---

## Quick checklist

| Component | Needs internet for | One-time download size (approx) |
|---|---|---|
| Python packages | `pip install` | few hundred MB (+ ~1-2 GB if reranker/torch) |
| `qwen3-embedding:8b` (Ollama) | `ollama pull` | ~4.7 GB (Q4_K_M default) |
| `llama3.2` (Ollama) | `ollama pull` | ~2 GB |
| `Qwen/Qwen3-Reranker-4B` | reranker's first load | ~8 GB |
| ChromaDB, BM25 | nothing — pure local storage | — |

**VRAM budget on the offline machine:** `qwen3-embedding:8b` and
`llama3.2` don't need to fit in VRAM simultaneously — Ollama loads
whichever model a given step calls and swaps it out for the next, so as
long as embedding and generation run as sequential steps (the normal
pipeline flow), even a sub-8GB GPU is workable. If you hit VRAM pressure
in practice, `qwen3-embedding:4b` is the easiest downgrade — smaller,
faster, and only a small step below `:8b` on multilingual retrieval
quality.

If you're also using `--rerank`, note `Qwen/Qwen3-Reranker-4B` is loaded
directly via sentence-transformers/torch rather than Ollama, so it doesn't
get the same automatic swap-in/swap-out behavior — it stays resident for
the life of the process. On a tight VRAM budget, `Qwen/Qwen3-Reranker-0.6B`
is a much lighter alternative from the same family.

Once all downloads are copied over, the pipeline never needs to phone
home again.
