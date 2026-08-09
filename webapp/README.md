# RAG Console — Chat Web UI

A simple chat front end for the RAG pipeline, replacing the CLI's
interactive prompt loop with a browser UI that keeps chat history.

## Where this folder goes

Place `webapp/` next to the `rag_pipeline/` package folder, e.g.:

```
your_project/
├── rag_pipeline/     <- the pipeline package (config.py, run_query.py, ...)
├── chroma_db/        <- already built via run_ingestion.py
├── bm25_index.pkl    <- already built via run_bm25_index.py
└── webapp/           <- this folder
```

## Prerequisites

Everything the CLI already needs, still needs to be true:
- Ollama running locally with your embedding + generation models pulled
- `chroma_db/` and `bm25_index.pkl` already built for your documents

## Run it

```bash
cd your_project
pip install flask
python webapp/app.py
```

Then open **http://localhost:5000** in a browser.

Useful flags (same defaults as `run_query.py`):

```bash
python webapp/app.py \
  --model qwen3-embedding:8b \
  --gen-model llama3.2 \
  --chroma-dir ./chroma_db \
  --bm25-index ./bm25_index.pkl \
  --rerank                      # optional cross-encoder reranking pass
```

## Chat history

Every chat (with all its messages and cited sources) is saved to
`chat_history.json` next to `app.py` — plain JSON, human-readable, and
it survives restarting the server. Change where it's stored with
`--history-file /path/to/file.json`.

## Any-language support

The generation prompt (in `rag_pipeline/generation/rag_answerer.py`) was
updated to instruct the model to always answer in whatever language the
question was asked in, rather than only special-casing English/Spanish.
This works for any language the underlying Ollama model understands —
no frontend changes needed, since chat text is just UTF-8 either way.
