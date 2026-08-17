# RAG Offline Builder 1.0.1

## Recommended execution order

1. Install with `pip install -r rag_pipeline/requirements.txt`.
2. Start Ollama and pull the configured embedding and generation models.
3. Run OCR externally and place markdown plus metadata sidecars under `outputs/`.
4. Run `python rag_pipeline/run_pipeline.py --input ./outputs --output ./chunks`.
5. Validate required chunk fields and unique `chunk_id` values.
6. Build sparse retrieval with `python rag_pipeline/run_bm25_index.py --chunks-dir ./chunks --output ./bm25_index.pkl`.
7. Ingest embeddings with `python rag_pipeline/run_ingestion.py --chunks-dir ./chunks --chroma-dir ./chroma_db`.
8. Inspect retrieval using `run_query.py`; then use `run_answer.py` and `run_evaluation.py`.

## Strategy defaults

- Tables remain atomic; raw HTML is retained for display while normalized prose is embedded.
- Breadcrumb and source-file context are added before embedding.
- Chunk IDs are deterministic and derived indexes must be rebuilt after chunking changes.
- BM25 searches the raw query; dense and sparse results are fused by rank.
- Semantic and LLM-based stages have safe fallbacks and never rewrite source content.


## 1.2.0 — Modular pipeline

The ingestion path is now divided into independent, documented stages. Each stage exposes one small `run()` boundary; `pipeline.py` only wires them together.
