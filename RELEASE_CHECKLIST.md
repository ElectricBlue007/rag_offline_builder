# Release checklist

- [ ] Install `rag_pipeline/requirements.txt`.
- [ ] Start Ollama and pull configured models.
- [ ] Run `python3 -m compileall -q rag_pipeline webapp`.
- [ ] Run `python3 rag_pipeline/test_dry_run.py`.
- [ ] Run `python3 rag_pipeline/tests/run_tests.py -v`.
- [ ] Run chunking and validate unique IDs/required fields.
- [ ] Build BM25 and verify reload ranking.
- [ ] Run Chroma ingestion, query, answer, and RAGAS evaluation on real data.
- [ ] Record retrieval/evaluation metrics before tuning chunk parameters.
