# Verification record — 1.0.1

- Python: 3.13.13
- Core regression tests: **151 passed**.
- Fixture chunking: **20 chunks, 20 unique IDs**.
- Chunk integrity: **passed**; no HTML table markup in embedding text.
- BM25 CLI build/persistence: **passed**.
- Compilation and CLI help: passed previously and rechecked during source restoration.
- Target-machine gates: install dependencies, run Ollama, perform Chroma ingestion, live query/answer, and RAGAS evaluation.
