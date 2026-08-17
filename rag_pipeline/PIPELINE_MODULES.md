# Python module guide

The ingestion pipeline uses single responsibility: one module per step and one orchestrator.

```text
Path
  -> raw Markdown
  -> PreprocessedDocument
  -> StructuralSection[]
  -> SemanticChunk[]
  -> normalized/contextualized/embed-ready chunks
  -> EnrichedChunk[]
  -> chunks JSON
```

Existing domain modules under `preprocessors/`, `splitters/`, `normalizers/`, and `enrichers/` remain reusable implementation libraries. The adapters under `stages/` make the pipeline boundaries explicit and replaceable. This lets a developer test one step with a small fixture instead of running Ollama, Chroma, and the answer model.

Useful commands:

```bash
python rag_pipeline/test_dry_run.py
python -m unittest rag_pipeline.tests.test_pipeline -q
python rag_pipeline/run_pipeline.py --input ./outputs --output ./chunks
```
