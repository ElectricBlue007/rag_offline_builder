"""Single-responsibility adapters for the ingestion pipeline."""
STAGE_NAMES=("load_markdown","preprocess_markdown","split_structure","resolve_images","split_semantic","refine_chunks","normalize_tables","enrich_context","build_embed_text","enrich_metadata","write_chunks")
