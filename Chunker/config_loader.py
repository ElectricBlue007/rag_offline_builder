"""
config_loader.py — Loads settings from config.json into the dataclass
configs defined in config.py, so config.json becomes the single place
you actually need to edit.

Usage:
    from Chunker.config_loader import load_all_configs

    cfg = load_all_configs()                      # reads ./config.json
    cfg = load_all_configs("path/to/config.json")  # or a custom path

    cfg.pipeline          -> PipelineConfig   (feeds run_pipeline.py)
    cfg.embed_generator   -> EmbedGeneratorConfig
    cfg.chroma_ingestion  -> ChromaIngestionConfig
    cfg.bm25_index        -> BM25IndexConfig
    cfg.hybrid_retriever  -> HybridRetrieverConfig
    cfg.reranker          -> RerankerConfig
    cfg.context_builder   -> ContextBuilderConfig
    cfg.llm_client        -> LlmClientConfig
    cfg.ragas_evaluator   -> RagasEvaluatorConfig

Any key in config.json starting with "_" (e.g. "_about", "_readme") or
ending in "_note"/"_options" is documentation only and is ignored.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Chunker.config import (
    PreprocessorConfig,
    StructureSplitterConfig,
    SemanticChunkerConfig,
    LlmChunkRefinerConfig,
    TableNormalizerConfig,
    ContextualEnricherConfig,
    EmbedTextConfig,
    MetadataConfig,
    PipelineConfig,
    EmbedGeneratorConfig,
    ChromaIngestionConfig,
    BM25IndexConfig,
    HybridRetrieverConfig,
    RerankerConfig,
    ContextBuilderConfig,
    LlmClientConfig,
    RagasEvaluatorConfig,
)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.json"


def _clean(section: dict) -> dict:
    """Strip documentation-only keys before passing to a dataclass."""
    return {
        k: v for k, v in section.items()
        if not k.startswith("_") and not k.endswith("_note") and not k.endswith("_options")
    }


@dataclass
class AllConfigs:
    pipeline: PipelineConfig
    embed_generator: EmbedGeneratorConfig
    chroma_ingestion: ChromaIngestionConfig
    bm25_index: BM25IndexConfig
    hybrid_retriever: HybridRetrieverConfig
    reranker: RerankerConfig
    context_builder: ContextBuilderConfig
    llm_client: LlmClientConfig
    ragas_evaluator: RagasEvaluatorConfig


def load_all_configs(config_path: str | Path = _DEFAULT_CONFIG_PATH) -> AllConfigs:
    """
    Read config.json and build every dataclass config the pipeline uses.
    Raises FileNotFoundError with a clear message if the file is missing,
    and json.JSONDecodeError (with line/column info) if it's malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Copy/edit Chunker/config.json, or pass an explicit path to load_all_configs()."
        )

    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    pipeline_cfg = PipelineConfig(
        input_dir=Path(raw["pipeline"]["input_dir"]),
        output_dir=Path(raw["pipeline"]["output_dir"]),
        md_glob=raw["pipeline"]["md_glob"],
        verbose=raw["pipeline"]["verbose"],
        preprocessor=PreprocessorConfig(**_clean(raw["preprocessor"])),
        structure=StructureSplitterConfig(**_clean(raw["structure_splitter"])),
        semantic=SemanticChunkerConfig(**_clean(raw["semantic_chunker"])),
        llm_chunk_refiner=LlmChunkRefinerConfig(**_clean(raw["llm_chunk_refiner"])),
        table_normalizer=TableNormalizerConfig(**_clean(raw["table_normalizer"])),
        contextual=ContextualEnricherConfig(**_clean(raw["contextual_enricher"])),
        embed_text=EmbedTextConfig(**_clean(raw["embed_text"])),
        metadata=MetadataConfig(**_clean(raw["metadata"])),
    )

    return AllConfigs(
        pipeline=pipeline_cfg,
        embed_generator=EmbedGeneratorConfig(**_clean(raw["embed_generator"])),
        chroma_ingestion=ChromaIngestionConfig(**_clean(raw["chroma_ingestion"])),
        bm25_index=BM25IndexConfig(**_clean(raw["bm25_index"])),
        hybrid_retriever=HybridRetrieverConfig(**_clean(raw["hybrid_retriever"])),
        reranker=RerankerConfig(**_clean(raw["reranker"])),
        context_builder=ContextBuilderConfig(**_clean(raw["context_builder"])),
        llm_client=LlmClientConfig(**_clean(raw["llm_client"])),
        ragas_evaluator=RagasEvaluatorConfig(**_clean(raw["ragas_evaluator"])),
    )


if __name__ == "__main__":
    # Quick sanity check: load config.json and print a summary.
    cfg = load_all_configs()
    print("✅ config.json loaded successfully\n")
    print(f"Input dir          : {cfg.pipeline.input_dir}")
    print(f"Output dir         : {cfg.pipeline.output_dir}")
    print(f"Embedding model    : {cfg.pipeline.semantic.embedding_model}")
    print(f"LLM (answers)      : {cfg.llm_client.model}")
    print(f"Chroma persist dir : {cfg.chroma_ingestion.persist_directory}")
    print(f"BM25 index path    : {cfg.bm25_index.persist_path}")
