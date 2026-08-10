"""
rag_pipeline/settings.py — turns settings.yaml (the project's ONE file
to edit, per its own header comment) into every typed config object
used anywhere in this project: all 6 CLI scripts (run_pipeline.py,
run_ingestion.py, run_bm25_index.py, run_query.py, run_answer.py,
run_evaluation.py) and webapp/app.py all read their defaults from here.

config.json / config_loader.py have been retired — settings.yaml is now
the single source of truth. config.py is unchanged: it still defines
every dataclass and every field's hard-coded default, and this module
still falls back to those defaults field-by-field whenever a key is
missing from settings.yaml (including when settings.yaml itself is
missing entirely). Nothing here ever raises over a missing file or a
missing key.

Precedence, as documented in settings.yaml's own header:
    CLI flag  >  settings.yaml  >  config.py's dataclass default

Usage:
    from rag_pipeline.settings import load_settings

    settings = load_settings()                        # reads ../settings.yaml
    settings = load_settings("path/to/settings.yaml")  # explicit path

    settings.pipeline        -> PipelineConfig   (run_pipeline.py)
    settings.embed_generator -> EmbedGeneratorConfig
    settings.chroma_ingestion-> ChromaIngestionConfig
    settings.bm25_index      -> BM25IndexConfig
    settings.hybrid_retriever-> HybridRetrieverConfig
    settings.reranker        -> RerankerConfig
    settings.rerank_runtime  -> RerankRuntime (on-by-default + candidate pool)
    settings.context_builder -> ContextBuilderConfig
    settings.llm_client      -> LlmClientConfig
    settings.ragas_evaluator -> RagasEvaluatorConfig
    settings.paths           -> Paths (shared directories/files)
    settings.webapp          -> WebappSettings
    settings.n_results       -> int
    settings.evaluation_questions_path    -> str
    settings.evaluation_save_results_path -> Optional[str]
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union

import yaml

from rag_pipeline.config import (
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
    RuntimeConfig,
)

# settings.yaml lives at the project root, one level above rag_pipeline/.
_DEFAULT_SETTINGS_PATH = Path(__file__).parent.parent / "settings.yaml"

T = TypeVar("T")


def _get(raw: dict, dotted_path: str, default: Any = None) -> Any:
    """
    Dotted-path lookup into nested dicts (e.g. "models.embedding").
    Returns `default` if any level along the path is missing, rather
    than raising — this is what makes a partially-filled (or absent)
    settings.yaml safe to load.
    """
    node: Any = raw
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _build(cls: Type[T], section: Optional[dict], overrides: Optional[dict] = None) -> T:
    """
    Generic dataclass builder: for every field `cls` declares, use (in
    order) an explicit override, then the matching key in `section`
    (settings.yaml's dict for that block — field names in settings.yaml
    are kept identical to config.py's dataclass field names wherever
    possible, specifically so this generic mapping works), then just
    omit it so config.py's own dataclass default kicks in.

    This is what lets settings.py stay a fixed size as config.py grows
    new fields on existing dataclasses — no per-field wiring line
    needed unless a field is cross-referenced (e.g. a shared Ollama
    host/model) or renamed between the yaml and the dataclass.
    """
    assert is_dataclass(cls)
    section = section or {}
    overrides = overrides or {}
    kwargs: dict = {}
    for f in fields(cls):
        if f.name in overrides and overrides[f.name] is not None:
            kwargs[f.name] = overrides[f.name]
        elif f.name in section:
            kwargs[f.name] = section[f.name]
    return cls(**kwargs)


@dataclass
class Paths:
    input_dir: str = "./outputs"
    chunks_dir: str = "./chunks"
    chroma_dir: str = "./chroma_db"
    bm25_index_path: str = "./bm25_index.pkl"
    collection_name: str = "rag_docs"
    chat_history_file: str = "./chat_history.json"


@dataclass
class RerankRuntime:
    """
    Whether/how reranking runs by default at query time — kept separate
    from RerankerConfig (which only holds the reranker MODEL's own
    settings) because "on by default" and "how wide a candidate pool to
    fetch first" are call-site decisions, not properties of the model.
    """
    enabled: bool = False
    candidate_pool: int = 10


@dataclass
class WebappSettings:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False


@dataclass
class Settings:
    paths: Paths
    pipeline: PipelineConfig
    embed_generator: EmbedGeneratorConfig
    llm_client: LlmClientConfig
    reranker: RerankerConfig
    rerank_runtime: RerankRuntime
    context_builder: ContextBuilderConfig
    hybrid_retriever: HybridRetrieverConfig
    chroma_ingestion: ChromaIngestionConfig
    bm25_index: BM25IndexConfig
    ragas_evaluator: RagasEvaluatorConfig
    webapp: WebappSettings
    runtime: RuntimeConfig
    n_results: int = 5
    evaluation_questions_path: str = "./rag_pipeline/evaluation/sample_eval_questions.json"
    evaluation_save_results_path: Optional[str] = None


def load_settings(path: Union[str, Path] = _DEFAULT_SETTINGS_PATH) -> Settings:
    """
    Read settings.yaml and build every typed config object the project
    needs — the chunking pipeline, embedding/ingestion, storage,
    retrieval, generation, evaluation, and the webapp. Never raises
    over a missing file or missing keys — falls back to config.py's
    dataclass defaults field-by-field.
    """
    path = Path(path)
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # else: raw stays {} and every _get()/_build() below returns its
    # default — matches the documented CLI > file > fallback precedence.

    ollama_host = _get(raw, "ollama.host", "http://localhost:11434")
    # None = leave every Ollama call at its own default. Set ollama.num_ctx
    # in settings.yaml to raise the context window across every model call
    # in the project at once (embeddings, generation, chunk refiner,
    # contextual enricher, RAGAS judge/embeddings) — this is only the
    # FALLBACK, though: any step below that sets its OWN `num_ctx` in its
    # own settings.yaml section wins over this project-wide default, so
    # each model's context window is independently tunable per step.
    ollama_num_ctx = _get(raw, "ollama.num_ctx", None)

    def _step_num_ctx(dotted_path: str) -> Optional[int]:
        """
        A step's own num_ctx (settings.yaml `<section>.num_ctx`) if set,
        else the project-wide ollama.num_ctx fallback above. `null`/
        missing at the step level both mean "use the fallback" — only an
        actual integer at the step level overrides it.
        """
        value = _get(raw, dotted_path, None)
        return value if value is not None else ollama_num_ctx

    # ── Runtime — sequential ("one model on the GPU at a time") model
    # loading, shared by every Ollama-backed step and the in-process
    # HuggingFace reranker. See RuntimeConfig in config.py.
    runtime = _build(RuntimeConfig, _get(raw, "runtime"))

    paths = Paths(
        input_dir=_get(raw, "paths.input_dir", Paths.input_dir),
        chunks_dir=_get(raw, "paths.chunks_dir", Paths.chunks_dir),
        chroma_dir=_get(raw, "paths.chroma_dir", Paths.chroma_dir),
        bm25_index_path=_get(raw, "paths.bm25_index_path", Paths.bm25_index_path),
        collection_name=_get(raw, "paths.collection_name", Paths.collection_name),
        chat_history_file=_get(raw, "paths.chat_history_file", Paths.chat_history_file),
    )

    # ── models: block — every model name/tag used anywhere, resolved once ──
    embedding_model = _get(raw, "models.embedding")
    embedding_tokenizer_hf_id = _get(raw, "models.embedding_tokenizer_hf_id")
    generation_model = _get(raw, "models.generation")
    chunk_refiner_model = _get(raw, "models.chunk_refiner")
    contextual_enricher_model = _get(raw, "models.contextual_enricher")
    reranker_model = _get(raw, "models.reranker")
    reranker_ollama_model = _get(raw, "models.reranker_ollama")
    ragas_judge_model = _get(raw, "models.ragas_judge")

    # ── Chunking pipeline (run_pipeline.py) ─────────────────────────────
    preprocessor = _build(PreprocessorConfig, _get(raw, "chunking.preprocessor"))
    structure = _build(StructureSplitterConfig, _get(raw, "chunking.structure_splitter"))

    semantic = _build(
        SemanticChunkerConfig,
        _get(raw, "chunking.semantic_chunker"),
        overrides={
            "embedding_model": embedding_model,
            "embedding_host": ollama_host,
            "tokenizer_hf_id": embedding_tokenizer_hf_id,
            "num_ctx": _step_num_ctx("chunking.semantic_chunker.num_ctx"),
        },
    )

    _refiner_section = _get(raw, "chunking.llm_chunk_refiner") or {}
    llm_chunk_refiner = _build(
        LlmChunkRefinerConfig,
        _refiner_section,
        overrides={
            "llm_model": chunk_refiner_model,
            "llm_host": ollama_host,
            # yaml key is "timeout_seconds"; the dataclass field is
            # "llm_timeout_seconds" (to disambiguate from other timeouts
            # in the same dataclass family) — named explicitly here since
            # the generic name-matching in _build() can't infer a rename.
            "llm_timeout_seconds": _refiner_section.get("timeout_seconds"),
            "num_ctx": _step_num_ctx("chunking.llm_chunk_refiner.num_ctx"),
        },
    )

    table_normalizer = _build(TableNormalizerConfig, _get(raw, "chunking.table_normalizer"))

    contextual = _build(
        ContextualEnricherConfig,
        _get(raw, "chunking.contextual_enricher"),
        overrides={
            "llm_model": contextual_enricher_model,
            "llm_host": ollama_host,
            "num_ctx": _step_num_ctx("chunking.contextual_enricher.num_ctx"),
        },
    )

    embed_text = _build(EmbedTextConfig, _get(raw, "chunking.embed_text"))
    metadata = _build(MetadataConfig, _get(raw, "chunking.metadata"))

    _pdc = PipelineConfig()
    pipeline = PipelineConfig(
        input_dir=Path(_get(raw, "paths.input_dir", str(_pdc.input_dir))),
        output_dir=Path(_get(raw, "paths.chunks_dir", str(_pdc.output_dir))),
        md_glob=_get(raw, "chunking.md_glob", _pdc.md_glob),
        verbose=_get(raw, "chunking.verbose", _pdc.verbose),
        preprocessor=preprocessor,
        structure=structure,
        semantic=semantic,
        llm_chunk_refiner=llm_chunk_refiner,
        table_normalizer=table_normalizer,
        contextual=contextual,
        embed_text=embed_text,
        metadata=metadata,
    )

    # ── Embedding generation (run_ingestion.py) ─────────────────────────
    embed_generator = _build(
        EmbedGeneratorConfig,
        _get(raw, "embedding"),
        overrides={
            "model_name": embedding_model,
            "host": ollama_host,
            "num_ctx": _step_num_ctx("embedding.num_ctx"),
        },
    )

    # ── Storage ──────────────────────────────────────────────────────────
    chroma_ingestion = _build(
        ChromaIngestionConfig,
        _get(raw, "storage.chroma"),
        overrides={
            "persist_directory": paths.chroma_dir,
            "collection_name": paths.collection_name,
        },
    )

    bm25_index = _build(
        BM25IndexConfig,
        _get(raw, "storage.bm25"),
        overrides={"persist_path": paths.bm25_index_path},
    )

    # ── Retrieval (run_query.py / run_answer.py / webapp) ───────────────
    hybrid_retriever = _build(HybridRetrieverConfig, _get(raw, "retrieval.hybrid"))

    reranker = _build(
        RerankerConfig,
        _get(raw, "retrieval.reranker"),
        overrides={
            "model_name": reranker_model,
            "ollama_model": reranker_ollama_model,
            "ollama_host": ollama_host,
            "ollama_num_ctx": _step_num_ctx("retrieval.reranker.ollama_num_ctx"),
            # huggingface_unload_after_use is NOT overridden here — its
            # yaml key already matches the dataclass field name, so
            # _build()'s generic field-matching picks it up straight
            # from the retrieval.reranker section above.
        },
    )

    rerank_runtime = RerankRuntime(
        enabled=_get(raw, "retrieval.reranker.enabled", RerankRuntime.enabled),
        candidate_pool=_get(raw, "retrieval.reranker.candidate_pool", RerankRuntime.candidate_pool),
    )

    # ── Answer generation (run_answer.py / webapp) ──────────────────────
    llm_client = _build(
        LlmClientConfig,
        _get(raw, "generation"),
        overrides={
            "model": generation_model,
            "host": ollama_host,
            "num_ctx": _step_num_ctx("generation.num_ctx"),
        },
    )

    context_builder = _build(ContextBuilderConfig, _get(raw, "generation.context_builder"))

    # ── RAGAS evaluation (run_evaluation.py) ────────────────────────────
    ragas_evaluator = _build(
        RagasEvaluatorConfig,
        _get(raw, "evaluation"),
        overrides={
            "judge_llm_model": ragas_judge_model,
            "judge_llm_host": ollama_host,
            "judge_num_ctx": _step_num_ctx("evaluation.judge_num_ctx"),
            "embedding_model": embedding_model,
            "embedding_host": ollama_host,
            "embedding_num_ctx": _step_num_ctx("evaluation.embedding_num_ctx"),
        },
    )

    webapp = WebappSettings(
        host=_get(raw, "webapp.host", WebappSettings.host),
        port=_get(raw, "webapp.port", WebappSettings.port),
        debug=_get(raw, "webapp.debug", WebappSettings.debug),
    )

    return Settings(
        paths=paths,
        pipeline=pipeline,
        embed_generator=embed_generator,
        llm_client=llm_client,
        reranker=reranker,
        rerank_runtime=rerank_runtime,
        context_builder=context_builder,
        hybrid_retriever=hybrid_retriever,
        chroma_ingestion=chroma_ingestion,
        bm25_index=bm25_index,
        ragas_evaluator=ragas_evaluator,
        webapp=webapp,
        runtime=runtime,
        n_results=_get(raw, "generation.n_results", 5),
        evaluation_questions_path=_get(
            raw, "evaluation.questions_path",
            "./rag_pipeline/evaluation/sample_eval_questions.json",
        ),
        evaluation_save_results_path=_get(raw, "evaluation.save_results_path", None),
    )


if __name__ == "__main__":
    # Quick sanity check: load settings.yaml and print a summary.
    s = load_settings()
    print(f"✅ settings.yaml loaded successfully ({_DEFAULT_SETTINGS_PATH})\n")
    print(f"Input dir         : {s.pipeline.input_dir}")
    print(f"Chunks dir        : {s.pipeline.output_dir}")
    print(f"Embedding model   : {s.embed_generator.model_name}")
    print(f"Generation model  : {s.llm_client.model} @ {s.llm_client.host}")
    print(f"Reranker model    : {s.reranker.model_name} (enabled by default: {s.rerank_runtime.enabled})")
    print(f"Chroma dir        : {s.paths.chroma_dir}")
    print(f"BM25 index        : {s.paths.bm25_index_path}")
    print(f"Webapp            : http://{s.webapp.host}:{s.webapp.port}")
    print(
        f"Sequential loading: {s.runtime.sequential_model_loading} "
        f"(unload_after_step={s.runtime.unload_after_step}, "
        f"verify_unload={s.runtime.verify_unload})"
    )
