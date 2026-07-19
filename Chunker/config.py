"""
config.py — Single source of truth for every tunable parameter.
Edit here; nothing else needs to change.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessorConfig:
    # Image alt-text: "keep" | "drop" | "separate"
    #   keep       → inline with surrounding text
    #   drop       → remove entirely (pure text RAG)
    #   separate   → becomes its own chunk with type=image_description
    image_alt_strategy: str = "separate"

    # Remove table-of-contents tables (detected by "....." patterns)
    drop_toc_tables: bool = True

    # Strip HTML superscripts (<sup>…</sup>) — footnote markers
    strip_superscripts: bool = True

    # Normalise adjacent tables (insert blank line between them)
    fix_adjacent_tables: bool = True

    # Remove decorative-only lines (e.g. "Continues next page...")
    strip_decorative_lines: bool = True


@dataclass
class StructureSplitterConfig:
    # Which header levels to use as section boundaries
    # 1 = #, 2 = ##, 3 = ###
    header_levels: list = field(default_factory=lambda: [1, 2, 3])

    # Tables are ALWAYS kept atomic (never split across chunks)
    tables_atomic: bool = True

    # Minimum characters a structural section must have to be kept
    # (avoids empty header-only sections)
    min_section_chars: int = 20


@dataclass
class SemanticChunkerConfig:
    # Embedding model — an Ollama model tag (not a Hugging Face repo id).
    # Served via a local Ollama instance rather than loaded in-process, so
    # switching models is just `ollama pull <tag>` + updating this string.
    # "qwen3-embedding:8b" tops the MTEB multilingual leaderboard and
    # covers English + Spanish (100+ languages) well. Lighter alternatives
    # if VRAM is tight: "qwen3-embedding:4b" or "qwen3-embedding:0.6b".
    #
    # Make sure the model is pulled before running the pipeline:
    #   ollama pull qwen3-embedding:8b
    embedding_model: str = "qwen3-embedding:8b"

    # Local Ollama service address — matches LlmClientConfig.host so both
    # the embedding model and the generation model are served from the
    # same running instance.
    embedding_host: str = "http://localhost:11434"

    # Hugging Face repo id used ONLY to fetch a matching tokenizer for
    # accurate token counts (see splitters/token_counter.py) — a small
    # (~1-5MB) file, not the full model weights. Ollama model tags aren't
    # valid HF repo ids, so this is kept separate from embedding_model.
    # If token counting fails to resolve this, it falls back to a
    # word-count heuristic automatically — the pipeline never hard-fails
    # over this.
    tokenizer_hf_id: str = "Qwen/Qwen3-Embedding-8B"


    # Breakpoint strategy: "percentile" | "standard_deviation" | "interquartile"
    breakpoint_threshold_type: str = "percentile"

    # Threshold value for the chosen strategy
    # percentile: 0-100  (95 = only split on very large semantic jumps)
    breakpoint_threshold_amount: float = 85.0

    # Hard floor/ceiling on chunk sizes — LEGACY character-based limits.
    # Only used when use_token_based_sizing=False below. Kept for backward
    # compatibility; token-based sizing (below) is the recommended default
    # because character count is a poor proxy for what the embedding model
    # actually truncates at.
    min_chunk_chars: int = 80
    max_chunk_chars: int = 2000

    # ── Token-aware sizing ───────────────────────────────────────────────
    # qwen3-embedding models support a much longer context (32K tokens)
    # than the previous intfloat/multilingual-e5-large (hard 512-token
    # limit), so silent truncation is far less of a risk now. Token-based
    # sizing is still the default because character count remains an
    # unreliable proxy for token count across languages and content types
    # (tables especially). See splitters/token_counter.py for the full
    # rationale.
    use_token_based_sizing: bool = True

    # Floor: chunks below this are merged upward (same intent as
    # min_chunk_chars, just measured in tokens).
    min_chunk_tokens: int = 15

    # Ceiling: chunks above this are hard-split. Kept well under the 32K
    # context window — a chunk this large is no longer semantically
    # coherent as a single retrieval unit, regardless of how much context
    # the model can technically accept. Leaves headroom for an optional
    # LLM-written contextual header (see contextual_enricher.py) prepended
    # AFTER chunking, which adds tokens on top of this budget.
    max_chunk_tokens: int = 1024


    # Try to load the real HuggingFace tokenizer for embedding_model for
    # exact counts (small one-time download, cached after). If False, or
    # if the download fails/is unavailable, falls back to a word-count
    # heuristic automatically — the pipeline never hard-fails over this.
    attempt_real_tokenizer: bool = True

    # When a structural section is below the minimum size, skip semantic
    # splitting and keep it as-is
    skip_semantic_if_small: bool = True


@dataclass
class LlmChunkRefinerConfig:
    """
    Optional stage that runs immediately after SemanticChunker, before any
    other enrichment. SemanticChunker splits on embedding-similarity
    breakpoints — a purely statistical signal that has no notion of
    whether a chunk actually reads as a complete, self-contained thought.
    In practice this means it sometimes cuts a passage right before the
    sentence that would have made it independently answerable (e.g. the
    conclusion of a procedure ends up alone in the NEXT chunk, orphaned
    from the steps that gave it meaning).

    This stage sends each run of adjacent chunks from the same structural
    section to a local Ollama model and asks one question per group: "if
    a user asked something only this passage could answer, does each
    chunk alone contain everything needed, or does it feel like it's
    missing something from just before/after it?" The model returns
    which chunks should be merged — nothing else.

    Deliberately narrow scope: the LLM only decides MERGE BOUNDARIES. It
    never rewrites, summarizes, or paraphrases chunk text — merged chunks
    are the exact original content concatenated verbatim. This matters
    because this pipeline embeds technical manuals where an LLM
    "helpfully" rephrasing a table value or a procedure step would
    silently corrupt the source of truth being retrieved. Splitting
    oversized results back down is still handled by the existing
    deterministic hard-split safety net (see semantic_chunker.py), not
    by the LLM.

    Opt-in and off by default: one extra LLM call per section (or per
    batch of max_chunks_per_call chunks, for large sections), so this
    adds real wall-clock time — expect it to roughly double chunking time
    on CPU-only local hardware. Falls back to leaving chunks unmerged
    (not to crashing the pipeline) on any failure, same philosophy as
    every other LLM-optional stage in this codebase.
    """
    enabled: bool = False

    # ── Ollama (matches ContextualEnricherConfig's LLM-mode settings) ──────
    llm_model: str = "llama3.2"
    llm_host: str = "http://localhost:11434"
    llm_timeout_seconds: int = 30

    # Sections with fewer chunks than this skip the LLM call entirely —
    # there's nothing to merge with only 0-1 chunks.
    min_chunks_to_consider: int = 2

    # Cap on how many chunks go into a single LLM call. Large sections are
    # processed in consecutive batches of this size rather than one huge
    # prompt — keeps the call fast and within a local model's comfortable
    # context window. Merge decisions only happen WITHIN a batch, never
    # across a batch boundary (a rare miss on a boundary-straddling split,
    # traded for bounded latency and prompt size).
    max_chunks_per_call: int = 12

    # If the LLM call fails or returns unparseable output, keep the
    # original chunk boundaries from SemanticChunker rather than halting
    # the pipeline.
    fallback_on_error: bool = True

    # Print a one-line warning to stdout on fallback.
    warn_on_fallback: bool = True


@dataclass
class MetadataConfig:
    # Attach source file path to every chunk
    include_source_file: bool = True

    # Attach the full header breadcrumb ("# Title > ## Section > ### Sub")
    include_breadcrumb: bool = True

    # Attach chunk type: "text" | "table" | "image_description"
    include_chunk_type: bool = True

    # If a matching _metadata.json exists next to the .md, load page info
    include_page_metadata: bool = True

    # Attach detected language ("en" | "es" | "unknown")
    include_language: bool = True

    # Attach sequential chunk index within the document
    include_chunk_index: bool = True


@dataclass
class TableNormalizerConfig:
    # Master switch — if False, table chunks keep normalized_text == content
    # (raw HTML gets embedded as-is). Useful for A/B comparing retrieval
    # quality with/without normalization.
    enabled: bool = True

    # If a table fails to parse (malformed HTML), fall back to the raw
    # HTML as normalized_text rather than halting the whole pipeline run.
    fallback_to_raw_on_error: bool = True

    # Print a one-line warning to stdout on fallback.
    warn_on_fallback: bool = True


@dataclass
class ContextualEnricherConfig:
    """
    Implements "Contextual Retrieval" (situating each chunk within its
    document before embedding, so an isolated fragment like "CMI Report: -"
    doesn't lose the fact that it belongs to "SERVICE: Great circle >
    System reports"). This is the single highest-leverage improvement for
    fragment-level retrieval quality — proven to substantially cut
    retrieval failures in Anthropic's own published research.

    Two modes:
      "deterministic" — builds the context from data already on the chunk
        (breadcrumb + source filename). Zero cost, zero setup, works
        immediately. Captures most of the value: it tells the embedding
        model WHERE a chunk is from, even though it can't paraphrase WHY
        the chunk matters the way an LLM reading the whole document can.
      "llm"           — sends the document + the chunk to a local Ollama
        model, which writes 1-2 sentences situating the chunk. Higher
        quality, but requires Ollama running locally and is much slower
        (one LLM call per chunk — for a document with 50 chunks, expect
        this to take real wall-clock minutes on CPU-only local hardware).
      "off"           — contextualized_text becomes an exact copy of
        normalized_text, no prefix added.
    """
    mode: str = "deterministic"  # "deterministic" | "llm" | "off"

    # ── LLM mode settings (Ollama) ──────────────────────────────────────
    llm_model: str = "llama3.2"
    llm_host: str = "http://localhost:11434"
    llm_timeout_seconds: int = 30

    # How much of the document to show the LLM as surrounding context.
    # Truncated from the start of the document — enough for the model to
    # infer overall topic/structure without blowing a small local model's
    # context window on every single chunk call.
    max_document_chars_for_llm: int = 3000

    # If the LLM call fails (Ollama not running, timeout, bad response),
    # fall back to deterministic context rather than leaving the chunk
    # unenriched or halting the pipeline.
    fallback_to_deterministic_on_error: bool = True

    # Print a one-line warning to stdout on LLM fallback.
    warn_on_fallback: bool = True

    # ── Deterministic mode settings ─────────────────────────────────────
    include_breadcrumb: bool = True
    include_source_filename: bool = True


@dataclass
class EmbedTextConfig:
    """
    The embedding model's instruction-prefix contract — see
    enrichers/embed_text_builder.py for the full rationale. This is the
    ONLY place these two prefix strings should be defined; both the
    pipeline's storage side (EmbedTextBuilder) and the future retriever's
    query side (prepare_query_text) read from here, so they can never
    drift apart.

    qwen3-embedding models use a different contract than the previous
    intfloat/multilingual-e5-large: no prefix on the stored-passage side,
    and an instruction-style prefix on the query side only (Qwen's model
    card reports skipping the query instruction costs ~1-5% retrieval
    accuracy — passage_prefix has no equivalent instruction, so it stays
    empty).
    """
    enabled: bool = True
    passage_prefix: str = ""
    query_prefix: str = (
        "Instruct: Given a search query, retrieve relevant passages that "
        "answer the query\nQuery: "
    )


@dataclass
class EmbedGeneratorConfig:
    """
    Config for embedding/embed_generator.py — the module that actually
    calls the embedding model to turn embed_text strings into vectors.

    model_name is intentionally a SEPARATE setting from
    SemanticChunkerConfig.embedding_model (even though they'll normally
    be set to the same value) because embedding generation is a distinct
    downstream step that may run independently of chunking — e.g.
    re-embedding all chunks after switching models, without reprocessing
    every PDF from scratch.
    """
    model_name: str = "qwen3-embedding:8b"

    # Local Ollama service address serving model_name. Must match a
    # model already pulled with `ollama pull qwen3-embedding:8b`.
    host: str = "http://localhost:11434"

    # Batch size for embedding calls. Ollama's /api/embed accepts a list
    # of inputs per request; keep this modest (8-16) on a shared/low-VRAM
    # GPU since qwen3-embedding:8b is much heavier per-request than the
    # previous ONNX-based model — raise it if you have headroom.
    batch_size: int = 16


@dataclass
class ChromaIngestionConfig:
    """
    Config for storage/chroma_ingestion.py — writes embedded chunks into
    a local, persistent ChromaDB collection (no server required).
    """
    # Local directory where ChromaDB persists its data. Created on first
    # ingestion; safe to point multiple pipeline runs at the same
    # directory — .upsert() with stable chunk_ids means re-ingesting
    # unchanged chunks updates them in place rather than duplicating.
    persist_directory: str = "./chroma_db"

    # Collection name — a single ChromaDB PersistentClient can host
    # multiple named collections if you want to separate document sets.
    collection_name: str = "rag_docs"

    # Batch size for .upsert() calls. ChromaDB can handle large single
    # calls, but batching keeps memory bounded for very large corpora.
    batch_size: int = 100


@dataclass
class BM25IndexConfig:
    """
    Config for storage/bm25_index.py — a keyword/exact-term search index
    that complements dense vector search. Essential for technical
    documents full of domain-specific abbreviations (CMI, MAC, GCI,
    LSSCONF...) that an embedding model has never seen in training and
    can only treat as near-meaningless tokens. BM25 finds exact matches
    regardless of what the embedding model "understands."
    """
    # Where the built index (+ parallel chunk_id list) is persisted via
    # pickle. Rebuilding is fast — seconds even for hundreds of
    # documents — so it's fine to rebuild on any pipeline/config change
    # rather than trying to incrementally update.
    persist_path: str = "./bm25_index.pkl"

    # BM25Okapi hyperparameters (rank_bm25 defaults shown) — standard
    # values from the original Okapi BM25 paper; only worth tuning if
    # you have a labeled eval set to tune against (see roadmap step 9).
    k1: float = 1.5
    b: float = 0.75


@dataclass
class HybridRetrieverConfig:
    """
    Config for retrieval/hybrid_retriever.py — combines dense (ChromaDB)
    and sparse (BM25) search via Reciprocal Rank Fusion (RRF).
    """
    # How many candidates each individual retriever contributes before
    # fusion. Wider than final_n_results so RRF has real material to
    # combine — a document ranked #15 by one retriever but #1 by the
    # other can still surface near the top after fusion.
    dense_n_results: int = 20
    sparse_n_results: int = 20

    # Final number of results returned after fusion.
    final_n_results: int = 10

    # RRF's smoothing constant: score(doc) = Σ 1/(k + rank) across the
    # ranked lists it appears in. k=60 is the standard value from the
    # original RRF paper — it prevents a #1 rank in one list from
    # completely dominating regardless of the other list's opinion.
    # Only worth changing against a labeled eval set (roadmap step 9),
    # not by intuition.
    rrf_k: int = 60

    # Pre-filter the dense (ChromaDB) search to the query's detected
    # language via a metadata `where` clause. Improves precision when
    # the query's language is unambiguous; automatically skipped when
    # detect_language() returns "unknown" (filtering by "unknown" would
    # incorrectly exclude everything).
    language_filter: bool = True


@dataclass
class RerankerConfig:
    """
    Config for retrieval/reranker.py — a cross-encoder that re-scores
    the hybrid retriever's top candidates for a final precision boost.

    Why this exists alongside the bi-encoder (E5) already used for
    dense search: a bi-encoder embeds the query and each document
    SEPARATELY, then compares vectors — fast (documents can be
    pre-embedded once and reused for every future query) but
    approximate. A cross-encoder feeds the query AND a candidate
    document THROUGH THE SAME MODEL TOGETHER, with full attention
    between them — much more accurate at judging "does this actually
    answer that," but too slow to run against an entire corpus (must
    run once per candidate, every query). The practical pattern —
    used here — is bi-encoder for fast wide recall (hybrid retrieval's
    top ~10-20), cross-encoder for a final precision pass on just
    those few candidates.
    """
    # Qwen/Qwen3-Reranker-4B: a 4B-parameter cross-encoder, state-of-the-art
    # on multilingual retrieval benchmarks (MTEB-R/CMTEB-R/MMTEB-R),
    # covers English + Spanish (100+ languages) well above the previous
    # cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (~120MB). Much heavier:
    # needs a GPU with a few GB of free VRAM for reasonable latency, and
    # requires transformers>=4.51.0 (see requirements.txt). Lighter
    # alternatives from the same family if VRAM is tight:
    # "Qwen/Qwen3-Reranker-0.6B".
    model_name: str = "Qwen/Qwen3-Reranker-4B"

    # Qwen3-Reranker is "instruction aware" — an optional task
    # description that's prepended to every (query, document) pair.
    # Qwen's own model card reports ~1-5% accuracy gain from using one
    # over leaving it unset. Set to None to skip it (matches the model's
    # own un-prompted default behavior).
    task_instruction: Optional[str] = (
        "Given a search query, determine whether this document contains "
        "information that answers the query."
    )

    # Qwen3-Reranker supports up to a 32K-token context, far beyond the
    # previous BERT-family cross-encoder's hard 512-token limit — but a
    # candidate chunk this large is already too big to be a useful
    # retrieval unit (see SemanticChunkerConfig.max_chunk_tokens), so
    # this is kept far below the model's real ceiling rather than set to
    # match it. Raise only if you have a specific reason to send much
    # longer candidates through reranking.
    max_length: int = 4096

    # How many of the reranked candidates to keep. Typically smaller
    # than HybridRetrieverConfig.final_n_results — the point of
    # reranking is to distill a wider recall set down to the few best.
    final_n_results: int = 5


@dataclass
class ContextBuilderConfig:
    """
    Config for generation/context_builder.py — formats retrieved chunks
    into the context block an LLM actually reads.

    Note on table content: normalized_text (what chunks carry by the
    time they reach retrieval) is ALREADY clean prose for table chunks
    — table_normalizer.py (roadmap step 1) converts HTML to natural
    language before anything is embedded or stored. This module's HTML
    stripping is a defensive fallback for the unusual case where table
    normalization was disabled, not the primary mechanism.
    """
    include_type_label: bool = True
    include_source: bool = True
    include_breadcrumb: bool = True

    # Safety cap on how many chunks go into the context block,
    # independent of how many the retriever/reranker returned — a
    # local LLM's context window is finite, and this is a second line
    # of defense against accidentally dumping too much in.
    max_chunks: int = 5

    # Truncate any single chunk's text beyond this length before
    # including it, to bound worst-case prompt size from one outlier
    # chunk. None disables truncation.
    max_chars_per_chunk: Optional[int] = 1500


@dataclass
class LlmClientConfig:
    """
    Config for generation/llm_client.py — wraps Ollama for fully
    offline LLM answer generation (no API key, no external network
    call beyond the local Ollama service).
    """
    model: str = "llama3.2"
    host: str = "http://localhost:11434"

    # Low temperature: this is factual RAG question-answering, not
    # creative writing — we want the model to stick close to the
    # provided context rather than improvise.
    temperature: float = 0.1

    timeout_seconds: int = 60


@dataclass
class RagasEvaluatorConfig:
    """
    Config for evaluation/ragas_evaluator.py — measures whether the
    whole pipeline (retrieval + generation) is actually working, using
    RAGAS's four core metrics: context_precision, context_recall,
    faithfulness, answer_relevancy.

    Critical departure from RAGAS's defaults: RAGAS assumes OpenAI for
    both its judge-LLM and its embeddings unless told otherwise. This
    entire project has been built fully offline, so this config wires
    RAGAS to use Ollama for BOTH the judge LLM and its embedding-based
    metrics (matching generation/llm_client.py and
    EmbedGeneratorConfig respectively) — never OpenAI.
    """
    # Judge LLM — used by RAGAS to assess faithfulness, relevancy, etc.
    # Can be the same or a different (e.g. larger/more careful) model
    # than the one used for actual answer generation in llm_client.py.
    judge_llm_model: str = "llama3.2"
    judge_llm_host: str = "http://localhost:11434"

    # Embedding model for RAGAS's embedding-based metrics (e.g. answer
    # relevancy compares embedded synthetic questions against the
    # embedded original question). Matches EmbedGeneratorConfig's
    # default so the whole pipeline stays on one consistent model.
    embedding_model: str = "qwen3-embedding:8b"
    embedding_host: str = "http://localhost:11434"

    # Which of RAGAS's metrics to compute. All four are the core set
    # the original evaluation plan calls for; trimming this list saves
    # judge-LLM calls (each metric costs its own LLM round-trips) at
    # the cost of a less complete picture.
    metrics: list = field(default_factory=lambda: [
        "context_precision", "context_recall", "faithfulness", "answer_relevancy",
    ])

    # Thresholds used only for the printed summary's warning flags —
    # not enforced anywhere, purely diagnostic signals matching the
    # original evaluation plan's guidance.
    context_recall_warning_threshold: float = 0.8
    faithfulness_warning_threshold: float = 0.85

    # Concurrency/timeout for RAGAS's judge-LLM calls. RAGAS's own default
    # (max_workers=16) assumes a scalable API backend (OpenAI, etc.) — a
    # single local Ollama process serializes concurrent requests instead,
    # so 16-wide concurrency mostly produces TimeoutErrors. Keep this low
    # and the timeout generous for local-only setups.
    judge_max_workers: int = 2
    judge_timeout_seconds: int = 300


@dataclass
class PipelineConfig:
    # Root directories
    input_dir: Path = Path("./outputs")       # where Chandra OCR .md files live
    output_dir: Path = Path("./chunks")       # where chunk JSONs will be saved

    # Glob pattern to find markdown files inside input_dir (recursive)
    md_glob: str = "**/*.md"

    # Sub-configs
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    structure: StructureSplitterConfig = field(default_factory=StructureSplitterConfig)
    semantic: SemanticChunkerConfig = field(default_factory=SemanticChunkerConfig)
    llm_chunk_refiner: LlmChunkRefinerConfig = field(default_factory=LlmChunkRefinerConfig)
    table_normalizer: TableNormalizerConfig = field(default_factory=TableNormalizerConfig)
    contextual: ContextualEnricherConfig = field(default_factory=ContextualEnricherConfig)
    embed_text: EmbedTextConfig = field(default_factory=EmbedTextConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    # Pretty-print progress to stdout
    verbose: bool = True
