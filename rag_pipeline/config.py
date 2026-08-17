"""
config.py — Single source of truth for every tunable parameter.
Edit here; nothing else needs to change.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RuntimeConfig:
    """
    Controls model residency on the GPU across the WHOLE project — see
    rag_pipeline/model_manager.py for the full mechanism. This is the
    "only one model loaded at a time, load it → use it → stop it →
    load the next one" switch.

    Every module that calls an Ollama model (embedding, generation,
    chunk refiner, contextual enricher, ollama-backend reranker, RAGAS
    judge) calls model_manager.ensure_active() with this config right
    before it does, and the various run_*.py / webapp entry points call
    release()/release_all() at the natural end of a step or request.
    """
    # Master switch. True (default) = never let two different models
    # sit resident on the same Ollama host at once — switching models
    # always unloads whatever else was loaded first. False = fall back
    # to plain Ollama caching behaviour (its own keep_alive heuristics
    # decide what stays resident); useful on a multi-GPU box or a box
    # with enough VRAM to comfortably hold everything at once.
    sequential_model_loading: bool = True

    # When a step/request finishes, explicitly unload the model it was
    # using rather than leaving it resident until Ollama's keep_alive
    # window naturally expires. Also governs whether the in-process
    # HuggingFace reranker frees its GPU memory after each use. Turn
    # this off if you'd rather let Ollama's own keep_alive/back-to-back
    # calls to the SAME model avoid a reload — e.g. a chat webapp that
    # expects the next question to reuse the generation model again
    # within seconds.
    unload_after_step: bool = True

    # After asking Ollama to unload a model, poll /api/ps until it's
    # actually gone before letting the NEXT model's call proceed — this
    # is what guarantees the two never briefly overlap in VRAM. False
    # = fire-and-forget the unload request and move on immediately
    # (faster, but the next model's load could race the previous one's
    # eviction on a slow unload).
    verify_unload: bool = True

    # How long to wait for /api/ps to confirm a model is gone before
    # giving up and proceeding anyway (never hangs the pipeline
    # indefinitely over a slow/stuck unload).
    unload_timeout_seconds: float = 30.0

    # How often to re-check /api/ps while waiting for an unload to be
    # confirmed.
    unload_poll_interval_seconds: float = 0.5


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

    # Context window (tokens) requested from Ollama for the embedding
    # model. None = use Ollama's own default (commonly 2048), which is
    # often far below what the model itself supports and can cause
    # "context length exceeded" errors on long sections/sentences during
    # semantic splitting. Raise this (e.g. 8192+) if you hit that error.
    num_ctx: Optional[int] = None

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
    llm_model: str = "Qwen-AgentWorld-35B-A3B:latest"
    llm_host: str = "http://localhost:11434"
    llm_timeout_seconds: int = 30

    # Context window (tokens) requested from Ollama for this call. None =
    # Ollama's own default. A batch of max_chunks_per_call chunks can add
    # up to a fairly long prompt, so raise this if calls start failing
    # with a context-length error.
    num_ctx: Optional[int] = None

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
    llm_model: str = "Qwen-AgentWorld-35B-A3B:latest"
    llm_host: str = "http://localhost:11434"
    llm_timeout_seconds: int = 30

    # Context window (tokens) requested from Ollama for this call. None =
    # Ollama's own default. Raise if calls fail with a context-length
    # error (the whole document excerpt + chunk text is sent per call).
    num_ctx: Optional[int] = None

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

    # Context window (tokens) requested from Ollama for the embedding
    # model. None = Ollama's own default (commonly 2048), often well
    # below what the model supports. Raise if embed() calls fail with a
    # context-length error.
    num_ctx: Optional[int] = None

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
    # Raised from 100 to 200 for a 5000-page corpus (likely tens of
    # thousands of chunks) — fewer round trips without meaningfully
    # increasing peak memory.
    batch_size: int = 200


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
    # Widened from 20 to 30: a 5000-page corpus has a much larger and
    # more topically diverse candidate space than a small test corpus,
    # so each individual retriever needs to look further down its own
    # ranked list for RRF to have enough material to correct the other
    # retriever's mistakes. With the reranker's final cross-encoder pass
    # (cheap at 0.6B) cleaning up the fused list afterward, the recall
    # gain from a wider pool costs little in final precision.
    dense_n_results: int = 30
    sparse_n_results: int = 30

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
    # Qwen/Qwen3-Reranker-0.6B: the smallest model in the Qwen3-Reranker
    # family — chosen here over the 4B/8B siblings because at 5000+ pages
    # of source documents, the candidate pool being reranked per query
    # (dense_n_results + sparse_n_results worth of chunks, see
    # HybridRetrieverConfig) is what dominates latency, not per-pair
    # model size. 0.6B keeps that pass fast enough to comfortably widen
    # the candidate pool for better recall, at a small (~1-3 point,
    # per Qwen's own MTEB-R numbers) accuracy cost vs the 4B/8B variants.
    # Loaded via sentence-transformers' CrossEncoder directly from
    # Hugging Face — NOT via Ollama. Ollama has no cross-encoder rerank
    # API; the only way to run Qwen3-Reranker locally today is through
    # transformers/sentence-transformers, so this pulls the HF weights on
    # first use (cached after) even though Ollama serves the embedding
    # and generation models elsewhere in this pipeline. Requires
    # `pip install sentence-transformers` plus `transformers>=4.51.0`.
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"

    # Which runtime loads and scores the reranker model:
    #   "huggingface" (default) — sentence_transformers.CrossEncoder,
    #     loads model_name (a HF repo id) directly from Hugging Face.
    #   "ollama" — scores candidates through a model already pulled into
    #     Ollama, using the query/document-pair-as-prompt technique
    #     below. No extra downloads or GPU memory beyond what Ollama
    #     already uses elsewhere in this pipeline, at the cost of one
    #     Ollama call per (query, candidate) pair instead of a single
    #     batched local forward pass — noticeably slower for a large
    #     candidate_pool. Good option if you don't want
    #     sentence-transformers/transformers installed at all, or you've
    #     already got a capable instruct model in Ollama and don't want
    #     to also pull the dedicated HF cross-encoder weights.
    backend: str = "huggingface"

    # ---- "ollama" backend settings (ignored by "huggingface") ----
    # Ollama has no native cross-encoder/rerank API, so this uses the
    # same technique Qwen3-Reranker's own "generative" mode uses: ask an
    # instruct model a strict yes/no question per (query, document) pair
    # and turn the yes-vs-no token probability into a relevance score.
    # Any decent instruct model can act as a reranker this way — it
    # doesn't have to be a model specifically trained for reranking,
    # though a dedicated reranker model will generally score better.
    # This is a plain Ollama library tag (e.g. "qwen3:4b"), NOT a
    # Hugging Face repo id — the two are different formats and are not
    # interchangeable (see the "huggingface" backend above).
    ollama_model: str = "qwen3:4b"
    ollama_host: str = "http://localhost:11434"
    ollama_num_ctx: Optional[int] = None
    # How many alternative tokens to ask Ollama for at the answer
    # position. Needs to be large enough that both "yes" and "no" (or
    # their tokenizer variants) reliably show up among the alternatives
    # even when the model is confident in one of them — 20 is generous
    # headroom for that on most tokenizers.
    ollama_top_logprobs: int = 20

    # Device to load the cross-encoder on. None lets sentence-transformers
    # auto-pick (cuda if available, else cpu). Set explicitly (e.g.
    # "cuda:0") if you're running multiple models across GPUs.
    device: Optional[str] = None

    # Passed straight through to AutoModel.from_pretrained via
    # sentence-transformers' CrossEncoder(model_kwargs=...). fp16 halves
    # VRAM and speeds up inference on GPU with a negligible accuracy
    # difference for a 0.6B model — set to {} for full fp32 precision if
    # running CPU-only (fp16 gives little/no speedup on CPU and some CPU
    # kernels don't support it).
    model_kwargs: dict = field(default_factory=lambda: {"torch_dtype": "float16"})

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

    # In-process (huggingface backend) reranker lifecycle: when True
    # (default), the CrossEncoder is loaded fresh on every rerank()
    # call and its GPU memory is explicitly freed immediately
    # afterward, matching the "only one model on the GPU at a time"
    # policy in RuntimeConfig for this project's one in-process (i.e.
    # non-Ollama) model. Set False to keep it resident/cached across
    # calls instead — faster for back-to-back queries, at the cost of
    # permanent VRAM usage for the life of the process. Ignored by the
    # "ollama" backend, which is governed by RuntimeConfig directly.
    huggingface_unload_after_use: bool = True

    # How many of the reranked candidates to keep. Typically smaller
    # than HybridRetrieverConfig.final_n_results — the point of
    # reranking is to distill a wider recall set down to the few best.
    # Bumped from the original 5 to 8: at 5000+ pages, more distinct
    # source sections plausibly contain partial answers, and reranking
    # with a 0.6B model is cheap enough that keeping a couple of extra
    # candidates costs little while giving ContextBuilderConfig.max_chunks
    # slightly more to work with.
    final_n_results: int = 8


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

    # Render the heading chain as NAMED levels in each context header
    # ("Chapter: Prompting > Section: Few-shot") instead of a bare
    # "Prompting > Few-shot". The explicit words matter: the model can
    # only report "chapter X, section Y" in its answer if those words
    # appear beside the titles in the context it was given.
    label_heading_levels: bool = True

    # Ask the model to append the document location (file, page, chapter,
    # section) to every bracketed citation, e.g.
    #   "... [1] (Binder2.pdf, page 12 - Chapter: Prompting)".
    # Turn off for terse answers that cite bare numbers only.
    cite_document_location: bool = True

    # --- Breadth controls for general questions --------------------------
    # A broad question ("what is prompt engineering?") retrieves the five
    # most similar paragraphs, and those are usually five NEIGHBOURING
    # paragraphs of one section — so the model answers as if that section
    # were the whole topic. This caps how many of the max_chunks slots any
    # single section may occupy, spending them on different parts of the
    # document instead. Never reduces the number of chunks sent: once every
    # section has hit the cap, skipped chunks backfill the remaining slots.
    # 0 or None disables the cap.
    max_chunks_per_section: Optional[int] = 2

    # Inject the heading tree above the numbered sources:
    #   "Chapter: Prompt Engineering — Sections: Zero-shot; Few-shot; ..."
    # "Few-shot is ONE OF SEVERAL strategies" is a fact about the document's
    # STRUCTURE, so it is written in no chunk and retrieval can never return
    # it. This is the only way the model can see it.
    #   "auto"   — only when the question looks like an overview question
    #   "always" — on every question
    #   "off"    — never
    document_map: str = "auto"

    # How many chapters the map may list, and how many sections per chapter,
    # so the map can never crowd out the actual passages.
    document_map_max_chapters: int = 3
    document_map_max_sections: int = 12

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
    model: str = "Qwen-AgentWorld-35B-A3B:latest"
    host: str = "http://localhost:11434"

    # Low temperature: this is factual RAG question-answering, not
    # creative writing — we want the model to stick close to the
    # provided context rather than improvise.
    temperature: float = 0.1

    timeout_seconds: int = 60

    # Context window (tokens) requested from Ollama for answer
    # generation. None = Ollama's own default. Raise if long context
    # blocks (see ContextBuilderConfig.max_chunks) cause a context-length
    # error.
    num_ctx: Optional[int] = None


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
    judge_llm_model: str = "Qwen-AgentWorld-35B-A3B:latest"
    judge_llm_host: str = "http://localhost:11434"

    # Context window (tokens) requested from Ollama for the judge LLM.
    # None = Ollama's own default.
    judge_num_ctx: Optional[int] = None

    # Embedding model for RAGAS's embedding-based metrics (e.g. answer
    # relevancy compares embedded synthetic questions against the
    # embedded original question). Matches EmbedGeneratorConfig's
    # default so the whole pipeline stays on one consistent model.
    embedding_model: str = "qwen3-embedding:8b"
    embedding_host: str = "http://localhost:11434"

    # Context window (tokens) requested from Ollama for RAGAS's
    # embedding-based metrics. None = Ollama's own default.
    embedding_num_ctx: Optional[int] = None

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
