# ============================================================
# FourthStep_query.py — Production RAG Pipeline
#
# ✅ Fully synced with ThirdStep_embed_and_store.py
#
# Features:
#   I.   Multilingual Query Processing & HyDE Expansion
#   II.  Hybrid Retrieval (Dense Qdrant + BM25) with RRF Fusion
#   III. Two-Stage Cross-Encoder Reranking
#   IV.  Parent-Child Context Assembly (2500-token ceiling)
#   V.   Breadcrumb Citations & Self-Correction Loop
#   VI.  VRAM Guard & Latency Monitoring
#
# Run:
#   pip install rank_bm25 langdetect sentence-transformers qdrant-client ollama tiktoken
#   python FourthStep_query.py -d ./vector_db
# ============================================================

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional

import ollama
import tiktoken
from langdetect import detect, LangDetectException
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── SYNC with Step 3: Qdrant instead of ChromaDB ──────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# ==========================================
# ⚙️  CONFIGURATION — must mirror Step 3
# ==========================================

# ── SYNC: identical to Step 3 ─────────────────────────────────────────────
import os as _os
MODEL_CACHE_ROOT = _os.environ.get(
    "HF_HOME",
    str(Path(__file__).parent / "model_cache")
)
MODEL_NAME         = "intfloat/multilingual-e5-large"
MODEL_NAME           = "intfloat/multilingual-e5-large"
EMBEDDING_DIM        = 1024          # must match Step 3

# ── SYNC: E5 query prefix — matches Step 3's QUERY_INSTRUCTION ────────────
QUERY_INSTRUCTION    = "query: "     # Step 3 uses "query: " for queries

OLLAMA_MODEL_NAME    = "llama3.2:3b"
HYDE_MODEL           = "llama3.2:3b"
RERANKER_MODEL       = "BAAI/bge-reranker-v2-m3"

# Retrieval knobs
INITIAL_N_RESULTS    = 50
BM25_TOP_K           = 50
RRF_K                = 60
RERANK_TOP_N         = 8
CONTEXT_TOKEN_LIMIT  = 2_500

# VRAM guard
VRAM_SAFE_THRESHOLD_GB = 4.0

# Latency targets (seconds)
RETRIEVAL_BUDGET     = 0.05
RERANK_BUDGET        = 0.15


# ==========================================
# 🛠️  UTILITY FUNCTIONS
# ==========================================

def find_snapshot_path(cache_root: str, model_name: str) -> Optional[Path]:
    """Locate the latest HuggingFace snapshot — identical logic to Step 3."""
    folder_name   = "models--" + model_name.replace("/", "--")
    model_path    = Path(cache_root) / folder_name
    if not model_path.exists():
        return None
    snapshots_dir = model_path / "snapshots"
    if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
        return None
    return sorted(snapshots_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[0]


def load_embedding_model(cache_root: str, model_name: str) -> SentenceTransformer:
    """
    Load embedding model offline — fully synced with Step 3's load_model():
      • Both TRANSFORMERS_OFFLINE and HF_HUB_OFFLINE are set (Step 3 sets both).
      • model_kwargs key is 'dtype', not 'torch_dtype' (Step 3 uses 'dtype').
      • Fallback to CPU float32 on GPU failure, same as Step 3.
    """
    # ── SYNC: Step 3 sets BOTH offline vars ───────────────────────────────
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"]       = "1"   # was missing in old Step 4

    snapshot = find_snapshot_path(cache_root, model_name)
    if snapshot is None:
        raise FileNotFoundError(f"Model '{model_name}' not found in cache: {cache_root}")

    print(f"🔄 Loading embedding model from: {snapshot}")
    device = "cuda" if _cuda_available() else "cpu"

    try:
        # ── SYNC: key is 'dtype', not 'torch_dtype' (matches Step 3) ──────
        model = SentenceTransformer(
            str(snapshot),
            device=device,
            model_kwargs={"dtype": "float16"} if device == "cuda" else {},
        )
        print(f"✅ Embedding model loaded on {device.upper()} (float16 active: {device=='cuda'})")
    except Exception as exc:
        print(f"⚠️  GPU load failed ({exc}). Falling back to CPU float32...")
        model = SentenceTransformer(str(snapshot), device="cpu")
        print("✅ Embedding model loaded on CPU")

    return model


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_free_vram_gb() -> float:
    """Return free VRAM in GB, or 0.0 if CUDA is unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def load_reranker() -> CrossEncoder:
    """Load cross-encoder reranker; falls back to CPU if VRAM is tight."""
    free_gb = get_free_vram_gb()
    device  = "cuda" if free_gb >= VRAM_SAFE_THRESHOLD_GB else "cpu"
    if device == "cpu" and free_gb > 0:
        print(f"⚠️  Only {free_gb:.1f} GB VRAM free — reranker will run on CPU")
    else:
        print(f"✅ {free_gb:.1f} GB VRAM free — reranker on {device.upper()}")
    return CrossEncoder(RERANKER_MODEL, device=device)


def count_tokens(text: str) -> int:
    """Approximate token count using cl100k_base tokenizer."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split()) * 4 // 3


# ==========================================
# I. MULTILINGUAL QUERY PROCESSING
# ==========================================

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def classify_intent(query: str) -> str:
    analytical_keywords = [
        "explain", "how does", "why", "compare", "difference",
        "چگونه", "چرا", "توضیح", "مقایسه", "تفاوت",
    ]
    return "analytical" if any(kw in query.lower() for kw in analytical_keywords) else "factoid"


def hyde_expand(query: str, ollama_model: str) -> str:
    """Hypothetical Document Embeddings: generate a fake expert answer to improve retrieval."""
    hyde_prompt = (
        "You are a senior technical expert. Write a short, dense, factual passage "
        "(2-3 sentences) that directly ANSWERS the following question as if it came "
        "from an authoritative technical manual. Do NOT say 'I think' or add caveats — "
        "just write the answer passage.\n\n"
        f"Question: {query}\n\nPassage:"
    )
    try:
        response = ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": hyde_prompt}],
        )
        hypothesis = response["message"]["content"].strip()
        print(f"🧠 HyDE hypothesis: {hypothesis[:100]}...")
        return hypothesis
    except Exception as e:
        print(f"⚠️  HyDE generation failed ({e}), using raw query")
        return query


def embed_query(model: SentenceTransformer, text: str) -> list[float]:
    """
    Embed a single query string.
    SYNC: prepends QUERY_INSTRUCTION = 'query: ' — matches Step 3's embed_texts(is_query=True).
    L2-normalised, same as Step 3.
    """
    prompted = QUERY_INSTRUCTION + text
    vec = model.encode(
        [prompted],
        normalize_embeddings=True,   # SYNC: Rule II — mandatory L2-norm
        convert_to_numpy=True,
    )
    return vec.tolist()[0]


# ==========================================
# II. HYBRID RETRIEVAL + RRF FUSION
# ==========================================

def dense_retrieval(
    client: QdrantClient,
    collection_name: str,
    query_embedding: list[float],
    n_results: int,
) -> list[dict]:
    """
    Dense (semantic) retrieval from Qdrant.
    SYNC: uses the same collection Step 3 wrote to.
    Supports both qdrant-client ≥1.7 (query_points) and older (search).
    """
    # qdrant-client ≥1.7 renamed .search() → .query_points()
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=n_results,
            with_payload=True,
        )
        hits = response.points
    else:
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=n_results,
            with_payload=True,
        )

    results = []
    for hit in hits:
        payload = hit.payload or {}
        text    = payload.get("text", "")
        results.append({
            "text":  text,
            "meta":  payload,
            "score": hit.score,
            "id":    str(hit.id),
        })
    return results


def bm25_retrieval(
    all_docs: list[str],
    all_metas: list[dict],
    query: str,
    top_k: int,
) -> list[dict]:
    """Sparse BM25 retrieval over the full corpus loaded into memory."""
    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25   = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"text": all_docs[i], "meta": all_metas[i], "score": float(scores[i]), "id": f"bm25_{i}"}
        for i in top_indices
    ]


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """Merge two ranked lists using RRF. Score = Σ 1 / (k + rank_i)."""
    scores:  dict[str, float] = {}
    doc_map: dict[str, dict]  = {}

    for rank, item in enumerate(dense_results):
        key = item["text"][:120]
        scores[key]  = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = item

    for rank, item in enumerate(bm25_results):
        key = item["text"][:120]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in doc_map:
            doc_map[key] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**doc_map[key], "rrf_score": score} for key, score in fused]


# ==========================================
# III. CROSS-ENCODER RERANKING
# ==========================================

def rerank(
    reranker: CrossEncoder,
    query: str,
    candidates: list[dict],
    top_n: int,
) -> list[dict]:
    """Cross-encoder joint scoring of (query, passage) pairs."""
    pairs     = [(query, c["text"]) for c in candidates]
    ce_scores = reranker.predict(pairs)
    for i, item in enumerate(candidates):
        item["ce_score"] = float(ce_scores[i])
    reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
    return reranked[:top_n]


# ==========================================
# IV. CONTEXT ASSEMBLY
# ==========================================

def assemble_context(
    chunks: list[dict],
    token_limit: int = CONTEXT_TOKEN_LIMIT,
) -> tuple[str, list[str]]:
    """
    Build the context string respecting the 2500-token ceiling.
    SYNC: citation labels are built from Step 3's actual metadata keys:
      breadcrumb, source_file, chunk_type, page_range.
    """
    context_parts: list[str] = []
    citations:     list[str] = []
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        meta = chunk.get("meta") or {}

        # ── SYNC: use Step 3's actual payload keys ─────────────────────────
        breadcrumb  = meta.get("breadcrumb", "")
        source_file = meta.get("source_file", "")
        chunk_type  = meta.get("chunk_type", "text")
        page_range  = meta.get("page_range", [])

        # Build citation label from available Step 3 metadata
        if breadcrumb:
            cite_label = f"[{breadcrumb}]"
        elif source_file:
            page_str   = f" p.{page_range}" if page_range else ""
            cite_label = f"[{source_file}{page_str}]"
        else:
            cite_label = f"[Chunk {i+1}]"

        text   = chunk.get("text", "")
        tokens = count_tokens(text)

        if total_tokens + tokens > token_limit:
            print(f"ℹ️  Context ceiling reached at chunk {i+1} ({total_tokens} tokens). Stopping.")
            break

        context_parts.append(f"{cite_label}\n{text}")
        citations.append(cite_label)
        total_tokens += tokens

    print(f"📦 Context assembled: {total_tokens} tokens across {len(context_parts)} chunks")
    return "\n\n---\n\n".join(context_parts), citations


# ==========================================
# V. ANSWER GENERATION + CITATIONS + SELF-CORRECTION
# ==========================================

SYSTEM_PROMPT_TEMPLATE = """\
You are a senior technical expert. Answer the user's question **strictly and exclusively** \
using the provided context passages.

STRICT RULES:
1. **Faithfulness:** Only use information from the Context below. Do NOT add outside knowledge.
2. **Citations:** Every factual claim must be followed by its citation label in brackets.
3. **Completeness:** Cover all relevant details found in the context.
4. **Structure:** Use headings, bullet points, and numbered steps where appropriate.
5. **Language:** Respond in {lang} (the same language the user asked in).
6. **Honesty:** If the context does not contain enough information, say so clearly.

Context:
{context}
"""


def build_prompt(question: str, context: str, query_lang: str) -> str:
    lang_map = {
        "fa": "Persian (Farsi)", "en": "English", "es": "Spanish",
        "de": "German", "fr": "French", "ar": "Arabic",
    }
    lang   = lang_map.get(query_lang, "the same language as the question")
    system = SYSTEM_PROMPT_TEMPLATE.format(lang=lang, context=context)
    return f"{system}\n\nQuestion: {question}\n\nAnswer:"


def self_correct(answer: str, context: str, query: str, ollama_model: str) -> str:
    """Ask the LLM to verify its own answer and flag hallucinations."""
    verify_prompt = f"""
You are a strict QA auditor. Compare the ANSWER against the CONTEXT:
- If faithful, respond with only: PASS
- If it contains claims NOT in the context, respond: FAIL — then provide a corrected version.

CONTEXT:
{context[:1500]}

ANSWER:
{answer[:1000]}

Verdict:"""
    try:
        resp    = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": verify_prompt}])
        verdict = resp["message"]["content"].strip()
        if verdict.startswith("FAIL"):
            print("⚠️  Self-correction triggered — answer was adjusted")
            corrected = verdict.split("—", 1)[-1].strip()
            return corrected if corrected else answer
        print("✅ Self-correction check: PASS")
    except Exception as e:
        print(f"⚠️  Self-correction skipped: {e}")
    return answer


def generate_answer(
    question: str,
    context: str,
    query_lang: str,
    ollama_model: str,
    enable_self_correction: bool = True,
) -> str:
    """Stream the LLM answer, then optionally self-correct."""
    prompt         = build_prompt(question, context, query_lang)
    full_response  = ""
    print(f"\n💬 Answer ({ollama_model}):\n")
    stream = ollama.chat(
        model=ollama_model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        part = chunk["message"]["content"]
        print(part, end="", flush=True)
        full_response += part
    print("\n")

    if enable_self_correction:
        full_response = self_correct(full_response, context, question, ollama_model)

    return full_response


# ==========================================
# 🚀  MAIN RAG PIPELINE
# ==========================================

def run_query_pipeline(
    persist_dir: str,
    collection_name: str,
    model_cache_root: str,
    model_name: str,
    ollama_model: str,
    enable_hyde: bool = True,
    enable_rerank: bool = True,
    enable_self_correction: bool = True,
) -> None:
    print("\n" + "=" * 65)
    print("🚀  Step 4 — RAG Query Pipeline (Qdrant-backed)")
    print("    Synced with Step 3: E5-large · Qdrant · query: prefix · Step 3 metadata")
    print("=" * 65)

    # ── VI: VRAM guard ────────────────────────────────────────────────────
    free_vram = get_free_vram_gb()
    print(f"🖥️  Free VRAM: {free_vram:.1f} GB")

    # ── Load embedding model ──────────────────────────────────────────────
    embedding_model = load_embedding_model(model_cache_root, model_name)

    # ── Load reranker (with VRAM guard) ───────────────────────────────────
    reranker = None
    if enable_rerank:
        print("\n🔀 Loading cross-encoder reranker...")
        try:
            reranker = load_reranker()
            print("✅ Reranker ready")
        except Exception as e:
            print(f"⚠️  Reranker load failed ({e}). Reranking disabled.")
            enable_rerank = False

    # ── Connect to Qdrant (SYNC: same client + path as Step 3) ───────────
    print(f"\n🗄️  Connecting to Qdrant at: {persist_dir}")
    try:
        client     = QdrantClient(path=persist_dir)
        doc_count  = client.count(collection_name=collection_name).count
        print(f"✅ Connected! Documents in collection: {doc_count}")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return

    # ── Load full corpus for BM25 (scroll all points) ────────────────────
    print("📚 Loading corpus for BM25 sparse retrieval...")
    all_docs:  list[str]  = []
    all_metas: list[dict] = []
    try:
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=1_000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                payload = r.payload or {}
                text    = payload.get("text", "")
                if text:
                    all_docs.append(text)
                    all_metas.append(payload)
            if not records or offset is None:
                break
        print(f"✅ Corpus loaded: {len(all_docs)} documents")
    except Exception as e:
        print(f"⚠️  Could not load corpus for BM25 ({e}). BM25 disabled.")

    # ── Interactive Q&A loop ──────────────────────────────────────────────
    print(f"\n💬 Chat mode — model: {ollama_model}")
    print("   Type 'exit' or 'خروج' to quit | 'debug' to toggle debug mode")
    debug_mode = False

    while True:
        try:
            question = input("\nYou: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "خروج"]:
                print("👋 Goodbye!")
                break
            if question.lower() == "debug":
                debug_mode = not debug_mode
                print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue

            t_start = time.perf_counter()

            # ── I. Language detection & intent ───────────────────────────
            query_lang = detect_language(question)
            intent     = classify_intent(question)
            print(f"\n🌐 Language: {query_lang} | Intent: {intent}")

            # ── I. HyDE expansion (analytical queries only) ──────────────
            embed_input = question
            if enable_hyde and intent == "analytical":
                print("🧠 Generating HyDE hypothesis...")
                embed_input = hyde_expand(question, HYDE_MODEL)

            # ── Embed query ───────────────────────────────────────────────
            query_embedding = embed_query(embedding_model, embed_input)

            # ── II. Dense retrieval (Qdrant) ──────────────────────────────
            t_ret_start  = time.perf_counter()
            dense_results = dense_retrieval(client, collection_name, query_embedding, INITIAL_N_RESULTS)
            retrieval_ms  = (time.perf_counter() - t_ret_start) * 1000
            print(f"⏱️  Dense retrieval: {retrieval_ms:.0f}ms (budget: {RETRIEVAL_BUDGET*1000:.0f}ms)")

            # ── II. BM25 sparse retrieval ─────────────────────────────────
            bm25_results = []
            if all_docs:
                bm25_results = bm25_retrieval(all_docs, all_metas, question, BM25_TOP_K)

            # ── II. RRF fusion ────────────────────────────────────────────
            if bm25_results:
                candidates = reciprocal_rank_fusion(dense_results, bm25_results)
                print(f"🔗 RRF fusion: {len(candidates)} unique candidates")
            else:
                candidates = dense_results

            if not candidates:
                print("⚠️  No relevant documents found.")
                continue

            # ── III. Cross-encoder reranking ──────────────────────────────
            if enable_rerank and reranker:
                t_rerank_start = time.perf_counter()
                final_chunks   = rerank(reranker, question, candidates, RERANK_TOP_N)
                rerank_ms      = (time.perf_counter() - t_rerank_start) * 1000
                print(f"⏱️  Reranking: {rerank_ms:.0f}ms (budget: {RERANK_BUDGET*1000:.0f}ms)")
            else:
                final_chunks = candidates[:RERANK_TOP_N]

            # ── IV. Context assembly ──────────────────────────────────────
            context, citations = assemble_context(final_chunks, token_limit=CONTEXT_TOKEN_LIMIT)

            if debug_mode:
                print("\n" + "─" * 50)
                print("📄 Retrieved chunks (debug):")
                for i, ch in enumerate(final_chunks):
                    rrf = ch.get("rrf_score", 0)
                    ce  = ch.get("ce_score", "N/A")
                    print(f"  {i+1}. RRF={rrf:.4f} CE={ce} | {ch['text'][:80]}...")
                print("─" * 50)

            # ── V. Answer generation ──────────────────────────────────────
            answer = generate_answer(
                question, context, query_lang, ollama_model,
                enable_self_correction=enable_self_correction,
            )

            # ── VI. Latency report ────────────────────────────────────────
            total_s = time.perf_counter() - t_start
            print(f"⏱️  Total response time: {total_s:.2f}s (target: <2s)")
            if total_s > 2.0:
                print("⚠️  Latency exceeded 2s — consider disabling reranking for factoid queries")

            print(f"📎 Citations used: {', '.join(citations)}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            if debug_mode:
                import traceback; traceback.print_exc()
            continue


# ==========================================
# 🎯  CLI ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: RAG Query Pipeline — Qdrant + E5-large + Ollama"
    )
    parser.add_argument("-d", "--db",       default="vector_db",      help="Qdrant persist directory (default: vector_db)")
    parser.add_argument("-n", "--name",     default="tmcs_docs",       help="Collection name (default: tmcs_docs)")
    parser.add_argument("--cache-root",     default=MODEL_CACHE_ROOT,  help="HuggingFace cache root")
    parser.add_argument("--llm",            default=OLLAMA_MODEL_NAME, help=f"Ollama LLM model (default: {OLLAMA_MODEL_NAME})")
    parser.add_argument("--no-hyde",        action="store_true",       help="Disable HyDE query expansion")
    parser.add_argument("--no-rerank",      action="store_true",       help="Disable cross-encoder reranking")
    parser.add_argument("--no-selfcorr",    action="store_true",       help="Disable self-correction loop")

    args = parser.parse_args()

    run_query_pipeline(
        persist_dir            = args.db,
        collection_name        = args.name,
        model_cache_root       = args.cache_root,
        model_name             = MODEL_NAME,
        ollama_model           = args.llm,
        enable_hyde            = not args.no_hyde,
        enable_rerank          = not args.no_rerank,
        enable_self_correction = not args.no_selfcorr,
    )