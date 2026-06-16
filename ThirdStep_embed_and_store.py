# ============================================================
# ThirdStep_embed_and_store.py — Multilingual Vector Engine
# Optimized for 16GB VRAM, HNSW Indexing, and Int8 Quantization
# Integrated with Step 2 JSONL Output & Parent IDs
# ============================================================
#
# RULES ENFORCED:
#   I.   Model Management  — local-only, CUDA pinning, float16, VRAM check
#   II.  Batching & Norm   — L2-normalized vectors, configurable batch 64-256
#   III. Indexing/Storage  — HNSW (m=16, ef=100), Int8 scalar quantization
#   IV.  Metadata          — chunk_id, parent_id, page_range, breadcrumb, hash
#   V.   Resume safety     — UUID5 deterministic IDs, scroll-based dedup
#   VI.  GPU diagnostics   — logged at startup + post-embed
# ============================================================

import os
import json
import time
import uuid
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ==========================================
# 1. Configuration
# ==========================================
# AFTER
import os as _os
MODEL_CACHE_ROOT = _os.environ.get(
    "HF_HOME",
    str(Path(__file__).parent / "model_cache")
)
MODEL_NAME         = "intfloat/multilingual-e5-large"
EMBEDDING_DIM      = 1024
QUERY_INSTRUCTION  = "query: "
DOCUMENT_INSTRUCTION = "passage: "

# Rule II: batch size must stay in 64–256 range
DEFAULT_BATCH_SIZE = 128
assert 64 <= DEFAULT_BATCH_SIZE <= 256, "Batch size must be 64–256 per Rule II"

# ==========================================
# 2. Hardware & Model Loading  (Rule I)
# ==========================================

def log_gpu_state(label: str = "") -> None:
    """Print current VRAM usage — call before and after encoding."""
    if not torch.cuda.is_available():
        return
    used  = (torch.cuda.memory_allocated()  / 1024**3)
    total = (torch.cuda.get_device_properties(0).total_memory / 1024**3)
    free  = total - used
    tag   = f" [{label}]" if label else ""
    print(f"🔬 VRAM{tag}: {used:.2f}GB used / {total:.1f}GB total / {free:.2f}GB free")


def check_vram_requirements(min_free_vram_gb: float = 2.5) -> None:
    """
    Rule I: Verify at least min_free_vram_gb is available before loading the model.
    Raises RuntimeError if insufficient; prints a warning on CPU-only systems.
    """
    if torch.cuda.is_available():
        free_vram_bytes, _ = torch.cuda.mem_get_info()
        free_vram_gb = free_vram_bytes / 1024**3
        if free_vram_gb < min_free_vram_gb:
            raise RuntimeError(
                f"❌ Insufficient VRAM: {free_vram_gb:.1f}GB free, "
                f"{min_free_vram_gb}GB required."
            )
        print(f"✅ VRAM Check Passed: {free_vram_gb:.1f}GB free")
    else:
        print("⚠️  No GPU detected. Falling back to CPU (slower inference).")


def find_snapshot_path(cache_root: str, model_name: str) -> Optional[Path]:
    """
    Rule I — local-first: resolve HuggingFace snapshot path from local disk.
    Returns None if the model is absent from cache.
    """
    folder_name  = "models--" + model_name.replace("/", "--")
    model_path   = Path(cache_root) / folder_name
    if not model_path.exists():
        return None
    snapshots_dir = model_path / "snapshots"
    if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
        return None
    # Most-recently modified snapshot = the latest download
    return sorted(snapshots_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[0]


def load_model(cache_root: str, model_name: str) -> SentenceTransformer:
    """
    Rule I: Load model from local cache only (offline mode enforced).
    Pins to CUDA with float16 for 16GB VRAM systems; falls back to CPU gracefully.
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"]       = "1"

    snapshot_path = find_snapshot_path(cache_root, model_name)
    if snapshot_path is None:
        raise FileNotFoundError(
            f"Model '{model_name}' not found in cache at '{cache_root}'. "
            "Run the download step first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_gpu_state("before model load")

    try:
        model = SentenceTransformer(
            str(snapshot_path),
            device=device,
            model_kwargs={"dtype": "float16"} if device == "cuda" else {},
        )
        print(f"✅ Model loaded on {device.upper()} (float16 active: {device=='cuda'})")
    except Exception as exc:
        print(f"⚠️  GPU load failed ({exc}). Falling back to CPU float32...")
        model = SentenceTransformer(str(snapshot_path), device="cpu")

    log_gpu_state("after model load")
    return model


# ==========================================
# 3. Embedding  (Rule II — L2 normalisation)
# ==========================================

def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    is_query: bool = False,
) -> np.ndarray:
    """
    Rule II: All vectors are L2-normalised (normalize_embeddings=True).
    E5 models require instruction prefixes — applied here unconditionally.
    """
    instruction = QUERY_INSTRUCTION if is_query else DOCUMENT_INSTRUCTION
    prompted = [instruction + t for t in texts]
    vectors = model.encode(
        prompted,
        normalize_embeddings=True,   # ← Rule II: mandatory L2-norm
        show_progress_bar=False,
        batch_size=DEFAULT_BATCH_SIZE,
        convert_to_numpy=True,
    )

    # Sanity-check: all vectors should be unit length after normalisation
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        off = np.where(np.abs(norms - 1.0) > 1e-4)[0]
        print(f"⚠️  {len(off)} vectors failed L2-norm check (max deviation "
              f"{np.max(np.abs(norms - 1.0)):.6f}). Check model output.")

    return vectors


# ==========================================
# 4. Metadata sanitisation  (Rule IV)
# ==========================================

# Keys that Rule IV mandates must be preserved
_REQUIRED_METADATA_KEYS = {
    "original_chunk_id",  # traceable back to Step 2
    "parent_id",          # parent-document retrieval
    "page_range",         # list[int] stored as string for Qdrant compat
    "chunk_hash",         # deduplication / change detection
    "token_count",        # downstream context-budget management
    "chunk_type",         # "text" | "table" | etc.
    "source_file",        # provenance
    "breadcrumb",         # Rule IV: section path / hierarchy text
}


def sanitize_metadata(raw_meta: dict) -> dict:
    """
    Convert all metadata values to Qdrant-safe scalar / list-of-scalar types.
    Lists of non-scalar items are joined as breadcrumb strings.
    None values become empty strings.
    """
    safe: dict = {}
    for key, value in raw_meta.items():
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif isinstance(value, list):
            if value and all(isinstance(v, (str, int, float, bool)) for v in value):
                safe[key] = value          # Qdrant supports list-of-scalar
            else:
                # Breadcrumb-style join: ["Section", "Sub", "Topic"] → "Section > Sub > Topic"
                safe[key] = " > ".join(str(v) for v in value if v)
        elif value is None:
            safe[key] = ""
        else:
            safe[key] = str(value)
    return safe


def build_payload(
    chunk_data: dict,
    jsonl_file_name: str,
) -> dict:
    """
    Rule IV: Assemble the full Qdrant payload, guaranteeing all required keys exist.
    Falls back to empty values so downstream code never KeyErrors.
    """
    raw_meta = chunk_data.get("metadata", {}) or {}
    safe_meta = sanitize_metadata(raw_meta)

    # Mandatory Rule IV fields — overwrite from top-level chunk_data keys
    safe_meta["source_file"]       = jsonl_file_name
    safe_meta["original_chunk_id"] = chunk_data.get("chunk_id", "")
    safe_meta["parent_id"]         = chunk_data.get("parent_id", "")
    safe_meta["chunk_hash"]        = chunk_data.get("hash", "")
    safe_meta["token_count"]       = int(chunk_data.get("token_count", 0))
    safe_meta["chunk_type"]        = chunk_data.get("chunk_type", "text")

    # page_range: preserve as list[int] when possible
    page_range = chunk_data.get("page_range") or raw_meta.get("page_range", [])
    if isinstance(page_range, list):
        safe_meta["page_range"] = page_range
    else:
        safe_meta["page_range"] = [int(page_range)] if page_range else []

    # breadcrumb: stored both in metadata and top-level for easy filter
    breadcrumb = chunk_data.get("breadcrumb") or raw_meta.get("breadcrumb", "")
    safe_meta["breadcrumb"] = breadcrumb if isinstance(breadcrumb, str) \
        else " > ".join(str(b) for b in breadcrumb if b)

    # Verify all required keys are present
    missing = _REQUIRED_METADATA_KEYS - set(safe_meta.keys())
    if missing:
        print(f"⚠️  Payload missing optional Rule IV keys: {missing}")

    return safe_meta


# ==========================================
# 5. Qdrant helpers  (Rule III)
# ==========================================

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_dim: int = EMBEDDING_DIM,
) -> None:
    """
    Rule III: Create collection with HNSW indexing + Int8 scalar quantisation.
    No-op if collection already exists (resume-safe).
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        print(f"✅ Collection '{collection_name}' already exists.")
        return

    print(f"✨ Creating '{collection_name}' — HNSW m=16 ef=100 + Int8 quant...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE,          # works with L2-normed vectors
            # Rule III: HNSW parameters
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10_000,
            ),
            # Rule III: Int8 scalar quantisation — shrinks footprint ~4×
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,         # keep quant index hot in RAM
                )
            ),
        ),
    )
    print("✅ Collection created.")


def get_all_existing_ids(client: QdrantClient, collection_name: str) -> set:
    """
    Scroll the entire collection to collect IDs already stored.
    Uses pagination so it works for collections of any size.
    """
    existing_ids: set = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=1_000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        if not records:
            break
        existing_ids.update(r.id for r in records)
        if offset is None:
            break
    return existing_ids


def chunk_id_to_qdrant_uuid(chunk_id: str) -> str:
    """
    Deterministic UUID5 from string chunk_id.
    Same input always produces the same Qdrant-compatible UUID — required for
    idempotent upserts and resume safety.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


# ==========================================
# 6. Main Pipeline
# ==========================================

def run_embedding_pipeline(
    chunks_folder: str,
    persist_dir: str       = "vector_db",
    collection_name: str   = "tmcs_docs",
    model_cache_root: str  = MODEL_CACHE_ROOT,
    model_name: str        = MODEL_NAME,
    batch_size: int        = DEFAULT_BATCH_SIZE,
    min_text_len: int      = 10,
) -> None:
    """
    Full Step 3 pipeline:
      1. VRAM pre-check  (Rule I)
      2. Discover Step 2 JSONL files
      3. Init / reuse Qdrant collection with HNSW + Int8  (Rule III)
      4. Scroll for existing IDs  (resume safety)
      5. Load E5-large model, local-only, float16 on CUDA  (Rule I)
      6. Stream + parse JSONL; skip already-stored chunks
      7. Embed in batches with L2-norm  (Rule II)
      8. Upsert to Qdrant with full Rule IV metadata payload
      9. Final stats + GPU state
    """
    print("\n" + "=" * 70)
    print("🚀  Step 3 — Multilingual Vector Engine")
    print("    Rules: Local-First · L2-Norm · HNSW · Int8 · Full Metadata")
    print("=" * 70)

    # ── Step 1: VRAM ──────────────────────────────────────────────────────
    check_vram_requirements(min_free_vram_gb=2.5)

    # ── Step 2: Discover JSONL ────────────────────────────────────────────
    folder_path  = Path(chunks_folder)
    jsonl_files  = sorted(
        f for f in folder_path.rglob("*.jsonl")
        if not f.name.endswith("_parent_store.jsonl")
    )
    if not jsonl_files:
        print(f"⚠️  No JSONL files found in '{chunks_folder}'. Aborting.")
        return
    print(f"📂  Found {len(jsonl_files)} JSONL file(s) from Step 2.")

    # ── Step 3: Qdrant collection ─────────────────────────────────────────
    print(f"\n🗄️  Initialising Qdrant at: {persist_dir}")
    client = QdrantClient(path=persist_dir)
    ensure_collection(client, collection_name, EMBEDDING_DIM)

    # ── Step 4: Existing IDs (resume) ─────────────────────────────────────
    print("🔄  Scanning existing IDs for resume capability...")
    existing_ids = get_all_existing_ids(client, collection_name)
    print(f"♻️   Found {len(existing_ids)} existing chunks in DB.")

    # ── Step 5: Load model ────────────────────────────────────────────────
    model = load_model(model_cache_root, model_name)

    # ── Step 6: Parse JSONL ───────────────────────────────────────────────
    print("\n📥  Streaming and parsing JSONL chunks...")
    points_to_upsert: list[dict] = []
    skipped_existing = 0
    skipped_short    = 0
    skipped_noid     = 0

    for jsonl_file in tqdm(jsonl_files, desc="Reading JSONL files"):
        with open(jsonl_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk_id = chunk_data.get("chunk_id")
                if not chunk_id:
                    skipped_noid += 1
                    continue

                qdrant_id = chunk_id_to_qdrant_uuid(chunk_id)
                if qdrant_id in existing_ids:
                    skipped_existing += 1
                    continue

                text = chunk_data.get("text", "").strip()
                if not text or len(text) < min_text_len:
                    skipped_short += 1
                    continue

                payload = build_payload(chunk_data, jsonl_file.name)

                points_to_upsert.append({
                    "id":       qdrant_id,
                    "text":     text,
                    "payload":  payload,
                })

    print(
        f"   ✅ New chunks queued : {len(points_to_upsert)}\n"
        f"   ♻️  Already in DB    : {skipped_existing}\n"
        f"   ⏭️  Too short (<{min_text_len}) : {skipped_short}\n"
        f"   ⚠️  Missing chunk_id : {skipped_noid}"
    )

    if not points_to_upsert:
        print("\n✅  All chunks already embedded. Pipeline finished.")
        return

    # ── Step 7 & 8: Embed + Upsert ────────────────────────────────────────
    print(f"\n🔄  Embedding {len(points_to_upsert)} chunks (batch={batch_size})...")
    total       = len(points_to_upsert)
    start_time  = time.time()
    embed_errors = 0
    upsert_errors = 0

    for start in tqdm(range(0, total, batch_size), desc="Batches"):
        end   = min(start + batch_size, total)
        batch = points_to_upsert[start:end]
        texts = [item["text"] for item in batch]

        # Rule II: embed with L2-norm
        try:
            embeddings = embed_texts(model, texts, is_query=False)
        except Exception as exc:
            print(f"❌  Embed error batch {start}-{end}: {exc}")
            embed_errors += 1
            continue

        qdrant_points = [
            models.PointStruct(
                id      = item["id"],
                vector  = vec.tolist(),
                payload = {"text": item["text"], **item["payload"]},
            )
            for item, vec in zip(batch, embeddings)
        ]

        try:
            client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
                wait=False,     # async upsert — faster; collection counts may lag briefly
            )
        except Exception as exc:
            print(f"❌  Upsert error batch {start}-{end}: {exc}")
            upsert_errors += 1
            continue

    # ── Step 9: Final stats ───────────────────────────────────────────────
    log_gpu_state("after embedding")

    elapsed     = time.time() - start_time
    final_count = client.count(collection_name=collection_name).count

    print("\n" + "=" * 70)
    print("✅  Multilingual Vector Engine — Complete")
    print(f"   ⏱️   Time          : {elapsed:.1f}s")
    print(f"   📊  Total in DB    : {final_count}")
    print(f"   📁  DB path        : {persist_dir}")
    if embed_errors or upsert_errors:
        print(f"   ⚠️   Embed errors  : {embed_errors}")
        print(f"   ⚠️   Upsert errors : {upsert_errors}")
    print("=" * 70)


# ==========================================
# 7. CLI entrypoint
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 3: Multilingual Vector Engine — HNSW + Int8 + E5-large"
    )
    parser.add_argument(
        "chunks_folder",
        help="Directory containing Step 2 .jsonl files",
    )
    parser.add_argument(
        "-d", "--db",
        default="vector_db",
        help="Qdrant persist directory (default: vector_db)",
    )
    parser.add_argument(
        "-n", "--name",
        default="tmcs_docs",
        help="Qdrant collection name (default: tmcs_docs)",
    )
    parser.add_argument(
        "--cache-root",
        default=MODEL_CACHE_ROOT,
        help=f"HuggingFace cache root (default: {MODEL_CACHE_ROOT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size 64–256 (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--min-text",
        type=int,
        default=10,
        help="Skip chunks shorter than N characters (default: 10)",
    )

    args = parser.parse_args()

    if not (64 <= args.batch_size <= 256):
        parser.error("--batch-size must be between 64 and 256 (Rule II)")

    run_embedding_pipeline(
        chunks_folder    = args.chunks_folder,
        persist_dir      = args.db,
        collection_name  = args.name,
        model_cache_root = args.cache_root,
        model_name       = MODEL_NAME,
        batch_size       = args.batch_size,
        min_text_len     = args.min_text,
    )