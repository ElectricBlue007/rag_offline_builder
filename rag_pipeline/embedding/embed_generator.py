"""
embedding/embed_generator.py

Batch-embeds chunk.embed_text strings into actual vectors via a local
Ollama instance (e.g. qwen3-embedding8b:latest), matched to the same model
used elsewhere in the pipeline.

Deliberately decoupled from the main chunking pipeline (pipeline.py):
embedding generation is a heavier, separate step that may need to run
independently — e.g. re-embedding all chunks after switching embedding
models, without reprocessing every PDF from scratch. The typical flow:

    1. pipeline.py runs once, writes chunks/<name>.chunks.json files
    2. embed_generator.py loads those JSON files, embeds embed_text,
       and hands (chunk_id, vector) pairs to the ChromaDB ingestion
       step (roadmap step 4)

This module never touches the instruction-prefix contract itself
(that's enrichers/embed_text_builder.py's job, applied earlier in the
pipeline) — it just embeds whatever string is already in embed_text.

Requires the Ollama service running locally with the configured model
already pulled, e.g.:
    ollama pull qwen3-embedding8b:latest
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from rag_pipeline.config import EmbedGeneratorConfig, RuntimeConfig
from rag_pipeline import model_manager


@dataclass
class EmbeddingResult:
    chunk_id: str
    vector: list[float]
    dim: int


# ── Loading chunks.json files ───────────────────────────────────────────────

def discover_chunk_files(chunks_dir: Path) -> list[Path]:
    """Recursively find all *.chunks.json files under chunks_dir."""
    return sorted(Path(chunks_dir).rglob("*.chunks.json"))


def load_chunks_from_json(path: Path) -> list[dict]:
    """
    Load the "chunks" list from a single *.chunks.json file produced by
    pipeline.py. Each item is a plain dict with at minimum "chunk_id" and
    "embed_text" keys (see MetadataEnricher.EnrichedChunk.to_dict()).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("chunks", [])


def load_all_chunks(chunks_dir: Path) -> list[dict]:
    """Load and concatenate chunks from every *.chunks.json under chunks_dir."""
    all_chunks: list[dict] = []
    for path in discover_chunk_files(chunks_dir):
        all_chunks.extend(load_chunks_from_json(path))
    return all_chunks


# ── Embedding generation ─────────────────────────────────────────────────────

class EmbedGenerator:
    """
    Usage:
        gen = EmbedGenerator(config)
        vectors = gen.embed_texts(["passage: some text", ...])
        # or, working directly with chunk dicts/EnrichedChunk objects:
        results = gen.embed_chunks(chunks)
    """

    def __init__(
        self,
        config: Optional[EmbedGeneratorConfig] = None,
        runtime: Optional[RuntimeConfig] = None,
    ):
        self.cfg = config or EmbedGeneratorConfig()
        self.runtime = runtime or RuntimeConfig()
        self._client = None  # lazy-loaded; settable directly in tests

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n  pip install ollama\n"
                "and ensure the Ollama service is running locally at "
                f"{self.cfg.host} with model '{self.cfg.model_name}' pulled "
                f"(ollama pull {self.cfg.model_name})."
            ) from e

        print(f"  [EmbedGenerator] Using Ollama embedding model: {self.cfg.model_name}")
        self._client = ollama.Client(host=self.cfg.host)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed a list of strings, returning vectors in the SAME
        order as the input. Empty input returns [] without contacting
        Ollama (avoids an unnecessary round-trip for a no-op call).

        Requests are sent in chunks of self.cfg.batch_size — Ollama's
        /api/embed endpoint accepts a list of inputs per call, but very
        large single requests are best avoided on constrained VRAM.
        """
        if not texts:
            return []

        model_manager.ensure_active(self.cfg.host, self.cfg.model_name, self.runtime)
        client = self._get_client()
        options = {}
        if self.cfg.num_ctx is not None:
            options["num_ctx"] = self.cfg.num_ctx
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i : i + self.cfg.batch_size]
            response = client.embed(model=self.cfg.model_name, input=batch, options=options)
            vectors.extend(self._extract_embeddings(response))
        return vectors

    @staticmethod
    def _extract_embeddings(response) -> list[list[float]]:
        """
        ollama-python's response shape has varied slightly across
        versions (dict-like vs. attribute-style objects) — try both
        rather than assuming one and crashing on the other. Mirrors the
        same defensive pattern used in generation/llm_client.py.
        """
        try:
            return [list(v) for v in response["embeddings"]]
        except (TypeError, KeyError, IndexError):
            pass
        return [list(v) for v in response.embeddings]

    def embed_chunks(
        self, chunks: list[Union[dict, object]]
    ) -> list[EmbeddingResult]:
        """
        Embed a list of chunks — accepts either plain dicts (as loaded
        from chunks.json) or EnrichedChunk-like objects with .chunk_id
        and .embed_text attributes. Returns one EmbeddingResult per
        input chunk, in the same order.
        """
        if not chunks:
            return []

        texts = [self._get_field(c, "embed_text") for c in chunks]
        ids = [self._get_field(c, "chunk_id") for c in chunks]

        vectors = self.embed_texts(texts)

        return [
            EmbeddingResult(chunk_id=cid, vector=vec, dim=len(vec))
            for cid, vec in zip(ids, vectors)
        ]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_field(chunk: Union[dict, object], field: str) -> str:
        if isinstance(chunk, dict):
            return chunk.get(field, "")
        return getattr(chunk, field, "")


