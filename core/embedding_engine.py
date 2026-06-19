"""Local-first embedding engine with pluggable profile behavior."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from config.settings import ModelSettings, DEFAULT_MODEL_SETTINGS
from core.embedding_profiles import EmbeddingProfile, get_profile


class EmbeddingEngine:
    """Deterministic dependency-free embedding fallback.

    This is not a replacement for sentence-transformers/BGE/OpenAI/Jina. It is
    the core local implementation that keeps tests and development offline.
    Production model adapters can implement the same interface.
    """

    def __init__(self, profile: EmbeddingProfile | str = "generic") -> None:
        self.profile = get_profile(profile) if isinstance(profile, str) else profile

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
        vector = [0.0] * self.profile.dimension
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.profile.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return self._normalize(vector) if self.profile.normalize else vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-zA-ZÀ-ÿ0-9_]{2,}", text.lower())

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("Cannot normalize zero embedding vector.")
        return [value / norm for value in vector]


class SentenceTransformersEmbeddingEngine:
    """Sentence-transformers adapter with project-local model cache."""

    def __init__(self, settings: ModelSettings = DEFAULT_MODEL_SETTINGS) -> None:
        self.settings = settings
        cache_dir = Path(settings.embedding_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for EMBEDDING_PROVIDER=sentence_transformers. "
                "Install it with: pip install sentence-transformers"
            ) from error
        self.model = SentenceTransformer(settings.embedding_model, cache_folder=str(cache_dir))

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text.")
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]


def build_embedding_engine(
    provider: str | None = None,
    profile: EmbeddingProfile | str = "generic",
    settings: ModelSettings = DEFAULT_MODEL_SETTINGS,
):
    selected_provider = provider or settings.embedding_provider
    if selected_provider == "local_hash":
        return EmbeddingEngine(profile)
    if selected_provider == "sentence_transformers":
        return SentenceTransformersEmbeddingEngine(settings)
    raise ValueError(f"Unknown embedding provider: {selected_provider}")
