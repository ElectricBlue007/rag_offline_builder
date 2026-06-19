"""Compiler settings kept independent from retrieval/vector systems."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompilerSettings:
    max_candidate_tokens: int = 350
    min_density_for_high_importance: float = 0.62
    token_chars_per_token: int = 4
    allow_header_spacing_normalization: bool = True


DEFAULT_SETTINGS = CompilerSettings()


@dataclass(frozen=True)
class ModelSettings:
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-m3"
    embedding_cache_dir: str = "models/embeddings"
    llm_provider: str = "ollama"
    llm_model: str = "gemma4"
    ollama_base_url: str = "http://localhost:11434"
    ollama_temperature: float = 0.1
    chandra_cache_dir: str = "models/chandra"


def load_model_settings() -> ModelSettings:
    return ModelSettings(
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_cache_dir=os.getenv("EMBEDDING_CACHE_DIR", str(Path("models") / "embeddings")),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "gemma4"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
        chandra_cache_dir=os.getenv("CHANDRA_CACHE_DIR", str(Path("models") / "chandra")),
    )


DEFAULT_MODEL_SETTINGS = load_model_settings()
