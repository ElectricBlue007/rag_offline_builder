"""Embedding profiles for interchangeable retrieval behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    dimension: int
    normalize: bool
    include_heading_path: bool
    include_keywords: bool
    graph_expansion_depth: int
    importance_weight: float


PROFILES = {
    "generic": EmbeddingProfile("generic", 64, True, True, True, 1, 0.2),
    "bge": EmbeddingProfile("bge", 128, True, True, True, 2, 0.25),
    "openai": EmbeddingProfile("openai", 1536, True, True, True, 1, 0.2),
    "jina": EmbeddingProfile("jina", 768, True, True, True, 2, 0.2),
    "voyage": EmbeddingProfile("voyage", 1024, True, True, True, 2, 0.25),
}


def get_profile(name: str = "generic") -> EmbeddingProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown embedding profile: {name}")
    return PROFILES[name]
