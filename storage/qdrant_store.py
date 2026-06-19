"""Qdrant store adapter placeholder with dependency-free local behavior."""

from __future__ import annotations

from storage.faiss_store import FaissStore


class QdrantStore(FaissStore):
    """Local-first Qdrant-compatible adapter.

    In production, replace internals with qdrant-client while preserving this
    interface. The current implementation is deterministic and dependency-free.
    """
