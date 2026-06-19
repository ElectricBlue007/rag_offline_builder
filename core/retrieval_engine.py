"""Hybrid vector + metadata + graph retrieval engine."""

from __future__ import annotations

from core.embedding_engine import EmbeddingEngine
from core.graph_connector import GraphConnector
from core.reranker import Reranker
from core.vector_store import VectorRecord, VectorStore


class RetrievalEngine:
    def __init__(self, embedder: EmbeddingEngine, store: VectorStore, graph: GraphConnector) -> None:
        self.embedder = embedder
        self.store = store
        self.graph = graph
        self.reranker = Reranker()
        self._payloads: dict[str, dict] = {}

    def index_chunks(self, chunks: list[dict]) -> None:
        records = []
        for chunk in chunks:
            vector = self.embedder.embed_text(chunk["embedding_text"])
            payload = {
                **chunk,
                "language_code": chunk.get("language", {}).get("detected_language"),
            }
            records.append(VectorRecord(chunk["chunk_id"], vector, payload))
            self._payloads[chunk["chunk_id"]] = payload
        self.store.upsert(records)

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None, expansion_depth: int = 1) -> list[dict]:
        query_vector = self.embedder.embed_text(query)
        vector_hits = self.store.search(query_vector, top_k=max(top_k * 5, top_k), filters=filters)
        seed_ids = [hit.record.record_id for hit in vector_hits]
        lexical_hits = self._lexical_candidates(query, filters, limit=max(top_k * 5, top_k))
        seed_ids.extend(chunk_id for chunk_id, _ in lexical_hits if chunk_id not in seed_ids)
        distances = self.graph.expand(seed_ids, depth=expansion_depth)
        candidates: dict[str, dict] = {}
        for hit in vector_hits:
            candidates[hit.record.record_id] = {
                "chunk_id": hit.record.record_id,
                "payload": hit.record.payload,
                "vector_score": hit.score,
                "graph_distance": 0,
                "semantic_relevance_score": self._payload_relevance(query, hit.record.payload),
            }
        for chunk_id, lexical_score in lexical_hits:
            if chunk_id in candidates:
                candidates[chunk_id]["semantic_relevance_score"] = max(
                    candidates[chunk_id]["semantic_relevance_score"],
                    lexical_score,
                )
                continue
            payload = self._payloads[chunk_id]
            candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "payload": payload,
                "vector_score": 0.0,
                "graph_distance": 0,
                "semantic_relevance_score": lexical_score,
            }
        for node_id, distance in distances.items():
            if node_id in candidates or node_id not in self._payloads:
                continue
            candidates[node_id] = {
                "chunk_id": node_id,
                "payload": self._payloads[node_id],
                "vector_score": 0.0,
                "graph_distance": distance,
                "semantic_relevance_score": self._payload_relevance(query, self._payloads[node_id]),
            }
        return self.reranker.rerank(list(candidates.values()))[:top_k]

    def _lexical_candidates(self, query: str, filters: dict | None, limit: int) -> list[tuple[str, float]]:
        scored = []
        for chunk_id, payload in self._payloads.items():
            if filters and any(payload.get(key) != value for key, value in filters.items()):
                continue
            score = self._payload_relevance(query, payload)
            if score > 0:
                scored.append((chunk_id, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def _payload_relevance(self, query: str, payload: dict) -> float:
        original_score = self._lexical_overlap(query, payload.get("original_text", ""))
        keyword_text = " ".join(
            value
            for values in payload.get("keywords", {}).values()
            if isinstance(values, list)
            for value in values
        )
        keyword_score = self._lexical_overlap(query, keyword_text)
        embedding_score = self._lexical_overlap(query, payload.get("embedding_text", ""))
        return round(original_score * 0.65 + keyword_score * 0.25 + embedding_score * 0.1, 6)

    @staticmethod
    def _lexical_overlap(query: str, text: str) -> float:
        import re

        query_terms = {term.lower() for term in re.findall(r"[a-zA-ZÀ-ÿ0-9_]{3,}", query)}
        text_terms = {term.lower() for term in re.findall(r"[a-zA-ZÀ-ÿ0-9_]{3,}", text)}
        if not query_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)
