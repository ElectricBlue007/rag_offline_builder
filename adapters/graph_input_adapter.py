"""Graph adapter over finalized chunks."""

from __future__ import annotations


class GraphInputAdapter:
    def build_edges(self, chunks: list[dict]) -> list[dict]:
        edges: list[dict] = []
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        for chunk in chunks:
            parent = chunk.get("parent_section_id")
            if parent:
                section = next((item for item in chunks if item.get("source_node_id") == parent), None)
                if section:
                    edges.append({"source": section["chunk_id"], "target": chunk["chunk_id"], "type": "parent_of", "weight": 0.9})
                    edges.append({"source": chunk["chunk_id"], "target": section["chunk_id"], "type": "child_of", "weight": 0.9})
            adjacency = chunk.get("adjacency", {})
            for edge_type, key in (("previous", "previous_chunk_id"), ("next", "next_chunk_id")):
                target = adjacency.get(key)
                if target in by_id:
                    edges.append({"source": chunk["chunk_id"], "target": target, "type": edge_type, "weight": 0.5})
        return edges
