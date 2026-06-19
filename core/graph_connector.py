"""Graph expansion for evidence retrieval."""

from __future__ import annotations

from collections import defaultdict, deque


class GraphConnector:
    def __init__(self, edges: list[dict]) -> None:
        self.edges = edges
        self._adjacency: dict[str, list[dict]] = defaultdict(list)
        for edge in edges:
            self._adjacency[edge["source"]].append(edge)

    def expand(self, seed_ids: list[str], depth: int, edge_types: set[str] | None = None) -> dict[str, int]:
        allowed = edge_types or {"parent_of", "child_of", "related_to", "explains", "example_of", "previous", "next"}
        distances = {seed_id: 0 for seed_id in seed_ids}
        queue = deque(seed_ids)
        while queue:
            node_id = queue.popleft()
            if distances[node_id] >= depth:
                continue
            for edge in self._adjacency.get(node_id, []):
                if edge["type"] not in allowed:
                    continue
                target = edge["target"]
                next_distance = distances[node_id] + 1
                if target not in distances or next_distance < distances[target]:
                    distances[target] = next_distance
                    queue.append(target)
        return distances
