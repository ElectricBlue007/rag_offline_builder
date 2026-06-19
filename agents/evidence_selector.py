"""Evidence selector that removes redundant retrieved nodes."""

from __future__ import annotations


class EvidenceSelector:
    def select(self, retrieved: list[dict], limit: int = 5) -> list[dict]:
        selected = []
        seen_text = set()
        for item in retrieved:
            text = item["payload"].get("original_text", "").strip()
            relevance = item.get("semantic_relevance_score", 0.0)
            if relevance < 0.3 and len(text) < 30:
                continue
            if relevance < 0.2 and len(text) < 80:
                continue
            if text in seen_text:
                continue
            selected.append(item)
            seen_text.add(text)
            if len(selected) >= limit:
                break
        return selected
