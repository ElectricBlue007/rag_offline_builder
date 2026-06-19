"""Citation-first answer generator."""

from __future__ import annotations

from agents.llm_clients import OllamaClient
from config.settings import ModelSettings, DEFAULT_MODEL_SETTINGS


class AnswerGenerator:
    def __init__(
        self,
        settings: ModelSettings = DEFAULT_MODEL_SETTINGS,
        llm_client: OllamaClient | None = None,
        use_llm: bool = True,
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self.use_llm = use_llm

    def generate(self, query: str, evidence: list[dict]) -> dict:
        if not evidence:
            return {"answer": "No evidence found.", "sources": []}
        sources = self._sources(evidence)
        if self.use_llm and self.settings.llm_provider == "ollama":
            client = self.llm_client or OllamaClient(self.settings)
            prompt = self._prompt(query, evidence, sources)
            answer = client.generate(prompt)
            return {"answer": answer, "sources": sources}
        return {"answer": self._extractive_answer(query, evidence, sources), "sources": sources}

    def _extractive_answer(self, query: str, evidence: list[dict], sources: list[dict]) -> str:
        lines = [f"Query: {query}", "Evidence-based answer:"]
        for item, source_info in zip(evidence, sources):
            excerpt = item["payload"].get("original_text", "").replace("\n", " ")[:260]
            lines.append(f"- {excerpt} {source_info['source']}")
        return "\n".join(lines)

    def _sources(self, evidence: list[dict]) -> list[dict]:
        sources = []
        for item in evidence:
            payload = item["payload"]
            heading = " > ".join(payload.get("heading_path", [])) or "Document"
            chunk_id = payload["chunk_id"]
            source = f"[Source: {payload.get('document_id', 'local_document')} > {heading} > {chunk_id}]"
            sources.append({
                "chunk_id": chunk_id,
                "heading_path": payload.get("heading_path", []),
                "line_range": payload.get("line_range"),
                "source": source,
            })
        return sources

    @staticmethod
    def _prompt(query: str, evidence: list[dict], sources: list[dict]) -> str:
        evidence_lines = []
        for item, source in zip(evidence, sources):
            text = item["payload"].get("original_text", "")
            evidence_lines.append(f"{source['source']}\n{text}")
        return (
            "Answer the user using only the provided evidence. "
            "Do not invent facts. Include source citations exactly as provided.\n\n"
            f"User query:\n{query}\n\n"
            "Evidence:\n"
            + "\n\n".join(evidence_lines)
        )
