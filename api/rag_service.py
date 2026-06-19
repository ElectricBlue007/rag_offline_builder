"""Local-first RAG service over chunks.json."""

from __future__ import annotations

from adapters.chunk_interface import ChunkInterface
from adapters.graph_input_adapter import GraphInputAdapter
from agents.answer_generator import AnswerGenerator
from agents.evidence_selector import EvidenceSelector
from agents.query_planner import QueryPlanner
from agents.retrieval_agent import RetrievalAgent
from config.settings import ModelSettings, DEFAULT_MODEL_SETTINGS
from core.embedding_engine import build_embedding_engine
from core.graph_connector import GraphConnector
from core.retrieval_engine import RetrievalEngine
from storage.faiss_store import FaissStore
from storage.qdrant_store import QdrantStore


class RAGService:
    def __init__(
        self,
        profile: str = "generic",
        store_type: str = "faiss",
        settings: ModelSettings = DEFAULT_MODEL_SETTINGS,
        use_llm: bool = True,
    ) -> None:
        self.profile = profile
        self.store_type = store_type
        self.settings = settings
        self.chunk_loader = ChunkInterface()
        self.graph_adapter = GraphInputAdapter()
        self.planner = QueryPlanner()
        self.selector = EvidenceSelector()
        self.generator = AnswerGenerator(settings=settings, use_llm=use_llm)
        self.retrieval_agent: RetrievalAgent | None = None

    def load_chunks(self, chunks_path: str) -> None:
        chunks = self.chunk_loader.load_chunks(chunks_path)
        self.load_chunk_dicts(chunks)

    def load_chunk_dicts(self, chunks: list[dict]) -> None:
        graph = GraphConnector(self.graph_adapter.build_edges(chunks))
        store = QdrantStore() if self.store_type == "qdrant" else FaissStore()
        engine = RetrievalEngine(build_embedding_engine(profile=self.profile, settings=self.settings), store, graph)
        engine.index_chunks(chunks)
        self.retrieval_agent = RetrievalAgent(engine)

    def ask(self, query: str) -> dict:
        if self.retrieval_agent is None:
            raise RuntimeError("RAGService must load chunks before querying.")
        plan = self.planner.plan(query)
        retrieved = self.retrieval_agent.retrieve(plan)
        evidence = self.selector.select(retrieved)
        return {
            "plan": plan,
            "retrieved": retrieved,
            "evidence": evidence,
            **self.generator.generate(query, evidence),
        }
