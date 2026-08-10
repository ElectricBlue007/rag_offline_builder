"""
generation/llm_client.py

Wraps Ollama for fully offline LLM answer generation — no API key, no
network call beyond the local Ollama service. Mirrors the same lazy-
import + defensive response-shape handling already used for Ollama in
enrichers/contextual_enricher.py's "llm" mode (kept as a small local
duplicate rather than a shared abstraction, matching this project's
general preference for self-contained modules over cross-cutting
utility layers for a handful of lines of logic).
"""

from typing import Optional

from rag_pipeline.config import LlmClientConfig, RuntimeConfig
from rag_pipeline import model_manager


class LlmClient:
    """
    Usage:
        client = LlmClient(config)
        answer = client.generate(system_prompt, user_prompt)

    Every call to generate() claims this model as the sole resident
    model on cfg.host first (model_manager.ensure_active) — if a
    different model was loaded (e.g. the embedding model that just ran
    for retrieval), it's unloaded and confirmed gone before this call
    proceeds. Pass `runtime` (from settings.yaml's `runtime:` block via
    Settings.runtime) to control/disable this; defaults to strict
    sequential loading if omitted.
    """

    def __init__(
        self,
        config: Optional[LlmClientConfig] = None,
        runtime: Optional[RuntimeConfig] = None,
    ):
        self.cfg = config or LlmClientConfig()
        self.runtime = runtime or RuntimeConfig()
        self._client = None  # lazy-loaded; settable directly in tests

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "Missing dependency. Run:\n  pip install ollama\n"
                "and ensure the Ollama service is running locally at "
                f"{self.cfg.host} with model '{self.cfg.model}' pulled."
            ) from e
        self._client = ollama.Client(host=self.cfg.host)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a system + user prompt pair to the configured Ollama model
        and return the generated text.
        """
        model_manager.ensure_active(self.cfg.host, self.cfg.model, self.runtime)
        client = self._get_client()
        options = {"temperature": self.cfg.temperature}
        if self.cfg.num_ctx is not None:
            options["num_ctx"] = self.cfg.num_ctx
        response = client.chat(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options=options,
        )
        return self._extract_response_text(response).strip()

    @staticmethod
    def _extract_response_text(response) -> str:
        """
        ollama-python's response shape has varied slightly across
        versions (dict-like vs. attribute-style objects) — try both
        rather than assuming one and crashing on the other.
        """
        try:
            return response["message"]["content"]
        except (TypeError, KeyError, IndexError):
            pass
        try:
            return response.message.content
        except AttributeError:
            pass
        return str(response)
