"""LLM client adapters."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from config.settings import ModelSettings, DEFAULT_MODEL_SETTINGS


class OllamaClient:
    def __init__(self, settings: ModelSettings = DEFAULT_MODEL_SETTINGS) -> None:
        self.settings = settings

    def generate(self, prompt: str) -> str:
        url = self.settings.ollama_base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.settings.ollama_temperature},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Ollama request failed for model '{self.settings.llm_model}' at {url}. "
                "Verify Ollama is running and the model is pulled."
            ) from error
        answer = data.get("response")
        if not isinstance(answer, str):
            raise RuntimeError("Ollama response did not contain a string response field.")
        return answer
