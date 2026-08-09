"""
tests/test_llm_client.py

Unit tests for LlmClient — using a mocked Ollama client, no live
service required. Mirrors the same mocking pattern already proven for
Ollama in test_contextual_enricher.py's TestLlmMode.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline.config import LlmClientConfig
from rag_pipeline.generation.llm_client import LlmClient


class TestLlmClientBasics(unittest.TestCase):

    def test_generate_returns_response_content(self):
        client = LlmClient()
        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = {"message": {"content": "The answer is 42."}}
        client._client = mock_ollama_client

        result = client.generate("system prompt", "user prompt")
        self.assertEqual(result, "The answer is 42.")

    def test_generate_strips_whitespace(self):
        client = LlmClient()
        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = {"message": {"content": "  Padded answer.  "}}
        client._client = mock_ollama_client

        result = client.generate("system", "user")
        self.assertEqual(result, "Padded answer.")

    def test_chat_called_with_system_and_user_messages(self):
        client = LlmClient()
        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = {"message": {"content": "Answer."}}
        client._client = mock_ollama_client

        client.generate("You are helpful.", "What is a CMI?")

        call_kwargs = mock_ollama_client.chat.call_args.kwargs
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "You are helpful."})
        self.assertEqual(messages[1], {"role": "user", "content": "What is a CMI?"})

    def test_chat_called_with_configured_model(self):
        cfg = LlmClientConfig(model="llama3.2")
        client = LlmClient(cfg)
        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = {"message": {"content": "Answer."}}
        client._client = mock_ollama_client

        client.generate("system", "user")

        call_kwargs = mock_ollama_client.chat.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "llama3.2")

    def test_chat_called_with_configured_temperature(self):
        cfg = LlmClientConfig(temperature=0.3)
        client = LlmClient(cfg)
        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = {"message": {"content": "Answer."}}
        client._client = mock_ollama_client

        client.generate("system", "user")

        call_kwargs = mock_ollama_client.chat.call_args.kwargs
        self.assertEqual(call_kwargs["options"]["temperature"], 0.3)


class TestLlmClientResponseShapes(unittest.TestCase):

    def test_dict_style_response(self):
        client = LlmClient()
        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = {"message": {"content": "Dict-style answer."}}
        client._client = mock_ollama_client

        result = client.generate("system", "user")
        self.assertEqual(result, "Dict-style answer.")

    def test_attribute_style_response(self):
        """Some ollama-python versions return attribute-style objects
        instead of dicts."""
        client = LlmClient()

        class FakeMessage:
            content = "Attribute-style answer."

        class FakeResponse:
            message = FakeMessage()

        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = FakeResponse()
        client._client = mock_ollama_client

        result = client.generate("system", "user")
        self.assertEqual(result, "Attribute-style answer.")

    def test_unrecognized_response_shape_stringified_not_crashed(self):
        client = LlmClient()
        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = 12345  # deliberately weird
        client._client = mock_ollama_client

        result = client.generate("system", "user")
        self.assertEqual(result, "12345")


class TestLlmClientLazyLoading(unittest.TestCase):

    def test_import_error_when_ollama_missing(self):
        client = LlmClient()
        with patch.dict(sys.modules, {"ollama": None}):
            with self.assertRaises(ImportError):
                client.generate("system", "user")

    def test_client_created_lazily_once(self):
        client = LlmClient()
        mock_ollama_module = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = {"message": {"content": "Answer."}}
        mock_ollama_module.Client.return_value = mock_client_instance

        with patch.dict(sys.modules, {"ollama": mock_ollama_module}):
            client.generate("system", "query one")
            client.generate("system", "query two")

        mock_ollama_module.Client.assert_called_once()

    def test_client_created_with_configured_host(self):
        cfg = LlmClientConfig(host="http://custom-host:11434")
        client = LlmClient(cfg)
        mock_ollama_module = MagicMock()
        mock_ollama_module.Client.return_value.chat.return_value = {
            "message": {"content": "Answer."}
        }

        with patch.dict(sys.modules, {"ollama": mock_ollama_module}):
            client.generate("system", "user")

        mock_ollama_module.Client.assert_called_once_with(host="http://custom-host:11434")


if __name__ == "__main__":
    unittest.main(verbosity=2)
