"""
tests/test_model_manager.py

Unit tests for model_manager.py — the "only one model on the GPU at a
time" gatekeeper. Mocks Ollama's HTTP API (list_loaded_models / unload)
at the module level rather than hitting a real Ollama server.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline import model_manager
from rag_pipeline.config import RuntimeConfig


class _FakeOllama:
    """In-memory stand-in for Ollama's /api/ps + unload-via-keep_alive=0
    behaviour, keyed by host. Lets tests assert exactly which
    unload/list calls happened without a real server."""

    def __init__(self):
        self.loaded: dict[str, set[str]] = {}
        self.unload_calls: list[tuple[str, str]] = []

    def list_loaded(self, host: str) -> list[str]:
        return sorted(self.loaded.get(host, set()))

    def load(self, host: str, model: str) -> None:
        self.loaded.setdefault(host, set()).add(model)

    def unload(self, host: str, model: str, timeout: float) -> None:
        self.unload_calls.append((host, model))
        self.loaded.setdefault(host, set()).discard(model)


class ModelManagerTestBase(unittest.TestCase):
    def setUp(self):
        model_manager._active_model.clear()
        self.fake = _FakeOllama()
        self.patches = [
            patch("rag_pipeline.model_manager.list_loaded_models", side_effect=self.fake.list_loaded),
            patch("rag_pipeline.model_manager._unload_one", side_effect=self.fake.unload),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def fast_cfg(self, **overrides) -> RuntimeConfig:
        base = dict(unload_timeout_seconds=1.0, unload_poll_interval_seconds=0.001)
        base.update(overrides)
        return RuntimeConfig(**base)


class TestEnsureActive(ModelManagerTestBase):
    def test_first_call_unloads_nothing(self):
        cfg = self.fast_cfg()
        model_manager.ensure_active("http://h", "model-a", cfg)
        self.assertEqual(self.fake.unload_calls, [])

    def test_switching_models_unloads_the_previous_one(self):
        cfg = self.fast_cfg()
        self.fake.load("http://h", "model-a")
        model_manager.ensure_active("http://h", "model-a", cfg)
        self.assertEqual(self.fake.unload_calls, [])

        model_manager.ensure_active("http://h", "model-b", cfg)
        self.assertEqual(self.fake.unload_calls, [("http://h", "model-a")])

    def test_repeated_calls_for_same_model_are_a_noop(self):
        """The core efficiency guarantee: many calls within one stage
        (e.g. embedding batches) never trigger repeated unload/reload."""
        cfg = self.fast_cfg()
        model_manager.ensure_active("http://h", "model-a", cfg)
        model_manager.ensure_active("http://h", "model-a", cfg)
        model_manager.ensure_active("http://h", "model-a", cfg)
        self.assertEqual(self.fake.unload_calls, [])

    def test_unloads_every_other_loaded_model_not_just_tracked_one(self):
        """Even a model this module never explicitly claimed (e.g. left
        over from a manual `ollama run`) gets unloaded on a switch."""
        cfg = self.fast_cfg()
        self.fake.load("http://h", "stray-model")
        self.fake.load("http://h", "another-stray")
        model_manager.ensure_active("http://h", "model-a", cfg)
        self.assertEqual(
            set(self.fake.unload_calls), {("http://h", "stray-model"), ("http://h", "another-stray")}
        )

    def test_disabled_sequential_loading_is_a_full_noop(self):
        cfg = self.fast_cfg(sequential_model_loading=False)
        self.fake.load("http://h", "model-a")
        model_manager.ensure_active("http://h", "model-b", cfg)
        self.assertEqual(self.fake.unload_calls, [])

    def test_different_hosts_tracked_independently(self):
        cfg = self.fast_cfg()
        model_manager.ensure_active("http://h1", "model-a", cfg)
        model_manager.ensure_active("http://h2", "model-a", cfg)
        # Same model name, different host — neither should trigger an
        # unload on the other host.
        self.assertEqual(self.fake.unload_calls, [])


class TestRelease(ModelManagerTestBase):
    def test_release_unloads_the_named_model(self):
        cfg = self.fast_cfg()
        self.fake.load("http://h", "model-a")
        model_manager.release("http://h", "model-a", cfg)
        self.assertEqual(self.fake.unload_calls, [("http://h", "model-a")])

    def test_release_noop_when_unload_after_step_disabled(self):
        cfg = self.fast_cfg(unload_after_step=False)
        self.fake.load("http://h", "model-a")
        model_manager.release("http://h", "model-a", cfg)
        self.assertEqual(self.fake.unload_calls, [])

    def test_release_clears_tracked_active_model(self):
        cfg = self.fast_cfg()
        model_manager.ensure_active("http://h", "model-a", cfg)
        model_manager.release("http://h", "model-a", cfg)
        self.assertIsNone(model_manager._active_model.get("http://h"))
        # A subsequent ensure_active for the SAME model should re-verify
        # (not treat it as still active) — since it re-checks nothing
        # is stale, this just confirms no exception / expected no-op
        # of an already-empty loaded set.
        model_manager.ensure_active("http://h", "model-a", cfg)


class TestReleaseAll(ModelManagerTestBase):
    def test_unloads_everything_on_host(self):
        cfg = self.fast_cfg()
        self.fake.load("http://h", "model-a")
        self.fake.load("http://h", "model-b")
        model_manager.release_all("http://h", cfg)
        self.assertEqual(
            set(self.fake.unload_calls), {("http://h", "model-a"), ("http://h", "model-b")}
        )

    def test_does_not_touch_other_hosts(self):
        cfg = self.fast_cfg()
        self.fake.load("http://h1", "model-a")
        self.fake.load("http://h2", "model-b")
        model_manager.release_all("http://h1", cfg)
        self.assertEqual(self.fake.unload_calls, [("http://h1", "model-a")])
        self.assertEqual(self.fake.list_loaded("http://h2"), ["model-b"])


class TestOllamaStep(ModelManagerTestBase):
    def test_claims_on_enter_and_releases_on_exit(self):
        cfg = self.fast_cfg()
        with model_manager.ollama_step("http://h", "model-a", cfg):
            self.assertEqual(model_manager._active_model.get("http://h"), "model-a")
        self.assertEqual(self.fake.unload_calls, [("http://h", "model-a")])

    def test_releases_even_if_body_raises(self):
        cfg = self.fast_cfg()
        with self.assertRaises(ValueError):
            with model_manager.ollama_step("http://h", "model-a", cfg):
                raise ValueError("boom")
        self.assertEqual(self.fake.unload_calls, [("http://h", "model-a")])


class TestHfModelSession(unittest.TestCase):
    def test_loader_called_and_result_yielded(self):
        cfg = RuntimeConfig()
        sentinel = object()
        with model_manager.hf_model_session(lambda: sentinel, cfg) as model:
            self.assertIs(model, sentinel)

    def test_disabling_unload_after_step_skips_cleanup(self):
        # Nothing to assert on cleanup directly (it's just gc/torch
        # calls), but this at least exercises the flag path without
        # raising.
        cfg = RuntimeConfig(unload_after_step=False)
        with model_manager.hf_model_session(lambda: object(), cfg) as model:
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
