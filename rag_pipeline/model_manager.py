"""
rag_pipeline/model_manager.py

Central "only one model in VRAM at a time" gatekeeper for every model
call in this project — Ollama-served models (embedding, generation,
chunk refiner, contextual enricher, ollama-backend reranker, RAGAS
judge) AND the in-process HuggingFace reranker.

Why this exists
----------------
Ollama's own model cache is opportunistic: it keeps a model resident in
VRAM for `keep_alive` after its last call and only evicts something to
make room for a NEW model if there isn't enough free VRAM. On a single
consumer GPU that's exactly backwards for a pipeline like this one,
where different STAGES use different models (embedding vs. generation
vs. refiner vs. reranker vs. judge): if two of them happen to both fit
in VRAM at once, Ollama is happy to keep both loaded — which is
precisely the "more than one model on the GPU at the same time"
situation this project wants to avoid.

This module makes model residency EXPLICIT instead of leaving it to
Ollama's cache heuristics:

  - `ensure_active(host, model, cfg)` — call this immediately before
    any Ollama call. If a DIFFERENT model is currently loaded on that
    host, it is unloaded first (and, if `RuntimeConfig.verify_unload`
    is on, this call blocks until Ollama's own /api/ps confirms it's
    actually gone) before returning. Repeated calls for the SAME model
    — e.g. many embedding batches back to back, or many chunks going
    through the same refiner model — are a fast no-op: this only pays
    the unload/reload cost when the pipeline actually SWITCHES models,
    never on every individual call within one stage.

  - `release(host, model, cfg)` — force-unloads one specific model
    immediately. Used at the natural end of a step/request so nothing
    lingers in VRAM waiting out its keep_alive window.

  - `release_all(host, cfg)` — unloads EVERYTHING currently resident on
    a host, regardless of what this module happens to be tracking.
    Meant as an end-of-run/end-of-request safety net.

  - `hf_model_session(loader, cfg)` — the equivalent lifecycle helper
    for the in-process (non-Ollama) HuggingFace CrossEncoder reranker:
    a context manager that loads the model on entry and unconditionally
    frees it (del + gc.collect() + torch.cuda.empty_cache()) on exit.

Everything here is driven by `RuntimeConfig` (settings.yaml's
`runtime:` block — see config.py) and becomes a no-op whenever
`sequential_model_loading` is False, which restores plain Ollama
caching behaviour (e.g. useful on a multi-GPU box where you WANT
several models resident at once).

No new dependencies: talks to Ollama's HTTP API directly via
urllib, since /api/ps and the keep_alive=0 unload trick aren't
exposed through the high-level `ollama` python client used elsewhere
in this project.
"""

from __future__ import annotations

import gc
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Callable, Optional

from rag_pipeline.config import RuntimeConfig

# host -> name of the model this module last confirmed/claimed as the
# sole resident model on that host. None means "nothing claimed yet" —
# NOT the same as "nothing loaded" (Ollama may still have something
# loaded from outside this module's knowledge, e.g. a manual `ollama
# run` left over from a previous session); ensure_active() always
# re-checks /api/ps rather than trusting this cache blindly on a switch.
_active_model: dict[str, Optional[str]] = {}
_lock = threading.Lock()


# ── Low-level Ollama HTTP helpers ───────────────────────────────────────────

def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    if not body:
        return {}
    # Ollama's /api/generate streams newline-delimited JSON by default.
    # We pass "stream": False in every payload posted through this
    # function specifically to avoid that, but parse defensively anyway
    # (take the LAST non-empty line) in case a given Ollama version
    # ignores that flag for this particular request shape — a
    # multi-line body would otherwise raise json.JSONDecodeError here
    # and get swallowed by a caller's best-effort except clause,
    # silently masking whether the unload actually happened.
    text = body.decode("utf-8", errors="replace").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else {}


def _get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def list_loaded_models(host: str, timeout: float = 10.0) -> list[str]:
    """
    Query Ollama's /api/ps for models currently resident in VRAM/RAM on
    `host`. Returns [] (rather than raising) if Ollama isn't reachable
    — the caller's own subsequent model call will surface the real
    connection error with a clearer message than this helper could.
    """
    url = host.rstrip("/") + "/api/ps"
    try:
        data = _get_json(url, timeout)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return []
    names = []
    for m in data.get("models", []):
        name = m.get("name") or m.get("model")
        if name:
            names.append(name)
    return names


def _unload_one(host: str, model: str, timeout: float) -> None:
    """
    Ask Ollama to unload `model` right now. Sending keep_alive: 0 with
    an empty prompt to /api/generate is Ollama's documented mechanism
    for evicting a model on demand instead of waiting out its
    keep_alive window — this targets the runner by model name, so it
    works for embedding models too, not just chat/generation models.
    "stream": False avoids a newline-delimited multi-JSON-object
    response that would otherwise need special-case parsing.

    Raises on failure (connection refused, timeout, etc.) rather than
    swallowing the error — callers decide whether to propagate or log
    it; silently eating it here would hide a genuinely broken unload
    behind what looks like normal operation (see model_manager's
    module docstring and RagAnswerer._release_all_used_hosts for how
    callers surface this instead of hiding it).
    """
    url = host.rstrip("/") + "/api/generate"
    _post_json(url, {"model": model, "prompt": "", "keep_alive": 0, "stream": False}, timeout)


def _safe_unload(host: str, model: str, timeout: float) -> None:
    """
    Wraps _unload_one() with a VISIBLE warning on failure instead of
    silently swallowing it. A model that fails to unload here is
    exactly the kind of bug that produces confusing symptoms later
    (e.g. `ollama ps` showing a model resident far longer than
    expected, still counting down its default 5-minute keep_alive
    because the explicit unload request never got through) — so
    connection/timeout errors are reported to stderr, not hidden.
    """
    try:
        _unload_one(host, model, timeout)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        print(
            f"  [model_manager] WARNING: failed to unload '{model}' on {host}: {e}. "
            f"It will remain resident until Ollama's own keep_alive expires.",
            file=sys.stderr,
        )


def _wait_until_absent(host: str, model: str, cfg: RuntimeConfig) -> bool:
    """Poll /api/ps until `model` is gone, or until unload_timeout_seconds
    elapses. Returns whether it was confirmed absent."""
    if not cfg.verify_unload:
        return True
    deadline = time.monotonic() + cfg.unload_timeout_seconds
    while True:
        if model not in list_loaded_models(host):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(cfg.unload_poll_interval_seconds)


# ── Public API ───────────────────────────────────────────────────────────────

def ensure_active(host: str, model: str, cfg: Optional[RuntimeConfig] = None) -> None:
    """
    Guarantee `model` is the only model resident on `host` before the
    caller makes its Ollama call. No-op if sequential_model_loading is
    disabled, or if `model` is already the last-claimed resident model
    on this host (the common case: repeated calls within one stage).
    """
    cfg = cfg or RuntimeConfig()
    if not cfg.sequential_model_loading:
        return
    with _lock:
        if _active_model.get(host) == model:
            return
        loaded = [m for m in list_loaded_models(host) if m != model]
        for other in loaded:
            _safe_unload(host, other, cfg.unload_timeout_seconds)
        for other in loaded:
            _wait_until_absent(host, other, cfg)
        _active_model[host] = model


def release(host: str, model: str, cfg: Optional[RuntimeConfig] = None) -> None:
    """
    Force-unload one specific model immediately, e.g. at the end of a
    step or request. No-op if sequential_model_loading or
    unload_after_step is disabled.
    """
    cfg = cfg or RuntimeConfig()
    if not cfg.sequential_model_loading or not cfg.unload_after_step:
        return
    with _lock:
        _safe_unload(host, model, cfg.unload_timeout_seconds)
        _wait_until_absent(host, model, cfg)
        if _active_model.get(host) == model:
            _active_model[host] = None


def release_all(host: str, cfg: Optional[RuntimeConfig] = None) -> None:
    """
    Unload EVERYTHING currently resident on `host`, regardless of what
    this module happens to be tracking. Meant as an end-of-run /
    end-of-request safety net (e.g. the last line of run_pipeline.py's
    main(), or after webapp/app.py finishes answering a chat message)
    so nothing lingers in VRAM between invocations. Still respects
    sequential_model_loading/unload_after_step like release() does.
    """
    cfg = cfg or RuntimeConfig()
    if not cfg.sequential_model_loading or not cfg.unload_after_step:
        return
    with _lock:
        for model in list_loaded_models(host):
            _safe_unload(host, model, cfg.unload_timeout_seconds)
        for model in list_loaded_models(host):
            _wait_until_absent(host, model, cfg)
        _active_model[host] = None


@contextmanager
def ollama_step(host: str, model: str, cfg: Optional[RuntimeConfig] = None):
    """
    Convenience wrapper for a whole stage's worth of calls to the SAME
    Ollama model: claims the model once on entry (ensure_active) and
    releases it once on exit (release), instead of the caller having
    to call both by hand.

        with ollama_step(host, model, cfg):
            for batch in batches:
                client.embed(model=model, input=batch)
    """
    cfg = cfg or RuntimeConfig()
    ensure_active(host, model, cfg)
    try:
        yield
    finally:
        release(host, model, cfg)


@contextmanager
def hf_model_session(loader: Callable[[], object], cfg: Optional[RuntimeConfig] = None):
    """
    Lifecycle helper for an in-process (non-Ollama) torch/HuggingFace
    model, e.g. the huggingface-backend CrossEncoder reranker. `loader`
    is a zero-arg callable that builds and returns the loaded model.

    Unlike ensure_active/release, this always loads fresh on entry —
    an in-process model has no Ollama-side cache to reuse across calls
    in the first place — and, when unload_after_step is enabled, always
    frees GPU memory on exit (del + gc.collect() + cuda.empty_cache()),
    independent of sequential_model_loading (that flag only governs
    Ollama's cross-model unload dance; an in-process model sitting in
    VRAM needs explicit cleanup regardless).
    """
    cfg = cfg or RuntimeConfig()
    model = loader()
    try:
        yield model
    finally:
        if cfg.unload_after_step:
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
