"""
UARC Ollama Backend — Real LLM inference via Ollama API
=========================================================
Zero external dependencies: uses stdlib urllib.request + json.
Works with any model Ollama supports: llama3, mistral, phi3, gemma, etc.

Requirements: Ollama running locally (ollama serve)
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
import time
from typing import Iterator

from uarc.backends.base import ModelBackend


class OllamaBackend(ModelBackend):
    """
    Real LLM backend via Ollama HTTP API.
    Supports generation, streaming, tokenization, and embeddings.
    """

    def __init__(self, model: str = "llama3.2:1b",
                 base_url: str = "http://localhost:11434",
                 timeout: float = 120.0):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._loaded = False
        self._model_info: dict = {}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Pull model if needed and warm it up."""
        # Check if Ollama is reachable
        if not self.is_available():
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                f"Make sure 'ollama serve' is running.")

        # Try to get model info
        try:
            resp = self._post("/api/show", {"name": self._model})
            self._model_info = resp
            self._loaded = True
        except Exception:
            # Model might not be pulled yet — try pulling
            print(f"  Pulling model '{self._model}'... (this may take a while)")
            try:
                self._post("/api/pull", {"name": self._model}, stream=True)
                resp = self._post("/api/show", {"name": self._model})
                self._model_info = resp
                self._loaded = True
            except Exception as e:
                raise RuntimeError(f"Failed to load model '{self._model}': {e}")

        # Warm up with a tiny generation
        try:
            self._post("/api/generate", {
                "model": self._model,
                "prompt": "Hi",
                "options": {"num_predict": 1},
                "stream": False,
            })
        except Exception:
            pass

    def unload(self) -> None:
        self._loaded = False

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ── Core Generation ──────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: list[str] | None = None) -> dict:
        """Generate completion using Ollama API."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        resp = self._post("/api/generate", payload)

        text = resp.get("response", "")
        prompt_toks = resp.get("prompt_eval_count", len(prompt.split()))
        comp_toks = resp.get("eval_count", len(text.split()))

        return {
            "text": text,
            "token_ids": list(range(comp_toks)),  # Ollama doesn't return IDs
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "model": resp.get("model", self._model),
            "total_duration_ns": resp.get("total_duration", 0),
            "eval_duration_ns": resp.get("eval_duration", 0),
        }

    def generate_stream(self, prompt: str, max_tokens: int = 256,
                        temperature: float = 0.7) -> Iterator[str]:
        """Stream tokens from Ollama."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    # ── Chat API ─────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict], max_tokens: int = 256,
             temperature: float = 0.7, stream: bool = False) -> dict | Iterator[str]:
        """Chat completion with message history."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if stream:
            return self._chat_stream(payload)

        resp = self._post("/api/chat", payload)
        msg = resp.get("message", {})
        return {
            "text": msg.get("content", ""),
            "role": msg.get("role", "assistant"),
            "prompt_tokens": resp.get("prompt_eval_count", 0),
            "completion_tokens": resp.get("eval_count", 0),
            "token_ids": [],
        }

    def _chat_stream(self, payload) -> Iterator[str]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    msg = chunk.get("message", {})
                    token = msg.get("content", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    # ── Tokenization ─────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> list[int]:
        """Tokenize using Ollama's tokenize endpoint (if available)."""
        try:
            resp = self._post("/api/tokenize", {
                "model": self._model,
                "text": text,
            })
            return resp.get("tokens", [])
        except Exception:
            # Fallback: approximate with word splitting
            return [hash(w) % 32000 for w in text.split()]

    def detokenize(self, token_ids: list[int]) -> str:
        """Detokenize — Ollama may not support this directly."""
        try:
            resp = self._post("/api/detokenize", {
                "model": self._model,
                "tokens": token_ids,
            })
            return resp.get("text", "")
        except Exception:
            return "[detokenize not supported]"

    # ── Embeddings ───────────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding via Ollama embeddings API."""
        try:
            resp = self._post("/api/embed", {
                "model": self._model,
                "input": text,
            })
            embeddings = resp.get("embeddings", [[]])
            return embeddings[0] if embeddings else []
        except Exception:
            # Fallback: try older endpoint
            try:
                resp = self._post("/api/embeddings", {
                    "model": self._model,
                    "prompt": text,
                })
                return resp.get("embedding", [])
            except Exception:
                return []

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def n_layers(self) -> int:
        # Extract from model info if available
        params = self._model_info.get("model_info", {})
        for k, v in params.items():
            if "block_count" in k or "num_hidden_layers" in k:
                return int(v)
        return 32  # Default

    @property
    def vocab_size(self) -> int:
        params = self._model_info.get("model_info", {})
        for k, v in params.items():
            if "vocab_size" in k:
                return int(v)
        return 32000

    @property
    def context_length(self) -> int:
        params = self._model_info.get("model_info", {})
        for k, v in params.items():
            if "context_length" in k:
                return int(v)
        return 4096

    # ── Available Models ─────────────────────────────────────────────────────

    def list_models(self) -> list[str]:
        """List all available models in Ollama."""
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── HTTP Helper ──────────────────────────────────────────────────────────

    def _post(self, endpoint: str, payload: dict,
              stream: bool = False) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        if stream:
            # For pull: consume all lines
            with urllib.request.urlopen(req, timeout=600) as resp:
                last = {}
                for line in resp:
                    if line.strip():
                        try:
                            last = json.loads(line.decode("utf-8"))
                            status = last.get("status", "")
                            if status:
                                print(f"    {status}")
                        except json.JSONDecodeError:
                            pass
                return last

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
