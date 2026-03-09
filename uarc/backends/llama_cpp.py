"""
UARC llama.cpp Backend — Direct GGUF model loading
====================================================
Requires: pip install llama-cpp-python
Loads GGUF models directly, no server needed.
"""
from __future__ import annotations

import os
from typing import Iterator

from uarc.backends.base import ModelBackend


class LlamaCppBackend(ModelBackend):
    """
    Direct GGUF model loading via llama-cpp-python.
    Most efficient path — no HTTP overhead, direct C++ inference.
    """

    def __init__(self, model_path: str,
                 n_ctx: int = 4096,
                 n_gpu_layers: int = -1,
                 n_threads: int = 0,
                 verbose: bool = False):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads or os.cpu_count()
        self._verbose = verbose
        self._llm = None
        self._metadata: dict = {}

    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "  pip install llama-cpp-python\n"
                "For GPU support:\n"
                "  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python")

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._n_threads,
            verbose=self._verbose,
            embedding=True,
        )
        md = self._llm.metadata or {}
        self._metadata = md

    def unload(self) -> None:
        if self._llm:
            del self._llm
            self._llm = None

    def is_available(self) -> bool:
        return self._llm is not None

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: list[str] | None = None) -> dict:
        result = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
        )
        choice = result["choices"][0]
        text = choice["text"]
        usage = result.get("usage", {})
        return {
            "text": text,
            "token_ids": self.tokenize(text),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    def generate_stream(self, prompt: str, max_tokens: int = 256,
                        temperature: float = 0.7) -> Iterator[str]:
        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def tokenize(self, text: str) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"))

    def detokenize(self, token_ids: list[int]) -> str:
        return self._llm.detokenize(token_ids).decode("utf-8", errors="replace")

    def get_embedding(self, text: str) -> list[float]:
        result = self._llm.embed(text)
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result[0]
        return result if isinstance(result, list) else []

    @property
    def model_name(self) -> str:
        return self._metadata.get("general.name",
               os.path.basename(self._model_path))

    @property
    def n_layers(self) -> int:
        return int(self._metadata.get("llama.block_count",
                   self._metadata.get("general.block_count", 32)))

    @property
    def vocab_size(self) -> int:
        return int(self._metadata.get("llama.vocab_size",
                   self._metadata.get("general.vocab_size", 32000)))

    @property
    def context_length(self) -> int:
        return self._n_ctx
