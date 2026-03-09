"""
UARC Model Backend — Abstract Interface
=========================================
All backends must implement this interface.
"""
from __future__ import annotations

import abc
from typing import Iterator


class ModelBackend(abc.ABC):
    """
    Abstract base for model backends.
    Implement this to plug in any LLM: Ollama, llama.cpp, HuggingFace, vLLM, etc.
    """

    @abc.abstractmethod
    def load(self) -> None:
        """Load/initialize the model."""
        ...

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        ...

    @abc.abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        ...

    @abc.abstractmethod
    def detokenize(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        ...

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: list[str] | None = None) -> dict:
        """
        Generate completion. Returns dict:
            {
                "text": str,
                "token_ids": list[int],
                "prompt_tokens": int,
                "completion_tokens": int,
            }
        """
        ...

    @abc.abstractmethod
    def generate_stream(self, prompt: str, max_tokens: int = 256,
                        temperature: float = 0.7) -> Iterator[str]:
        """Stream tokens one by one."""
        ...

    @abc.abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Get text embedding vector (for NSC)."""
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    @abc.abstractmethod
    def n_layers(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def context_length(self) -> int:
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if backend is ready (model loaded, server reachable, etc.)."""
        ...
