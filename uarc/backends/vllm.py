"""
UARC vLLM Backend — High-throughput PagedAttention inference
============================================================
Requires: pip install vllm
Ideal for production deployments requiring high concurrency.
"""
from __future__ import annotations

import os
from typing import Iterator

from uarc.backends.base import ModelBackend


class VLLMBackend(ModelBackend):
    """
    High-throughput backend using vLLM's LLMEngine.
    """

    def __init__(self, model_id: str,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 trust_remote_code: bool = False):
        self._model_id = model_id
        self._tp_size = tensor_parallel_size
        self._gpu_util = gpu_memory_utilization
        self._trust_remote_code = trust_remote_code
        self._llm = None
        self._q_id = 0

    def load(self) -> None:
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vllm is not installed. Install with:\n"
                "  pip install vllm")

        self._llm = LLM(
            model=self._model_id,
            tensor_parallel_size=self._tp_size,
            gpu_memory_utilization=self._gpu_util,
            trust_remote_code=self._trust_remote_code,
            enforce_eager=True, # better compatibility dynamically
        )

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    def is_available(self) -> bool:
        return self._llm is not None

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: list[str] | None = None) -> dict:
        
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or []
        )
        
        outputs = self._llm.generate([prompt], sampling_params, use_tqdm=False)
        output = outputs[0]
        
        toks = list(output.outputs[0].token_ids)
        text = output.outputs[0].text
        
        return {
            "text": text,
            "token_ids": toks,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(toks),
        }

    def generate_stream(self, prompt: str, max_tokens: int = 256,
                        temperature: float = 0.7) -> Iterator[str]:
        # Note: vLLM's offline LLM engine doesn't stream token-by-token cleanly
        # without the AsyncLLMEngine. We'll simulate stream by generating and chunking.
        res = self.generate(prompt, max_tokens, temperature)
        yield res["text"]

    def tokenize(self, text: str) -> list[int]:
        if not self._llm: return []
        tokenizer = self._llm.llm_engine.tokenizer.tokenizer
        return tokenizer.encode(text)

    def detokenize(self, token_ids: list[int]) -> str:
        if not self._llm: return ""
        tokenizer = self._llm.llm_engine.tokenizer.tokenizer
        return tokenizer.decode(token_ids)

    def get_embedding(self, text: str) -> list[float]:
        # vLLM is generally for generation, not embedding APIs natively in this object
        return []

    @property
    def model_name(self) -> str:
        return self._model_id.split("/")[-1]

    @property
    def n_layers(self) -> int:
        return 32 # Assuming general 7B baseline

    @property
    def vocab_size(self) -> int:
        if not self._llm: return 32000
        return len(self._llm.llm_engine.tokenizer.tokenizer)

    @property
    def context_length(self) -> int:
        if not self._llm: return 4096
        return self._llm.llm_engine.model_config.max_model_len
