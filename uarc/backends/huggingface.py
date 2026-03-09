"""
UARC HuggingFace Backend — Direct inference via transformers
============================================================
Requires: pip install torch transformers accelerate
Loads Safetensors/PyTorch models directly from HF Hub.
"""
from __future__ import annotations

import os
from typing import Iterator

from uarc.backends.base import ModelBackend


class HuggingFaceBackend(ModelBackend):
    """
    Direct model loading via HuggingFace transformers.
    Supports local paths or HF Hub IDs.
    """

    def __init__(self, model_id: str,
                 device: str = "auto",
                 dtype: str = "auto",
                 trust_remote_code: bool = False):
        self._model_id = model_id
        self._device_map = device
        self._dtype = dtype
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers or torch not installed. Install with:\n"
                "  pip install torch transformers accelerate")

        # Handle dtype
        torch_dtype = "auto"
        if self._dtype == "float16": torch_dtype = torch.float16
        elif self._dtype == "bfloat16": torch_dtype = torch.bfloat16
        elif self._dtype == "float32": torch_dtype = torch.float32

        # Load Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=self._trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load Model
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            device_map=self._device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self._trust_remote_code
        )
        self._model.eval()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def is_available(self) -> bool:
        return self._model is not None

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: list[str] | None = None) -> dict:
        import torch
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # Generate config
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            
        # Handle early stopping strings conceptually (HF stop_strings require latest transformers)
        # We'll just generate and trim later for simplicity
        
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
            
        generated_ids = outputs[0][prompt_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Manual stop string check
        if stop:
            for s in stop:
                if s in text:
                    text = text[:text.index(s)]
                    
        return {
            "text": text,
            "token_ids": generated_ids.tolist(),
            "prompt_tokens": prompt_len,
            "completion_tokens": len(generated_ids),
        }

    def generate_stream(self, prompt: str, max_tokens: int = 256,
                        temperature: float = 0.7) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "streamer": streamer,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        
        for new_text in streamer:
            if new_text:
                yield new_text
                
        thread.join()

    def tokenize(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_embedding(self, text: str) -> list[float]:
        # Most generative models don't have a direct embeding pipeline exposed cleanly
        # Use SentenceTransformers if available, else fallback to last hidden state
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embedder'):
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            return self._embedder.encode(text).tolist()
        except ImportError:
            # Very rough fallback
            import torch
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                if hasattr(self._model, "get_input_embeddings"):
                    embeds = self._model.get_input_embeddings()(inputs["input_ids"])
                    return embeds.mean(dim=1).squeeze().tolist()
            return []

    @property
    def model_name(self) -> str:
        return self._model_id.split("/")[-1]

    @property
    def n_layers(self) -> int:
        if not self._model: return 32
        cfg = self._model.config
        return getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 32))

    @property
    def vocab_size(self) -> int:
        if not self._model: return 32000
        return self._model.config.vocab_size

    @property
    def context_length(self) -> int:
        if not self._model: return 4096
        cfg = self._model.config
        return getattr(cfg, "max_position_embeddings", getattr(cfg, "n_positions", 4096))
