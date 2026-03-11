"""
UARC Runtime Core
==================
Orchestrates all 7 modules into a unified inference pipeline.
Supports real backends (Ollama, llama.cpp) and simulated fallback.
"""
from __future__ import annotations

import math
import random
import time
import threading
import uuid
from collections import defaultdict
from typing import Iterator, Optional

from uarc.core.config import UARCConfig
from uarc.core.types import (
    Batch, DeviceType, InferenceRequest, InferenceResponse,
    MemoryTier, Precision, RouteTarget, UARCStats, RoutingDecision,
)
from uarc.routing.tde import TokenDifficultyEstimator
from uarc.routing.nsc import NeuralSemanticCache
from uarc.memory.aivm import AIVirtualMemoryManager, PredictiveLayerLoader
from uarc.scheduling.dpe_acs import DynamicPrecisionEngine, AdaptiveComputeScheduler
from uarc.scheduling.eads import EADSScheduler
from uarc.backends.base import ModelBackend


# ── Simulated Fallback (kept for testing / offline use) ──────────────────────

class SimulatedBackend(ModelBackend):
    """Synthetic backend for testing without real models."""

    def __init__(self, n_layers_cfg=32, vocab_size_cfg=32000):
        self._n_layers = n_layers_cfg
        self._vocab_size = vocab_size_cfg
        self._rng = random.Random(42)

    def load(self): pass
    def unload(self): pass
    def is_available(self) -> bool: return True

    def tokenize(self, text: str) -> list[int]:
        rng = random.Random(hash(text))
        return [rng.randint(10, self._vocab_size - 1) for _ in text.split()]

    def detokenize(self, token_ids: list[int]) -> str:
        words = ["the", "a", "is", "in", "it", "of", "to", "and", "for", "on",
                 "with", "as", "at", "from", "this", "that", "are", "was", "by",
                 "an", "be", "or", "not", "but"]
        rng = random.Random(sum(token_ids[:10]) if token_ids else 0)
        return " ".join(rng.choice(words) for _ in token_ids)

    def generate(self, prompt, max_tokens=256, temperature=0.7,
                 top_p=0.9, stop=None) -> dict:
        tids = [self._rng.randint(10, self._vocab_size - 1) for _ in range(max_tokens)]
        time.sleep(max_tokens * 0.001)
        return {
            "text": self.detokenize(tids), "token_ids": tids,
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": max_tokens,
        }

    def generate_stream(self, prompt, max_tokens=256, temperature=0.7):
        result = self.generate(prompt, max_tokens, temperature)
        for word in result["text"].split():
            yield word + " "
            time.sleep(0.01)

    def get_embedding(self, text: str) -> list[float]:
        rng = random.Random(hash(text))
        vec = [rng.gauss(0, 1) for _ in range(256)]
        norm = math.sqrt(sum(x*x for x in vec))
        return [x/norm for x in vec] if norm > 0 else vec

    @property
    def model_name(self): return "simulated"
    @property
    def n_layers(self): return self._n_layers
    @property
    def vocab_size(self): return self._vocab_size
    @property
    def context_length(self): return 4096


# ── Backend Auto-Detection ───────────────────────────────────────────────────

def _detect_backend(cfg: UARCConfig) -> ModelBackend:
    """Auto-detect the best available backend."""
    backend_type = cfg.backend

    if backend_type == "ollama" or (backend_type == "auto"):
        try:
            from uarc.backends.ollama import OllamaBackend
            ob = OllamaBackend(
                model=cfg.model_name,
                base_url=cfg.ollama_url,
            )
            if ob.is_available():
                print(f"  [BACKEND] Ollama detected -> model: {cfg.model_name}")
                return ob
        except Exception:
            pass

    if backend_type == "hf" or (backend_type == "auto" and cfg.model_name):
        try:
            import transformers
            import torch
            from uarc.backends.huggingface import HuggingFaceBackend
            # Use cfg.model_name if provided, otherwise fallback to a default
            hf_model_name = cfg.model_name if cfg.model_name else "gpt2"
            hb = HuggingFaceBackend(
                model_id=hf_model_name,
                draft_model_id=cfg.draft_model_name
            )
            if hb.is_available():
                print(f"  [BACKEND] HuggingFace detected -> model: {hf_model_name}")
                if cfg.draft_model_name:
                    print(f"  [BACKEND] Speculative Drafting enabled -> draft: {cfg.draft_model_name}")
                return hb
        except ImportError:
            pass
        except Exception as e:
            print(f"  [BACKEND] HuggingFace backend failed to initialize: {e}")
            pass

    if backend_type == "vllm" or (backend_type == "auto" and cfg.model_name):
        try:
            from uarc.backends.vllm import VLLMBackend
            # Use cfg.model_name if provided, otherwise fallback to a default
            vllm_model_name = cfg.model_name if cfg.model_name else "mistralai/Mistral-7B-Instruct-v0.1"
            vb = VLLMBackend(vllm_model_name)
            if vb.is_available():
                print(f"  [BACKEND] vLLM detected -> model: {vllm_model_name}")
                return vb
        except ImportError:
            pass
        except Exception as e:
            print(f"  [BACKEND] vLLM backend failed to initialize: {e}")
            pass

    if backend_type == "llama_cpp" or (backend_type == "auto" and cfg.model_path):
        try:
            from uarc.backends.llama_cpp import LlamaCppBackend
            lb = LlamaCppBackend(
                model_path=cfg.model_path,
                n_ctx=cfg.model.hidden_dim if hasattr(cfg.model, 'hidden_dim') else 4096,
            )
            print(f"  [BACKEND] llama.cpp -> model: {cfg.model_path}")
            return lb
        except Exception:
            pass

    if backend_type not in ("simulated",) and backend_type != "auto":
        print(f"  [BACKEND] WARNING: Requested '{backend_type}' not available, falling back to simulated")

    print("  [BACKEND] Using simulated backend (no real model)")
    return SimulatedBackend(cfg.model.n_layers, cfg.model.vocab_size)


# ── Runtime ──────────────────────────────────────────────────────────────────

class UARCRuntime:
    """
    Unified Adaptive Runtime Core.
    Wires all 7 modules: NSC -> TDE -> DPE -> PLL -> AI-VM -> ACS -> Router.
    Now with REAL model backends.
    """

    def __init__(self, cfg: UARCConfig | None = None):
        self.cfg = cfg or UARCConfig()
        self._stats = UARCStats()
        self._lock = threading.RLock()
        self._running = False

        # Detect and create backend
        self.backend = _detect_backend(self.cfg)

        # M1: TDE
        self.tde = TokenDifficultyEstimator(self.cfg.tde)
        # M2: AI-VM
        self.aivm = AIVirtualMemoryManager(self.cfg.aivm)
        # M3: DPE
        n_layers = self.backend.n_layers
        params_per_layer = self.cfg.model.n_params // max(n_layers, 1)
        profiles = DynamicPrecisionEngine.build_profiles(n_layers, params_per_layer)
        self.dpe = DynamicPrecisionEngine(self.cfg.dpe, profiles)
        # M4: PLL
        self.pll = PredictiveLayerLoader(
            self.cfg.pll, n_layers, [2.0]*n_layers, aivm=self.aivm)
        # M5: ACS
        self.acs = AdaptiveComputeScheduler(self.cfg.acs)
        # M6: NSC
        self.nsc = NeuralSemanticCache(self.cfg.nsc)
        # M7: EADS
        self.eads = EADSScheduler(self.cfg.eads) if self.cfg.enable_eads else None

        # Inject scheduler into backend if supported (Production EADS Engine)
        if self.eads and hasattr(self.backend, "set_eads_scheduler"):
            self.backend.set_eads_scheduler(self.eads)

        # Store n_layers for internals
        self._n_layers = n_layers
        self._eviction_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the runtime: load model + start background threads."""
        self.backend.load()
        self._running = True
        self._load_model_pages()
        self._eviction_thread = threading.Thread(target=self._eviction_loop, daemon=True)
        self._eviction_thread.start()

    def stop(self):
        self._running = False
        self.backend.unload()

    def _load_model_pages(self):
        for i in range(self._n_layers):
            try:
                self.aivm.allocate(
                    f"layer_{i}_weights", 2.0, "weight", layer_id=i,
                    preferred_tier=MemoryTier.NVME,
                    pinned=(i == 0 or i == self._n_layers - 1))
            except MemoryError:
                pass

    def _eviction_loop(self):
        while self._running:
            time.sleep(0.2)
            self.aivm.run_eviction_cycle()

    # ── Main Inference Pipeline ──────────────────────────────────────────────

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Full UARC pipeline: NSC -> TDE -> DPE -> PLL -> Backend -> NSC."""
        t0 = time.perf_counter()
        self._stats.total_requests += 1

        # Tokenize if needed
        if not request.token_ids and request.prompt:
            request.token_ids = self.backend.tokenize(request.prompt)

        # Stage 1: NSC lookup
        if self.cfg.enable_nsc:
            hit = self.nsc.lookup(request.token_ids, request.prompt)
            if hit is not None:
                ct, cx = hit
                self._stats.cache_hits += 1
                lat = (time.perf_counter() - t0) * 1000
                self._stats.total_latency_ms += lat
                return InferenceResponse(
                    request_id=request.request_id, text=cx, token_ids=ct,
                    prompt_tokens=len(request.token_ids),
                    completion_tokens=len(ct),
                    latency_ms=round(lat, 2),
                    tokens_per_second=round(len(ct)/max(lat/1000, 1e-6), 1),
                    route_taken="cache", cache_hit=True,
                    compute_saved_pct=100.0)

        # Stage 2: TDE routing
        if self.cfg.enable_tde:
            decision = self.tde.estimate(request.token_ids)
            request.difficulty_score = decision.estimated_ppl
        else:
            decision = RoutingDecision(
                route=RouteTarget.FULL, estimated_ppl=5.0,
                confidence=1.0, latency_ms=0.0, compute_saved_pct=0.0)

        # Stage 3: DPE precision planning
        plan = None
        if self.cfg.enable_dpe:
            budget = int(self.cfg.dpe.default_budget_gb * 1024**3)
            plan = self.dpe.allocate(budget)
            plan = self.dpe.adapt_for_token(plan, decision.estimated_ppl)

        # Stage 4: PLL prefetch
        if self.cfg.enable_pll:
            self.pll.reset()
            for i in range(min(4, self._n_layers)):
                self.pll.on_layer_start(i)
                time.sleep(0.0001)
                self.pll.on_layer_complete(i)

        # Stage 5: REAL MODEL GENERATION
        n_new = min(request.max_new_tokens, 512)

        # 5a. EADS Cascade Execution (Draft + Verify)
        # 5a. EADS Speculative Execution
        if self.cfg.enable_eads and decision.route == RouteTarget.DRAFT:
            self._stats.draft_routes += 1
            
            # If the backend has a real speculative engine, use it
            # Otherwise carry out the simulated rollback loop
            if hasattr(self.backend, "_spec_engine") and self.backend._spec_engine is not None:
                result = self.backend.generate(
                    request.prompt, max_tokens=n_new,
                    temperature=0.7, top_p=0.9
                )
                ct = result.get("token_ids", [])
                cx = result.get("text", "")
                prompt_toks = result.get("prompt_tokens", len(request.token_ids))
                comp_toks = result.get("completion_tokens", len(ct))
            else:
                # Fallback Simulated speculative loop (for demo/mock purposes)
                current_k = self.eads.seq_states.get(request.request_id, {}).get('current_k', self.cfg.eads.base_k)
                draft_result = self.backend.generate(
                    request.prompt, max_tokens=current_k, 
                    temperature=0.9, top_p=0.95
                )
                draft_toks = draft_result.get("token_ids", [])
                draft_prompt = request.prompt + draft_result.get("text", "")

                verification_result = self.backend.generate(
                    draft_prompt, max_tokens=n_new - len(draft_toks),
                    temperature=0.7, top_p=0.9
                )
                accepted_k = max(1, int(current_k * (1.0 - (decision.estimated_ppl / 10.0))))
                accepted_k = min(current_k, accepted_k)
                
                self.eads.update_and_get_k(request.request_id, current_k, accepted_k, decision.estimated_ppl)
                
                ct = draft_toks[:accepted_k] + verification_result.get("token_ids", [])
                cx = self.backend.detokenize(ct)
                prompt_toks = len(request.token_ids)
                comp_toks = len(ct)

        # 5b. Fallback Standard Execution
        elif decision.route == RouteTarget.PARTIAL:
            self._stats.partial_routes += 1
            result = self.backend.generate(
                request.prompt, max_tokens=n_new,
                temperature=0.7, top_p=0.9)
            ct = result.get("token_ids", [])
            cx = result.get("text", "")
            prompt_toks = result.get("prompt_tokens", len(request.token_ids))
            comp_toks = result.get("completion_tokens", len(ct) or len(cx.split()))
            
        else:
            self._stats.full_routes += 1
            result = self.backend.generate(
                request.prompt, max_tokens=n_new,
                temperature=0.5, top_p=0.85)
            ct = result.get("token_ids", [])
            cx = result.get("text", "")
            prompt_toks = result.get("prompt_tokens", len(request.token_ids))
            comp_toks = result.get("completion_tokens", len(ct) or len(cx.split()))

        # Stage 6: NSC store
        if self.cfg.enable_nsc:
            self.nsc.store(request.token_ids, request.prompt, ct, cx)

        lat = (time.perf_counter() - t0) * 1000
        self._stats.total_tokens_generated += comp_toks
        self._stats.total_latency_ms += lat

        return InferenceResponse(
            request_id=request.request_id, text=cx, token_ids=ct,
            prompt_tokens=prompt_toks, completion_tokens=comp_toks,
            latency_ms=round(lat, 2),
            tokens_per_second=round(comp_toks/max(lat/1000, 1e-6), 1),
            route_taken=decision.route.value, cache_hit=False,
            precision_plan_summary=plan.summary() if plan else {},
            compute_saved_pct=decision.compute_saved_pct)

    def infer_stream(self, request: InferenceRequest) -> Iterator[str]:
        """Stream tokens from real backend."""
        if not request.token_ids and request.prompt:
            request.token_ids = self.backend.tokenize(request.prompt)

        # NSC check
        if self.cfg.enable_nsc:
            hit = self.nsc.lookup(request.token_ids, request.prompt)
            if hit is not None:
                self._stats.cache_hits += 1
                for word in hit[1].split():
                    yield word + " "
                return

        # Real streaming from backend
        self._stats.total_requests += 1
        n_new = min(request.max_new_tokens, 512)
        collected = []
        for token in self.backend.generate_stream(
                request.prompt, max_tokens=n_new):
            yield token
            collected.append(token)

        # Store in NSC
        full_text = "".join(collected)
        if self.cfg.enable_nsc and full_text:
            self.nsc.store(request.token_ids, request.prompt, [], full_text)

    def infer_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        for req in requests:
            self.acs.submit(req)
        batch = self.acs.form_batch()
        if batch is None:
            return []
        return [self.infer(r) for r in batch.requests]

    def status(self) -> dict:
        return {
            "runtime": {
                "running": self._running,
                "version": "0.2.0",
                "backend": self.backend.model_name,
                "backend_available": self.backend.is_available(),
            },
            "config": self.cfg.to_dict(),
            "memory": self.aivm.status(),
            "modules": {
                "tde": self.tde.stats(),
                "nsc": self.nsc.stats(),
                "dpe": self.dpe.stats(),
                "pll": self.pll.stats_report(),
                "acs": self.acs.stats_report(),
                "eads": self.eads.stats() if self.eads else {"enabled": False},
            },
            "performance": self._stats.to_dict(),
        }
