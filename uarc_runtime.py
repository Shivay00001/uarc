"""
UARC Runtime Core
==================
Orchestrates all 7 modules into a single inference pipeline.
Simulated model backend — swap in llama.cpp / vLLM for real weights.
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
    MemoryTier, Precision, RouteTarget, UARCStats,
)
from uarc.routing.tde  import TokenDifficultyEstimator
from uarc.routing.nsc  import NeuralSemanticCache
from uarc.memory.aivm  import AIVirtualMemoryManager, PredictiveLayerLoader
from uarc.scheduling.dpe_acs import DynamicPrecisionEngine, AdaptiveComputeScheduler


class SimulatedModel:
    """
    Synthetic model backend: produces plausible token streams
    without requiring real weights. Every token is a small vocabulary
    sample; perplexity is simulated based on context entropy.
    """

    def __init__(self, n_layers: int, vocab_size: int, hidden: int):
        self.n_layers   = n_layers
        self.vocab_size = vocab_size
        self.hidden     = hidden
        self._rng       = random.Random(42)

    def _sim_ppl(self, token_ids: list[int]) -> float:
        """Simulate perplexity based on token diversity."""
        if not token_ids:
            return 5.0
        window = token_ids[-64:]
        uniq   = len(set(window)) / max(len(window), 1)
        return 1.5 + uniq * 15.0 + self._rng.gauss(0, 0.5)

    def forward_draft(self, token_ids: list[int],
                      n_new: int) -> list[int]:
        """Draft model: 1B param equivalent, fast."""
        time.sleep(n_new * 0.001)   # 1ms/token
        return [self._rng.randint(100, self.vocab_size - 1)
                for _ in range(n_new)]

    def forward_partial(self, token_ids: list[int],
                        n_new: int,
                        skip_top_pct: float = 0.30) -> list[int]:
        """Partial model: skip top 30% of layers."""
        active = int(self.n_layers * (1 - skip_top_pct))
        time.sleep(n_new * 0.002 * active / self.n_layers)
        return [self._rng.randint(50, self.vocab_size - 1)
                for _ in range(n_new)]

    def forward_full(self, token_ids: list[int],
                     n_new: int,
                     precision: Precision = Precision.FP16) -> list[int]:
        """Full model forward pass."""
        scale = {
            Precision.INT4:  0.8,
            Precision.INT8:  0.9,
            Precision.FP16:  1.0,
            Precision.FP32:  1.2,
        }.get(precision, 1.0)
        time.sleep(n_new * 0.003 * scale)
        return [self._rng.randint(10, self.vocab_size - 1)
                for _ in range(n_new)]

    def detokenize(self, token_ids: list[int]) -> str:
        """Simulate detokenization."""
        words = [
            "the", "a", "is", "in", "it", "of", "to", "and",
            "for", "on", "with", "as", "at", "from", "this",
            "that", "are", "was", "by", "an", "be", "or",
        ]
        result = []
        rng = random.Random(sum(token_ids[:10]) if token_ids else 0)
        for _ in token_ids:
            result.append(rng.choice(words))
        return " ".join(result)

    def tokenize(self, text: str) -> list[int]:
        """Simulate tokenisation."""
        rng = random.Random(hash(text))
        return [rng.randint(10, self.vocab_size - 1)
                for _ in text.split()]


class UARCRuntime:
    """
    Unified Adaptive Runtime Core.

    Wires all 7 modules and exposes a simple inference API:
        runtime = UARCRuntime(cfg)
        runtime.start()
        response = runtime.infer(request)
        runtime.stop()
    """

    def __init__(self, cfg: Optional[UARCConfig] = None):
        self.cfg = cfg or UARCConfig()
        self._stats = UARCStats()
        self._lock  = threading.RLock()
        self._running = False

        # ── Module 1: TDE ─────────────────────────────────────────────────────
        self.tde = TokenDifficultyEstimator(self.cfg.tde)

        # ── Module 2: AI-VM ───────────────────────────────────────────────────
        self.aivm = AIVirtualMemoryManager(self.cfg.aivm)

        # ── Module 3: DPE ─────────────────────────────────────────────────────
        profiles = DynamicPrecisionEngine.build_profiles(
            self.cfg.model.n_layers,
            params_per_layer=self.cfg.model.n_params // max(self.cfg.model.n_layers, 1),
        )
        self.dpe = DynamicPrecisionEngine(self.cfg.dpe, profiles)

        # ── Module 4: PLL ─────────────────────────────────────────────────────
        layer_sizes = [2.0] * self.cfg.model.n_layers  # 2MB per layer
        self.pll = PredictiveLayerLoader(
            self.cfg.pll,
            self.cfg.model.n_layers,
            layer_sizes,
            aivm=self.aivm,
        )

        # ── Module 5: ACS ─────────────────────────────────────────────────────
        self.acs = AdaptiveComputeScheduler(self.cfg.acs)

        # ── Module 6: NSC ─────────────────────────────────────────────────────
        self.nsc = NeuralSemanticCache(self.cfg.nsc)

        # ── Simulated Model ───────────────────────────────────────────────────
        self.model = SimulatedModel(
            n_layers=self.cfg.model.n_layers,
            vocab_size=self.cfg.model.vocab_size,
            hidden=self.cfg.model.hidden_dim,
        )

        # ── Background threads ────────────────────────────────────────────────
        self._eviction_thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise memory and start background workers."""
        self._running = True
        self._load_model_pages()
        self._eviction_thread = threading.Thread(
            target=self._eviction_loop, daemon=True)
        self._eviction_thread.start()

    def stop(self) -> None:
        self._running = False

    def _load_model_pages(self) -> None:
        """Pre-allocate all model layer pages on NVMe (Tier 2)."""
        for i in range(self.cfg.model.n_layers):
            pid = f"layer_{i}_weights"
            try:
                self.aivm.allocate(
                    pid, size_mb=2.0, data_type="weight",
                    layer_id=i, preferred_tier=MemoryTier.NVME,
                    pinned=(i == 0 or i == self.cfg.model.n_layers - 1),
                )
            except MemoryError:
                pass   # NVMe full — skip in demo

    def _eviction_loop(self) -> None:
        """Background eviction cycle every 200ms."""
        while self._running:
            time.sleep(0.2)
            self.aivm.run_eviction_cycle()

    # ── Primary Inference API ─────────────────────────────────────────────────

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Full UARC inference pipeline:
          NSC lookup → TDE estimate → ACS schedule → DPE plan →
          PLL prefetch → Model execute → NSC store → Response
        """
        t_start = time.perf_counter()
        self._stats.total_requests += 1

        # Tokenise if needed
        if not request.token_ids and request.prompt:
            request.token_ids = self.model.tokenize(request.prompt)

        # ── Stage 1: NSC Lookup ───────────────────────────────────────────────
        if self.cfg.enable_nsc:
            hit = self.nsc.lookup(request.token_ids, request.prompt)
            if hit is not None:
                comp_toks, comp_text = hit
                self._stats.cache_hits += 1
                latency = (time.perf_counter() - t_start) * 1000
                self._stats.total_latency_ms += latency
                return InferenceResponse(
                    request_id=request.request_id,
                    text=comp_text,
                    token_ids=comp_toks,
                    prompt_tokens=len(request.token_ids),
                    completion_tokens=len(comp_toks),
                    latency_ms=round(latency, 2),
                    tokens_per_second=round(len(comp_toks) / max(latency/1000, 1e-6), 1),
                    route_taken="cache",
                    cache_hit=True,
                    compute_saved_pct=100.0,
                )

        # ── Stage 2: TDE Routing ──────────────────────────────────────────────
        if self.cfg.enable_tde:
            decision = self.tde.estimate(request.token_ids)
            request.difficulty_score = decision.estimated_ppl
        else:
            from uarc.core.types import RoutingDecision
            decision = RoutingDecision(
                route=RouteTarget.FULL,
                estimated_ppl=5.0, confidence=1.0,
                latency_ms=0.0, compute_saved_pct=0.0,
            )

        # ── Stage 3: DPE Precision Plan ───────────────────────────────────────
        if self.cfg.enable_dpe:
            budget = int(self.cfg.dpe.default_budget_gb * 1024**3)
            plan   = self.dpe.allocate(budget)
            plan   = self.dpe.adapt_for_token(plan, decision.estimated_ppl)
        else:
            plan = None

        # Determine precision for model execution
        precision = Precision.FP16
        if plan and plan.assignment:
            mid = len(plan.assignment) // 2
            precision = plan.assignment[mid]

        # ── Stage 4: PLL Prefetch (first N layers) ────────────────────────────
        if self.cfg.enable_pll:
            self.pll.reset()
            # Simulate prefetch for first few layers
            for i in range(min(4, self.cfg.model.n_layers)):
                self.pll.on_layer_start(i)
                time.sleep(0.0001)
                self.pll.on_layer_complete(i)

        # ── Stage 5: Model Execution ──────────────────────────────────────────
        n_new = min(request.max_new_tokens, 64)   # cap for simulation speed

        if decision.route == RouteTarget.DRAFT:
            self._stats.draft_routes += 1
            comp_toks = self.model.forward_draft(request.token_ids, n_new)
        elif decision.route == RouteTarget.PARTIAL:
            self._stats.partial_routes += 1
            comp_toks = self.model.forward_partial(request.token_ids, n_new)
        else:
            self._stats.full_routes += 1
            comp_toks = self.model.forward_full(
                request.token_ids, n_new, precision)

        comp_text = self.model.detokenize(comp_toks)

        # ── Stage 6: NSC Store ────────────────────────────────────────────────
        if self.cfg.enable_nsc:
            self.nsc.store(request.token_ids, request.prompt,
                           comp_toks, comp_text)

        # ── Finalise ──────────────────────────────────────────────────────────
        latency = (time.perf_counter() - t_start) * 1000
        self._stats.total_tokens_generated += len(comp_toks)
        self._stats.total_latency_ms += latency

        return InferenceResponse(
            request_id=request.request_id,
            text=comp_text,
            token_ids=comp_toks,
            prompt_tokens=len(request.token_ids),
            completion_tokens=len(comp_toks),
            latency_ms=round(latency, 2),
            tokens_per_second=round(len(comp_toks) / max(latency/1000, 1e-6), 1),
            route_taken=decision.route.value,
            cache_hit=False,
            precision_plan_summary=plan.summary() if plan else {},
            compute_saved_pct=decision.compute_saved_pct,
        )

    def infer_stream(self, request: InferenceRequest) -> Iterator[str]:
        """Streaming inference — yields tokens as they are produced."""
        resp = self.infer(request)
        words = resp.text.split()
        for word in words:
            yield word + " "
            time.sleep(0.005)

    # ── Batch API ─────────────────────────────────────────────────────────────

    def infer_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        """Submit a batch and execute all requests."""
        for req in requests:
            self.acs.submit(req)
        batch = self.acs.form_batch()
        if batch is None:
            return []
        return [self.infer(r) for r in batch.requests]

    # ── Status & Stats ────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "runtime":  {"running": self._running, "version": "0.1.0"},
            "config":   self.cfg.to_dict(),
            "memory":   self.aivm.status(),
            "modules": {
                "tde":  self.tde.stats(),
                "nsc":  self.nsc.stats(),
                "dpe":  self.dpe.stats(),
                "pll":  self.pll.stats_report(),
                "acs":  self.acs.stats_report(),
            },
            "performance": self._stats.to_dict(),
        }
