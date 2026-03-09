"""
UARC Modules 3+5: Dynamic Precision Engine (DPE) + Adaptive Compute Scheduler (ACS)
=====================================================================================
DPE: Per-layer bit-width allocation via greedy knapsack (INT4→INT8→FP16→FP32).
ACS: Priority batch formation + roofline-based CPU/GPU routing.
"""
from __future__ import annotations

import hashlib
import heapq
import math
import random
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from uarc.core.config import DPEConfig, ACSConfig
from uarc.core.types import (
    Precision, DeviceType, InferenceRequest, Batch, RequestPriority,
)

# ── DPE Constants ────────────────────────────────────────────────────────────
PRECISION_ORDER = [Precision.INT4, Precision.INT8, Precision.FP16, Precision.FP32]
BITS = {Precision.INT4: 4, Precision.INT8: 8, Precision.FP16: 16, Precision.FP32: 32}

@dataclass
class LayerProfile:
    layer_id: int; n_params: int; sensitivity: dict

@dataclass
class PrecisionPlan:
    assignment: list; total_bytes: int; estimated_ppl_delta: float; memory_budget_bytes: int

    @property
    def avg_bits(self) -> float:
        return sum(BITS[p] for p in self.assignment) / max(len(self.assignment), 1)

    def summary(self) -> dict:
        counts = defaultdict(int)
        for p in self.assignment: counts[p.name] += 1
        return {
            "precision_counts": dict(counts), "avg_bits_per_param": round(self.avg_bits, 2),
            "total_MB": round(self.total_bytes / 1e6, 1),
            "estimated_ppl_delta": round(self.estimated_ppl_delta, 4),
            "budget_utilisation_pct": round(
                100 * self.total_bytes / max(self.memory_budget_bytes, 1), 1),
        }


class DynamicPrecisionEngine:
    """Runtime precision allocator using greedy knapsack."""

    def __init__(self, cfg: DPEConfig | None = None,
                 layer_profiles: list[LayerProfile] | None = None):
        self.cfg = cfg or DPEConfig()
        profiles = layer_profiles or []
        self.profiles = {lp.layer_id: lp for lp in profiles}
        self.n_layers = len(profiles)
        self._cache: dict[int, PrecisionPlan] = {}
        self._n_allocations = 0; self._n_adaptations = 0

    @staticmethod
    def build_profiles(n_layers: int, params_per_layer: int = 100_000_000,
                       rng: random.Random = None) -> list[LayerProfile]:
        """Build simulated layer profiles with U-shaped sensitivity."""
        rng = rng or random.Random(0)
        profiles = []
        for i in range(n_layers):
            edge = 1.0 - abs(i - n_layers / 2) / (n_layers / 2)
            edge = 1.0 - edge
            sens = {
                Precision.INT4: round(edge * rng.uniform(0.3, 1.2) + 0.1, 4),
                Precision.INT8: round(edge * rng.uniform(0.05, 0.3) + 0.02, 4),
                Precision.FP16: 0.0, Precision.FP32: 0.0,
            }
            profiles.append(LayerProfile(i, params_per_layer, sens))
        return profiles

    def allocate(self, memory_budget_bytes: int) -> PrecisionPlan:
        """Greedy knapsack precision allocation. O(L²) worst case."""
        bucket = memory_budget_bytes // (1024 * 1024)
        if bucket in self._cache: return self._cache[bucket]
        asgn = {lid: Precision.INT4 for lid in self.profiles}

        def total_bytes(a):
            return sum(self.profiles[lid].n_params * BITS[p] // 8 for lid, p in a.items())

        def ppl_delta(a):
            return sum(self.profiles[lid].sensitivity.get(p, 0.0) for lid, p in a.items())

        cur_bytes = total_bytes(asgn)
        for _ in range(self.n_layers * len(PRECISION_ORDER)):
            best_gpb, best_lid, best_prec = -math.inf, None, None
            for lid, cur_p in asgn.items():
                idx = PRECISION_ORDER.index(cur_p)
                if idx + 1 >= len(PRECISION_ORDER): continue
                nxt = PRECISION_ORDER[idx + 1]
                db = self.profiles[lid].n_params * (BITS[nxt] - BITS[cur_p]) // 8
                if cur_bytes + db > memory_budget_bytes: continue
                gain = self.profiles[lid].sensitivity.get(cur_p, 0.0) - self.profiles[lid].sensitivity.get(nxt, 0.0)
                gpb = gain / max(db, 1)
                if gpb > best_gpb: best_gpb, best_lid, best_prec = gpb, lid, nxt
            if best_lid is None: break
            db = self.profiles[best_lid].n_params * (BITS[best_prec] - BITS[asgn[best_lid]]) // 8
            asgn[best_lid] = best_prec; cur_bytes += db

        plan = PrecisionPlan(
            assignment=[asgn[i] for i in sorted(asgn)], total_bytes=cur_bytes,
            estimated_ppl_delta=ppl_delta(asgn), memory_budget_bytes=memory_budget_bytes)
        self._cache[bucket] = plan; self._n_allocations += 1; return plan

    def adapt_for_token(self, base_plan: PrecisionPlan,
                        difficulty: float = 5.0,
                        critical_layers: list[int] | None = None) -> PrecisionPlan:
        """Per-token adaptation: upgrade critical layers for hard tokens."""
        if critical_layers is None:
            nc = max(1, self.n_layers // 10)
            critical_layers = list(range(nc)) + list(range(self.n_layers - nc, self.n_layers))
        asgn = list(base_plan.assignment)
        if difficulty > 8.0:
            for lid in critical_layers:
                if lid < len(asgn) and BITS[asgn[lid]] < 16:
                    asgn[lid] = Precision.FP16
        nb = sum(self.profiles.get(i, LayerProfile(i, 1, {})).n_params * BITS[asgn[i]] // 8
                 for i in range(len(asgn)))
        self._n_adaptations += 1
        return PrecisionPlan(asgn, nb, base_plan.estimated_ppl_delta, base_plan.memory_budget_bytes)

    def stats(self) -> dict:
        return {"allocations": self._n_allocations, "adaptations": self._n_adaptations,
                "cached_plans": len(self._cache)}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5: Adaptive Compute Scheduler (ACS)
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveComputeScheduler:
    """Priority-weighted, difficulty-aware batch scheduler with roofline routing."""

    def __init__(self, cfg: ACSConfig | None = None):
        self.cfg = cfg or ACSConfig()
        self.max_batch = self.cfg.max_batch_size
        self._queues = {p: [] for p in RequestPriority}
        self._lock = threading.Lock()
        self._batch_counter = 0
        self._submit_counter = 0
        self.stats = defaultdict(int)
        self.gpu_util = 0.0; self.cpu_util = 0.0

    def submit(self, req: InferenceRequest):
        """Submit a request. Thread-safe."""
        req.prefix_hash = self._prefix_hash(req.token_ids if req.token_ids else
                                             self._text_tokens(req.prompt))
        with self._lock:
            self._submit_counter += 1
            heapq.heappush(self._queues[req.priority],
                           (-req.priority_score(), self._submit_counter, req))
        self.stats["submitted"] += 1

    def _prefix_hash(self, tids: list, n=64) -> str:
        return hashlib.md5(str(tids[:n]).encode()).hexdigest()[:8]

    def _text_tokens(self, text: str) -> list[int]:
        rng = random.Random(hash(text))
        return [rng.randint(10, 32000) for _ in text.split()]

    def form_batch(self) -> Optional[Batch]:
        """Form next batch using priority + prefix-sharing optimisation."""
        with self._lock:
            cands = []
            for prio in [RequestPriority.REALTIME, RequestPriority.STANDARD, RequestPriority.BATCH]:
                while len(cands) < self.max_batch and self._queues[prio]:
                    _, _, req = heapq.heappop(self._queues[prio]); cands.append(req)
            if not cands: return None
            cands = self._reorder_prefix(cands)
            self._batch_counter += 1
            batch = Batch(requests=cands, batch_id=f"batch_{self._batch_counter:06d}")
            self.stats["batches_formed"] += 1
            self.stats["total_batched"] += len(cands)
            return batch

    def _reorder_prefix(self, reqs):
        groups = defaultdict(list)
        for r in reqs: groups[r.prefix_hash].append(r)
        ordered = []
        for _, grp in sorted(groups.items(), key=lambda x: -len(x[1])):
            ordered.extend(grp)
        self.stats["kv_sharing_pairs"] += sum(len(g) - 1 for g in groups.values())
        return ordered

    def route(self, batch_size: int = 1, n_params: int = 100_000_000) -> DeviceType:
        """Roofline model: intensity > 4 FLOP/byte → GPU."""
        intensity = (2 * batch_size) / 2  # FP16
        if self.gpu_util > 0.90: return DeviceType.CPU
        return DeviceType.GPU if intensity >= 4.0 else DeviceType.CPU

    def stats_report(self) -> dict:
        s = self.stats; nb = max(s["batches_formed"], 1)
        return {
            "submitted": s["submitted"], "batches_formed": s["batches_formed"],
            "avg_batch_size": round(s["total_batched"] / nb, 2) if s["batches_formed"] else 0,
            "kv_sharing_pairs": s["kv_sharing_pairs"],
            "kv_sharing_pct": round(100 * s["kv_sharing_pairs"] / max(s["submitted"], 1), 1),
        }
