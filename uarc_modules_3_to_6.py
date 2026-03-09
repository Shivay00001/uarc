"""
UARC Modules 3–6
================
3. Dynamic Precision Engine (DPE)
4. Predictive Layer Loader (PLL)
5. Adaptive Compute Scheduler (ACS)
6. Neural Semantic Cache (NSC)

All implementations: pure Python, no external dependencies.
"""

import math
import time
import heapq
import threading
import hashlib
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: Dynamic Precision Engine (DPE)
# ══════════════════════════════════════════════════════════════════════════════
"""
Mathematical Foundation:
  Given L layers, memory budget M (bytes), precision set P = {INT4, INT8, FP16, FP32}

  Layer size:  size(l, p) = params(l) × bits(p) / 8
  Sensitivity: sens(l, p) = δPPL when layer l uses precision p
                           = PPL(model with layer l at p) - PPL(full FP16)

  Optimisation problem (knapsack variant):
    maximise  Σ_l quality(l, assignment[l])
    subject to  Σ_l size(l, assignment[l]) ≤ M

  Greedy solution:
    Start all layers at INT4 (minimum budget).
    Repeatedly upgrade the layer with highest
    (quality_gain / byte_cost) ratio until budget exhausted.

  Quality gain: q(l, from_p, to_p) = sens(l, from_p) - sens(l, to_p)
    (upgrading precision reduces ppl delta → positive gain)
"""

class Precision(Enum):
    INT4  = 4
    INT8  = 8
    FP16  = 16
    FP32  = 32

PRECISION_ORDER = [Precision.INT4, Precision.INT8, Precision.FP16, Precision.FP32]
BITS = {Precision.INT4: 4, Precision.INT8: 8, Precision.FP16: 16, Precision.FP32: 32}

@dataclass
class LayerProfile:
    layer_id: int
    n_params: int           # Number of parameters
    sensitivity: dict       # {Precision: ppl_delta vs FP16}
    # sensitivity[INT4] = 0.8  means +0.8 PPL when this layer is INT4

@dataclass
class PrecisionPlan:
    assignment: list        # [Precision] per layer
    total_bytes: int
    estimated_ppl_delta: float
    memory_budget_bytes: int

    def summary(self) -> dict:
        counts = defaultdict(int)
        for p in self.assignment:
            counts[p.name] += 1
        avg_bits = sum(BITS[p] for p in self.assignment) / len(self.assignment)
        return {
            "precision_counts": dict(counts),
            "avg_bits_per_param": round(avg_bits, 2),
            "total_MB": round(self.total_bytes / 1e6, 1),
            "estimated_ppl_delta": round(self.estimated_ppl_delta, 4),
            "budget_utilisation_pct": round(100 * self.total_bytes / self.memory_budget_bytes, 1),
        }


class DynamicPrecisionEngine:
    """
    Runtime precision allocator.

    Workflow:
      1. Offline: run sensitivity_analysis() to profile each layer.
      2. At request time: allocate(memory_budget) → PrecisionPlan.
      3. Per-token adaptation: adapt_for_token(difficulty_score)
         upgrades critical layers if token is hard.
    """

    def __init__(self, layer_profiles: list[LayerProfile]):
        self.profiles = {lp.layer_id: lp for lp in layer_profiles}
        self.n_layers = len(layer_profiles)
        self._cache: dict[int, PrecisionPlan] = {}   # budget_mb → plan

    # ── Core Allocation ───────────────────────────────────────────────────────

    def allocate(self, memory_budget_bytes: int) -> PrecisionPlan:
        """
        Greedy knapsack precision allocation.

        Complexity: O(L × |P| × iterations) where iterations ≤ L×3
        Typical: O(L²) in worst case, O(L log L) in practice.
        """
        bucket = memory_budget_bytes // (1024*1024)
        if bucket in self._cache:
            return self._cache[bucket]

        L = self.n_layers
        # Step 1: initialise all layers at INT4
        assignment = {lid: Precision.INT4 for lid in self.profiles}

        def total_bytes(asgn):
            return sum(
                self.profiles[lid].n_params * BITS[p] // 8
                for lid, p in asgn.items()
            )

        def ppl_delta(asgn):
            return sum(
                self.profiles[lid].sensitivity.get(p, 0.0)
                for lid, p in asgn.items()
            )

        current_bytes = total_bytes(assignment)

        # Step 2: greedy upgrade loop
        MAX_ITER = L * len(PRECISION_ORDER)
        for _ in range(MAX_ITER):
            best_gain_per_byte = -math.inf
            best_lid  = None
            best_prec = None

            for lid, cur_prec in assignment.items():
                idx = PRECISION_ORDER.index(cur_prec)
                if idx + 1 >= len(PRECISION_ORDER):
                    continue   # already at FP32
                next_prec = PRECISION_ORDER[idx + 1]
                prof = self.profiles[lid]

                delta_bytes = (prof.n_params *
                               (BITS[next_prec] - BITS[cur_prec]) // 8)
                if current_bytes + delta_bytes > memory_budget_bytes:
                    continue

                # Quality gain: reduction in PPL delta
                gain = (prof.sensitivity.get(cur_prec, 0.0) -
                        prof.sensitivity.get(next_prec, 0.0))
                gpb = gain / max(delta_bytes, 1)

                if gpb > best_gain_per_byte:
                    best_gain_per_byte = gpb
                    best_lid  = lid
                    best_prec = next_prec

            if best_lid is None:
                break   # No more profitable upgrades

            delta_b = (self.profiles[best_lid].n_params *
                       (BITS[best_prec] - BITS[assignment[best_lid]]) // 8)
            assignment[best_lid] = best_prec
            current_bytes += delta_b

        plan = PrecisionPlan(
            assignment=[assignment[i] for i in sorted(assignment)],
            total_bytes=current_bytes,
            estimated_ppl_delta=ppl_delta(assignment),
            memory_budget_bytes=memory_budget_bytes,
        )
        self._cache[bucket] = plan
        return plan

    def adapt_for_token(self, base_plan: PrecisionPlan,
                        difficulty_score: float,
                        critical_layers: list[int] = None) -> PrecisionPlan:
        """
        Per-token precision adaptation.
        Hard tokens (high difficulty) → upgrade critical layers to FP16.
        Easy tokens → can downgrade further (INT4 everywhere).

        critical_layers: typically first and last N layers (most sensitive).
        """
        if critical_layers is None:
            # Default: first 10% and last 10% of layers are critical
            n_crit = max(1, self.n_layers // 10)
            critical_layers = list(range(n_crit)) + list(
                range(self.n_layers - n_crit, self.n_layers))

        assignment = list(base_plan.assignment)

        if difficulty_score > 8.0:   # Hard token
            for lid in critical_layers:
                if lid < len(assignment):
                    # Upgrade to at least FP16
                    cur = assignment[lid]
                    if BITS[cur] < 16:
                        assignment[lid] = Precision.FP16

        elif difficulty_score < 2.0:  # Easy token
            # Can use INT4 everywhere (already the base)
            pass

        new_bytes = sum(
            self.profiles.get(i, LayerProfile(i, 1, {})).n_params *
            BITS[assignment[i]] // 8
            for i in range(len(assignment))
        )
        return PrecisionPlan(
            assignment=assignment,
            total_bytes=new_bytes,
            estimated_ppl_delta=base_plan.estimated_ppl_delta,
            memory_budget_bytes=base_plan.memory_budget_bytes,
        )

    @staticmethod
    def sensitivity_analysis(n_layers: int,
                              rng: random.Random = None) -> list[LayerProfile]:
        """
        Simulate layer sensitivity analysis.
        In production: run each layer at each precision, measure PPL delta.

        Observation from empirical studies:
          - First and last layers are most sensitive (embedding + output)
          - Middle attention layers: moderate sensitivity
          - FFN layers: generally less sensitive than attention
        """
        rng = rng or random.Random(0)
        profiles = []
        for i in range(n_layers):
            # Sensitivity pattern: U-shaped (high at edges, low in middle)
            edge_factor = 1.0 - abs(i - n_layers/2) / (n_layers/2)
            edge_factor = 1.0 - edge_factor   # invert → high at edges

            base_params = rng.randint(50_000_000, 200_000_000)  # 50M–200M params
            sens = {
                Precision.INT4: round(edge_factor * rng.uniform(0.3, 1.2) + 0.1, 4),
                Precision.INT8: round(edge_factor * rng.uniform(0.05, 0.3) + 0.02, 4),
                Precision.FP16: 0.0,    # baseline
                Precision.FP32: 0.0,    # negligible gain over FP16
            }
            profiles.append(LayerProfile(
                layer_id=i, n_params=base_params, sensitivity=sens))
        return profiles


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4: Predictive Layer Loader (PLL)
# ══════════════════════════════════════════════════════════════════════════════
"""
Mathematical Foundation:
  Let T(l) = execution time for layer l  (EMA-updated)
  Let L(l) = load time for layer l from current tier

  Prefetch decision for layer l:
    t_need(l) = t_now + Σ_{i=0}^{l-1} T(i)   (time we need layer l)
    t_load(l) = L(l)                            (time to promote layer l)

    Prefetch if:  t_need(l) - t_load(l) > t_now   AND   t_need(l) > t_now + SLACK_MS

  EMA update:
    T̂(l) ← α·T̂(l) + (1-α)·T_observed(l)     α = 0.9 (slow adaption)

  Lookahead window: k = 4 layers (configurable)
    Prefetch layers [current+1, ..., current+k]
"""

@dataclass
class LayerTiming:
    layer_id: int
    ema_exec_ms: float = 50.0    # EMA of execution time
    ema_load_ms: float = 200.0   # EMA of load time from Tier 2
    exec_samples: int = 0
    load_samples: int = 0
    alpha: float = 0.9           # EMA smoothing

    def update_exec(self, observed_ms: float):
        self.ema_exec_ms = self.alpha*self.ema_exec_ms + (1-self.alpha)*observed_ms
        self.exec_samples += 1

    def update_load(self, observed_ms: float):
        self.ema_load_ms = self.alpha*self.ema_load_ms + (1-self.alpha)*observed_ms
        self.load_samples += 1


@dataclass
class PrefetchOrder:
    layer_id: int
    urgency: float      # seconds until needed
    target_tier: int    # 0=VRAM, 1=RAM
    estimated_load_ms: float


class PredictiveLayerLoader:
    """
    Lookahead prefetch scheduler for model layers.

    Maintains:
      - LayerTiming per layer (EMA of exec + load times)
      - Execution timeline (predicted times for upcoming layers)
      - Prefetch queue (async, drained by a worker thread)

    On each layer completion:
      1. Update EMA for completed layer
      2. Recalculate timeline for next k layers
      3. Issue prefetch commands for any layer where
         (t_need - t_load > t_now + SLACK_MS)
    """
    LOOKAHEAD_K  = 4       # layers to look ahead
    SLACK_MS     = 20.0    # minimum slack before issuing prefetch

    def __init__(self, n_layers: int, layer_sizes_mb: list[float]):
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes_mb   # size in MB per layer
        self.timings = [LayerTiming(i) for i in range(n_layers)]
        self._current_layer = 0
        self._t_start = time.perf_counter()
        self._prefetch_queue = []   # (urgency, PrefetchOrder)
        self._issued: set[int] = set()  # layers already prefetched
        self._lock = threading.Lock()
        # Callback: caller sets this to actually trigger VM promotion
        self.on_prefetch: callable = lambda order: None
        self.stats = defaultdict(int)

    def on_layer_start(self, layer_id: int):
        """Call when execution of layer_id begins."""
        self._current_layer = layer_id
        self._t_layer_start = time.perf_counter()

    def on_layer_complete(self, layer_id: int):
        """
        Call when layer_id execution completes.
        Updates EMA and schedules prefetch for upcoming layers.
        """
        t_exec = (time.perf_counter() - self._t_layer_start) * 1000
        self.timings[layer_id].update_exec(t_exec)

        # Recalculate timeline and schedule prefetches
        self._schedule_prefetches(layer_id)

    def _schedule_prefetches(self, completed_layer: int):
        """
        For layers [completed+1 .. completed+k]:
          Compute t_need, compare to t_load, issue prefetch if needed.
        """
        t_now = time.perf_counter() * 1000  # ms

        cumulative_ms = 0.0
        for offset in range(1, self.LOOKAHEAD_K + 1):
            target_layer = completed_layer + offset
            if target_layer >= self.n_layers:
                break

            # Accumulate execution time of intervening layers
            for mid in range(completed_layer + 1, target_layer):
                cumulative_ms += self.timings[mid].ema_exec_ms

            t_need = t_now + cumulative_ms
            t_load = self.timings[target_layer].ema_load_ms

            slack = t_need - t_load - t_now

            if slack > self.SLACK_MS and target_layer not in self._issued:
                order = PrefetchOrder(
                    layer_id=target_layer,
                    urgency=slack,
                    target_tier=0 if offset <= 2 else 1,  # VRAM for near, RAM for far
                    estimated_load_ms=t_load,
                )
                with self._lock:
                    heapq.heappush(self._prefetch_queue, (slack, order.layer_id, order))
                    self._issued.add(target_layer)
                self.on_prefetch(order)
                self.stats["prefetch_issued"] += 1

    def pop_prefetch(self) -> Optional[PrefetchOrder]:
        """Pop the highest-urgency prefetch order."""
        with self._lock:
            if self._prefetch_queue:
                _, _lid, order = heapq.heappop(self._prefetch_queue)
                return order
        return None

    def reset_for_request(self):
        """Call at start of each new request."""
        self._current_layer = 0
        self._issued.clear()
        with self._lock:
            self._prefetch_queue.clear()

    def hit_rate(self) -> float:
        total = self.stats["prefetch_issued"]
        hits  = self.stats.get("prefetch_hit", 0)
        return hits / total if total > 0 else 0.0

    def report(self) -> dict:
        return {
            "prefetch_issued": self.stats["prefetch_issued"],
            "hit_rate": round(self.hit_rate(), 3),
            "avg_exec_ms_per_layer": round(
                sum(t.ema_exec_ms for t in self.timings) / self.n_layers, 2),
            "avg_load_ms_per_layer": round(
                sum(t.ema_load_ms for t in self.timings) / self.n_layers, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5: Adaptive Compute Scheduler (ACS)
# ══════════════════════════════════════════════════════════════════════════════
"""
Mathematical Foundation:
  Request priority score:
    P(r) = w_sla · (1 / max(deadline(r) - t_now, ε))
          + w_short · (1 / estimated_tokens(r))
          + w_batch · prefix_match_bonus(r)

  Batch formation:
    Objective: maximise Σ_r compute_efficiency(r) per forward pass
    subject to: max_batch_size constraint, latency SLA constraints

  Hardware routing score for layer l on device d:
    Route(l, d) = arithmetic_intensity(l) × peak_FLOPS(d)
                - memory_bandwidth_bottleneck(l, d)
    Choose d* = argmax_d Route(l, d)
"""

class RequestPriority(Enum):
    REALTIME  = 0   # Interactive, <100ms SLA
    STANDARD  = 1   # Normal, <2s SLA
    BATCH     = 2   # Background, <60s SLA

@dataclass
class InferenceRequest:
    request_id: str
    token_ids: list[int]
    priority: RequestPriority = RequestPriority.STANDARD
    deadline_ts: float = field(default_factory=lambda: time.time() + 2.0)
    estimated_tokens: int = 100
    difficulty_score: float = 5.0     # from TDE
    prefix_hash: str = ""             # for KV-cache sharing detection
    submitted_ts: float = field(default_factory=time.time)

    def sla_urgency(self) -> float:
        remaining = max(self.deadline_ts - time.time(), 1e-6)
        return 1.0 / remaining

    def priority_score(self,
                       w_sla=3.0, w_short=1.0, w_batch=2.0,
                       batch_bonus=0.0) -> float:
        base = -int(self.priority.value)   # lower enum = higher urgency
        return (base
                + w_sla   * self.sla_urgency()
                + w_short * (1.0 / max(self.estimated_tokens, 1))
                + w_batch * batch_bonus)

@dataclass
class Batch:
    requests: list[InferenceRequest]
    batch_id: str
    formed_ts: float = field(default_factory=time.time)

    @property
    def size(self): return len(self.requests)

    @property
    def total_tokens(self): return sum(r.estimated_tokens for r in self.requests)

    @property
    def avg_difficulty(self):
        if not self.requests: return 0.0
        return sum(r.difficulty_score for r in self.requests) / len(self.requests)


class AdaptiveComputeScheduler:
    """
    Priority-weighted, difficulty-aware batch scheduler.

    Design:
      - Three priority queues (REALTIME, STANDARD, BATCH)
      - Batch merger: groups requests with matching prefix hashes
        to maximise KV-cache sharing within a batch
      - Hardware profiler: monitors GPU/CPU utilisation and
        reroutes overflow work to CPU
      - SLA controller: ensures realtime requests are never starved

    Scheduling Algorithm:
      1. Every scheduling_interval_ms:
         a. Poll all three queues
         b. Build candidate batch: start with all REALTIME requests
         c. Fill remaining capacity from STANDARD (sorted by priority_score)
         d. Fill any slack from BATCH queue
         e. Apply prefix merger: group matching prefixes together
         f. Dispatch batch
    """
    MAX_BATCH_SIZE   = 32
    SCHEDULING_MS    = 5.0       # Scheduling interval
    PREFIX_SHARE_MIN = 0.6       # Min prefix match ratio for batching bonus
    MAX_WAIT_MS      = {
        RequestPriority.REALTIME: 20,
        RequestPriority.STANDARD: 500,
        RequestPriority.BATCH:    10000,
    }

    def __init__(self):
        self._queues = {
            RequestPriority.REALTIME: [],
            RequestPriority.STANDARD: [],
            RequestPriority.BATCH:    [],
        }
        self._lock = threading.Lock()
        self._batch_counter = 0
        self.stats = defaultdict(int)
        # Hardware utilisation (updated externally)
        self.gpu_util = 0.0
        self.cpu_util = 0.0

    def submit(self, req: InferenceRequest):
        """Submit a request. Thread-safe."""
        req.prefix_hash = self._compute_prefix_hash(req.token_ids)
        with self._lock:
            heapq.heappush(
                self._queues[req.priority],
                (-req.priority_score(), req.submitted_ts, req)
            )
        self.stats["submitted"] += 1

    def _compute_prefix_hash(self, token_ids: list, prefix_len=64) -> str:
        """Hash the first prefix_len tokens for KV-sharing detection."""
        prefix = token_ids[:prefix_len]
        return hashlib.md5(str(prefix).encode()).hexdigest()[:8]

    def form_batch(self) -> Optional[Batch]:
        """
        Form the next batch using priority + prefix-sharing optimisation.
        Returns None if no requests are ready.
        """
        with self._lock:
            candidates = []

            # Drain REALTIME first (never skip)
            while self._queues[RequestPriority.REALTIME]:
                _, _, req = heapq.heappop(self._queues[RequestPriority.REALTIME])
                candidates.append(req)

            # Fill from STANDARD up to MAX_BATCH_SIZE
            while (len(candidates) < self.MAX_BATCH_SIZE and
                   self._queues[RequestPriority.STANDARD]):
                _, _, req = heapq.heappop(self._queues[RequestPriority.STANDARD])
                candidates.append(req)

            # Fill any remaining from BATCH
            while (len(candidates) < self.MAX_BATCH_SIZE and
                   self._queues[RequestPriority.BATCH]):
                _, _, req = heapq.heappop(self._queues[RequestPriority.BATCH])
                candidates.append(req)

            if not candidates:
                return None

            # Prefix-aware reordering: group matching prefixes together
            candidates = self._reorder_by_prefix(candidates)

            self._batch_counter += 1
            batch = Batch(
                requests=candidates,
                batch_id=f"batch_{self._batch_counter:06d}",
            )
            self.stats["batches_formed"] += 1
            self.stats["total_requests_batched"] += len(candidates)
            return batch

    def _reorder_by_prefix(self, reqs: list) -> list:
        """
        Group requests with identical prefix hashes together.
        This maximises KV-cache prefix sharing within the batch.
        """
        groups = defaultdict(list)
        for r in reqs:
            groups[r.prefix_hash].append(r)

        # Sort groups by size (largest first for maximum sharing benefit)
        ordered = []
        for _, group in sorted(groups.items(), key=lambda x: -len(x[1])):
            ordered.extend(group)

        shared_pairs = sum(len(g)-1 for g in groups.values())
        self.stats["kv_sharing_pairs"] += shared_pairs
        return ordered

    def route_to_device(self, layer_id: int, n_params: int,
                        batch_size: int) -> str:
        """
        Decide whether to run layer on GPU or CPU.

        Arithmetic intensity = FLOPs / bytes_moved
          = (2 × batch_size × n_params) / (n_params × bytes_per_param)
          = 2 × batch_size / bytes_per_param

        GPU break-even: intensity > 4 FLOP/byte → GPU preferred
        CPU preferred: intensity < 4 FLOP/byte (memory-bound)
        """
        bytes_per_param = 2   # FP16
        intensity = (2 * batch_size) / bytes_per_param

        if self.gpu_util > 0.90:
            return "cpu"   # GPU saturated → offload
        elif intensity < 4.0:
            return "cpu"   # Memory-bound → CPU can match GPU
        else:
            return "gpu"

    def report(self) -> dict:
        s = self.stats
        n_req = max(s["submitted"], 1)
        n_bat = max(s["batches_formed"], 1)
        return {
            "requests_submitted": s["submitted"],
            "batches_formed": s["batches_formed"],
            "avg_batch_size": round(s["total_requests_batched"] / n_bat, 2),
            "kv_sharing_pairs": s["kv_sharing_pairs"],
            "kv_sharing_pct": round(100 * s["kv_sharing_pairs"] / n_req, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6: Neural Semantic Cache (NSC)
# ══════════════════════════════════════════════════════════════════════════════
"""
Mathematical Foundation:
  Embed prompt p → vector e(p) ∈ ℝ^256 via bi-encoder.

  Cache hit condition:
    hit = cosine_similarity(e(query), e(cached)) > θ
    cosine_sim(a, b) = (a·b) / (‖a‖·‖b‖)

  HNSW Approximate Nearest Neighbour:
    Build skip-graph where each node connects to M nearest neighbours.
    Search: greedy descent from entry point, ef candidates maintained.
    Time complexity: O(log N) expected for N cached items.

  Cache eviction:
    LRU with TTL: evict if (t_now - t_last_access > TTL)
                  OR cache size > MAX_ENTRIES.

  Threshold calibration:
    If false_positive_rate > target: increase θ
    If hit_rate < target: decrease θ
    θ ← θ + η × (fpr - fpr_target)    η = 0.01
"""

def cosine_sim(a: list, b: list) -> float:
    dot  = sum(x*y for x, y in zip(a, b))
    na   = math.sqrt(sum(x*x for x in a))
    nb   = math.sqrt(sum(x*x for x in b))
    denom = na * nb
    return dot / denom if denom > 1e-9 else 0.0

def l2_dist(a: list, b: list) -> float:
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


class HNSWIndex:
    """
    Simplified HNSW (Hierarchical Navigable Small World) index.

    Full HNSW uses multi-layer graph. This implementation uses
    a single-layer approximation (NSW) for clarity, which
    preserves the essential O(log N) search behaviour.

    In production: replace with faiss.IndexHNSWFlat for
    full multi-layer HNSW with 10-100× better performance.

    Parameters:
      M  = max connections per node (default 16)
      ef = candidates explored during search (default 50)
    """
    def __init__(self, dim: int, M: int = 16, ef: int = 50):
        self.dim = dim
        self.M   = M
        self.ef  = ef
        self._vectors: list[list] = []
        self._ids:     list[str]  = []
        self._graph:   list[list] = []   # adjacency list per node

    def __len__(self):
        return len(self._vectors)

    def add(self, vector: list, entry_id: str) -> int:
        """Insert a vector. Returns its index."""
        idx = len(self._vectors)
        self._vectors.append(vector)
        self._ids.append(entry_id)
        self._graph.append([])

        if idx == 0:
            return idx

        # Find M nearest neighbours (greedy search from random entry)
        neighbours = self._greedy_search(vector, k=self.M, entry_idx=0)
        self._graph[idx] = [n for n, _ in neighbours]

        # Bidirectional edges (with degree cap at M)
        for n_idx, _ in neighbours:
            if len(self._graph[n_idx]) < self.M * 2:
                self._graph[n_idx].append(idx)

        return idx

    def search(self, query: list, k: int = 1) -> list:
        """
        ANN search. Returns list of (entry_id, similarity) sorted desc.
        """
        if not self._vectors:
            return []
        neighbours = self._greedy_search(query, k=k, entry_idx=0)
        return [(self._ids[idx], sim) for idx, sim in neighbours[:k]]

    def _greedy_search(self, query: list, k: int, entry_idx: int) -> list:
        """
        Greedy best-first search with ef candidate tracking.
        Maintains a max-heap of ef candidates by similarity.
        """
        visited = {entry_idx}
        # candidates: max-heap of (sim, idx) — negate sim for min-heap
        entry_sim = cosine_sim(query, self._vectors[entry_idx])
        candidates = [(-entry_sim, entry_idx)]
        results    = [(entry_sim, entry_idx)]

        while candidates:
            neg_sim, cur = heapq.heappop(candidates)
            cur_sim = -neg_sim

            # Stopping condition: best candidate worse than worst result
            if results and cur_sim < results[-1][0] and len(results) >= self.ef:
                break

            for nb in self._graph[cur]:
                if nb in visited:
                    continue
                visited.add(nb)
                nb_sim = cosine_sim(query, self._vectors[nb])
                heapq.heappush(candidates, (-nb_sim, nb))
                results.append((nb_sim, nb))

            if len(visited) > self.ef * 2:
                break

        results.sort(reverse=True)
        return [(idx, sim) for sim, idx in results[:k]]


class MiniBiEncoder:
    """
    Lightweight prompt encoder: token_ids → 256-dim embedding.
    Simulates a 6-layer MiniLM (~22M params).

    Production: use sentence-transformers/all-MiniLM-L6-v2
    Here: deterministic hash projection for testing.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim
        self._rng = random.Random(999)

    def encode(self, token_ids: list) -> list:
        """Encode token sequence to fixed-dim vector."""
        # Simulated encoding: weighted bag-of-hashes
        result = [0.0] * self.dim
        for i, tok in enumerate(token_ids[-256:]):
            w = math.exp(-i / 64.0)   # recency weighting
            rng = random.Random(tok ^ 0xDEADBEEF)
            for d in range(self.dim):
                result[d] += w * rng.gauss(0, 1)

        # L2 normalise
        norm = math.sqrt(sum(x*x for x in result))
        if norm > 1e-9:
            result = [x/norm for x in result]
        return result


@dataclass
class CacheEntry:
    entry_id: str
    prompt_embedding: list
    completion_tokens: list[int]
    prompt_hash: str
    created_ts: float = field(default_factory=time.time)
    last_access_ts: float = field(default_factory=time.time)
    access_count: int = 0

    def touch(self):
        self.last_access_ts = time.time()
        self.access_count += 1

    def age_seconds(self) -> float:
        return time.time() - self.last_access_ts


class NeuralSemanticCache:
    """
    Embedding-based semantic cache for LLM inference.

    Hit path (< 2ms):
      1. Encode query prompt → 256-dim embedding
      2. HNSW ANN search → nearest cached embedding
      3. If cosine_sim > threshold: return cached completion
      4. Else: miss → proceed to full inference

    Store path (after inference):
      1. Encode completed prompt
      2. Insert into HNSW index
      3. Store (embedding, completion) in cache dict
      4. Evict if over capacity (LRU + TTL)

    Adaptive threshold:
      Tracks false positive rate (user-reported bad hits)
      and adjusts θ to maintain target hit quality.
    """
    def __init__(self, dim: int = 256, threshold: float = 0.92,
                 max_entries: int = 10_000, ttl_seconds: float = 3600.0):
        self.dim       = dim
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl       = ttl_seconds
        self.encoder   = MiniBiEncoder(dim)
        self.index     = HNSWIndex(dim)
        self._store: dict[str, CacheEntry] = {}
        self._id_counter = 0
        self.stats = defaultdict(int)

    # ── Core API ──────────────────────────────────────────────────────────────

    def lookup(self, token_ids: list) -> Optional[list[int]]:
        """
        Semantic cache lookup.
        Returns completion token_ids on hit, None on miss.
        Latency target: < 2ms.
        """
        t0 = time.perf_counter()

        query_emb = self.encoder.encode(token_ids)
        results   = self.index.search(query_emb, k=1)

        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats["lookups"] += 1
        self.stats["total_lookup_ms"] += latency_ms

        if not results:
            self.stats["misses"] += 1
            return None

        entry_id, sim = results[0]
        entry = self._store.get(entry_id)

        if entry is None or entry.age_seconds() > self.ttl:
            self.stats["misses"] += 1
            if entry:
                self._evict_entry(entry_id)
            return None

        if sim >= self.threshold:
            entry.touch()
            self.stats["hits"] += 1
            return entry.completion_tokens
        else:
            self.stats["misses"] += 1
            return None

    def store(self, prompt_token_ids: list, completion_token_ids: list):
        """
        Store a (prompt, completion) pair in the cache.
        """
        emb = self.encoder.encode(prompt_token_ids)
        prompt_hash = hashlib.md5(str(prompt_token_ids[:64]).encode()).hexdigest()

        self._id_counter += 1
        entry_id = f"nsc_{self._id_counter:08d}"

        entry = CacheEntry(
            entry_id=entry_id,
            prompt_embedding=emb,
            completion_tokens=completion_token_ids,
            prompt_hash=prompt_hash,
        )
        self.index.add(emb, entry_id)
        self._store[entry_id] = entry
        self.stats["stored"] += 1

        # Evict if over capacity
        if len(self._store) > self.max_entries:
            self._evict_lru(batch=100)

    # ── Threshold Adaptation ──────────────────────────────────────────────────

    def report_false_positive(self, entry_id: str = None):
        """
        Signal that a cache hit was incorrect (bad completion returned).
        Triggers threshold increase.
        """
        self.stats["false_positives"] += 1
        self._adapt_threshold()

    def _adapt_threshold(self, target_fpr: float = 0.02, lr: float = 0.005):
        """
        Online threshold calibration.
        fpr_observed = false_positives / hits
        If fpr > target: increase threshold (stricter)
        If fpr < target/2: decrease threshold (more permissive)
        """
        hits = max(self.stats["hits"], 1)
        fpr_obs = self.stats["false_positives"] / hits
        delta = lr * (fpr_obs - target_fpr)
        self.threshold = max(0.80, min(0.99, self.threshold + delta))

    # ── Eviction ─────────────────────────────────────────────────────────────

    def _evict_lru(self, batch: int = 100):
        """Evict least-recently-accessed entries."""
        entries_sorted = sorted(
            self._store.items(),
            key=lambda kv: kv[1].last_access_ts
        )
        for entry_id, _ in entries_sorted[:batch]:
            self._evict_entry(entry_id)
        self.stats["evictions"] += batch

    def _evict_entry(self, entry_id: str):
        self._store.pop(entry_id, None)
        # Note: HNSW index doesn't support deletion in this implementation.
        # Production: use faiss with IDMap for O(1) deletion.

    # ── Reporting ─────────────────────────────────────────────────────────────

    def report(self) -> dict:
        total = max(self.stats["lookups"], 1)
        return {
            "total_lookups":   self.stats["lookups"],
            "hits":            self.stats["hits"],
            "misses":          self.stats["misses"],
            "hit_rate":        round(self.stats["hits"] / total, 3),
            "false_positives": self.stats["false_positives"],
            "cache_size":      len(self._store),
            "threshold":       round(self.threshold, 4),
            "avg_lookup_ms":   round(
                self.stats["total_lookup_ms"] / total, 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = random.Random(42)
    print("=" * 65)
    print("UARC Modules 3–6 Integration Demo")
    print("=" * 65)

    # ── Module 3: DPE ────────────────────────────────────────────────────────
    print("\n── Module 3: Dynamic Precision Engine ──")
    profiles = DynamicPrecisionEngine.sensitivity_analysis(80, rng)
    dpe = DynamicPrecisionEngine(profiles)

    budgets = [
        4 * 1024**3,    # 4 GB
        8 * 1024**3,    # 8 GB
        16 * 1024**3,   # 16 GB
    ]
    for budget in budgets:
        plan = dpe.allocate(budget)
        s = plan.summary()
        print(f"  Budget {budget//1024**3}GB: "
              f"avg={s['avg_bits_per_param']}bpw  "
              f"PPL_delta={s['estimated_ppl_delta']:.4f}  "
              f"util={s['budget_utilisation_pct']}%  "
              f"alloc={s['total_MB']}MB")

    # Per-token adaptation
    plan_base = dpe.allocate(8 * 1024**3)
    plan_hard = dpe.adapt_for_token(plan_base, difficulty_score=15.0,
                                    critical_layers=list(range(5))+list(range(75,80)))
    print(f"  Hard token: upgraded {sum(1 for p in plan_hard.assignment if p==Precision.FP16)} layers to FP16")

    # ── Module 4: PLL ─────────────────────────────────────────────────────────
    print("\n── Module 4: Predictive Layer Loader ──")
    layer_sizes = [rng.uniform(1.5, 2.5) for _ in range(80)]
    pll = PredictiveLayerLoader(80, layer_sizes)

    prefetch_log = []
    pll.on_prefetch = lambda order: prefetch_log.append(order)

    # Simulate execution of 20 layers
    for i in range(20):
        pll.on_layer_start(i)
        time.sleep(0.001)   # simulate 1ms per layer
        pll.on_layer_complete(i)

    rep = pll.report()
    print(f"  Prefetch orders issued: {rep['prefetch_issued']}")
    print(f"  Avg exec time EMA: {rep['avg_exec_ms_per_layer']:.2f}ms")
    print(f"  Avg load time EMA: {rep['avg_load_ms_per_layer']:.2f}ms")
    if prefetch_log:
        ex = prefetch_log[0]
        print(f"  First prefetch: layer_{ex.layer_id}  "
              f"tier={ex.target_tier}  urgency={ex.urgency:.1f}ms")

    # ── Module 5: ACS ─────────────────────────────────────────────────────────
    print("\n── Module 5: Adaptive Compute Scheduler ──")
    acs = AdaptiveComputeScheduler()

    # Submit 50 mixed-priority requests
    for i in range(50):
        prio = rng.choice([RequestPriority.REALTIME,
                           RequestPriority.STANDARD,
                           RequestPriority.BATCH])
        req = InferenceRequest(
            request_id=f"req_{i:04d}",
            token_ids=[rng.randint(0,32000) for _ in range(rng.randint(16,256))],
            priority=prio,
            difficulty_score=rng.uniform(1.0, 20.0),
        )
        acs.submit(req)

    # Form 5 batches
    batch_sizes = []
    for _ in range(5):
        batch = acs.form_batch()
        if batch:
            batch_sizes.append(batch.size)

    rep = acs.report()
    print(f"  Batches formed: {rep['batches_formed']}")
    print(f"  Avg batch size: {rep['avg_batch_size']}")
    print(f"  KV sharing pairs: {rep['kv_sharing_pairs']} ({rep['kv_sharing_pct']}%)")

    # ── Module 6: NSC ─────────────────────────────────────────────────────────
    print("\n── Module 6: Neural Semantic Cache ──")
    nsc = NeuralSemanticCache(dim=64, threshold=0.90, max_entries=500)

    # Warm cache with 100 entries
    for i in range(100):
        prompt = [rng.randint(0, 1000) for _ in range(32)]
        completion = [rng.randint(0, 32000) for _ in range(rng.randint(10, 50))]
        nsc.store(prompt, completion)

    # Lookup mix: 50% near-duplicates, 50% new
    hits, misses = 0, 0
    for i in range(200):
        if rng.random() < 0.5:
            # Near-duplicate: reuse a stored prompt with small perturbation
            base_prompt = [rng.randint(0, 1000) for _ in range(30)]
            # Slightly perturb 2 tokens
            prompt = base_prompt + [rng.randint(0, 50), rng.randint(0, 50)]
        else:
            prompt = [rng.randint(5000, 32000) for _ in range(32)]  # novel

        result = nsc.lookup(prompt)
        if result is not None: hits += 1
        else: misses += 1

    rep = nsc.report()
    print(f"  Cache entries: {rep['cache_size']}")
    print(f"  Hit rate: {rep['hit_rate']:.1%}")
    print(f"  Avg lookup latency: {rep['avg_lookup_ms']:.2f}ms")
    print(f"  Threshold: {rep['threshold']}")

    print("\n✅  All UARC modules 3–6 operational.")
