"""
UARC Module 1: Token Difficulty Estimator (TDE)
================================================
Per-token compute routing via 4-layer MLP perplexity regression.

Mathematical Foundation:
  Let x_t = context embedding at token position t
  TDE learns: f_θ(x_t) → ŷ_t  where ŷ_t ≈ perplexity(t)

  Routing decision:
    ŷ_t < τ_easy  → DraftModel   (skip full model, ~85% compute saved)
    τ_easy ≤ ŷ_t < τ_hard → PartialModel (skip top 30%, ~35% saved)
    ŷ_t ≥ τ_hard  → FullModel    (all layers, FP16)
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from uarc.core.config import TDEConfig
from uarc.core.types import RoutingDecision, RouteTarget


# ── Simulated tensor ops (pure Python, no external deps) ─────────────────────

def _zeros(shape):
    if isinstance(shape, int):
        return [0.0] * shape
    rows, cols = shape
    return [[0.0] * cols for _ in range(rows)]


def _randn(shape, scale=0.02):
    """Box-Muller Gaussian samples."""
    def _bm():
        while True:
            u, v = random.random(), random.random()
            if u > 0:
                break
        return scale * math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)
    if isinstance(shape, int):
        return [_bm() for _ in range(shape)]
    rows, cols = shape
    return [[_bm() for _ in range(cols)] for _ in range(rows)]


def _relu(x):
    if isinstance(x, list):
        return [max(0.0, v) for v in x]
    return max(0.0, x)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _matvec(W, x):
    return [_dot(row, x) for row in W]


def _layer_norm(x, eps=1e-6):
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]


def _add_vectors(a, b):
    return [x + y for x, y in zip(a, b)]


# ── MLP ──────────────────────────────────────────────────────────────────────

class _MLP:
    """
    4-layer MLP for perplexity regression.

    Architecture:
      Input(context_dim) → Linear → LayerNorm → ReLU  (×n_hidden)
                         → Linear → Scalar output (log-ppl)
    """

    def __init__(self, cfg: TDEConfig):
        self.cfg = cfg
        dims = [cfg.context_dim] + [cfg.hidden_dim] * cfg.n_hidden + [1]
        self.W = [_randn((dims[i + 1], dims[i]),
                         scale=math.sqrt(2 / dims[i]))
                  for i in range(len(dims) - 1)]
        self.b = [_zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.ln_g = [[1.0] * dims[i + 1] for i in range(len(dims) - 2)]
        self.ln_b = [[0.0] * dims[i + 1] for i in range(len(dims) - 2)]
        self.dW = [_zeros((len(self.W[i]), len(self.W[i][0])))
                   for i in range(len(self.W))]
        self.db = [_zeros(len(self.b[i])) for i in range(len(self.b))]

    def forward(self, x: list, training=False) -> tuple:
        cache = [x]
        h = x[:]
        for i in range(len(self.W) - 1):
            z = _add_vectors(_matvec(self.W[i], h), self.b[i])
            z = _layer_norm(z)
            z = [g * v + b for g, v, b in zip(self.ln_g[i], z, self.ln_b[i])]
            h = _relu(z)
            cache.append(h)
        out = _add_vectors(_matvec(self.W[-1], h), self.b[-1])
        log_ppl = out[0]
        cache.append([log_ppl])
        return log_ppl, cache

    def predict_ppl(self, context_embedding: list) -> float:
        log_ppl, _ = self.forward(context_embedding, training=False)
        return math.exp(max(-20, min(20, log_ppl)))

    def backward(self, cache: list, target_log_ppl: float) -> float:
        pred = cache[-1][0]
        loss = (pred - target_log_ppl) ** 2
        grad = 2.0 * (pred - target_log_ppl)
        h_prev = cache[-2]
        for j in range(len(h_prev)):
            self.dW[-1][0][j] += grad * h_prev[j]
        self.db[-1][0] += grad
        return loss

    def step(self):
        lr = self.cfg.lr
        for i in range(len(self.W)):
            for r in range(len(self.W[i])):
                for c in range(len(self.W[i][r])):
                    self.W[i][r][c] -= lr * self.dW[i][r][c]
                    self.dW[i][r][c] = 0.0
            for j in range(len(self.b[i])):
                self.b[i][j] -= lr * self.db[i][j]
                self.db[i][j] = 0.0


# ── Context Encoder ──────────────────────────────────────────────────────────

class _ContextEncoder:
    """
    Lightweight encoder: token_ids[-128:] → 128-dim embedding.
    Uses bag-of-positions + hash projection. O(1) wrt sequence length.
    """

    def __init__(self, dim=128):
        self.dim = dim
        self.seed = 42
        self.tau_pos = 32.0

    def _hash_embed(self, token_id: int) -> list:
        rng = random.Random(token_id ^ self.seed)
        return [rng.gauss(0, 1) for _ in range(self.dim)]

    def encode(self, token_ids: list) -> list:
        window = token_ids[-128:]
        n = len(window)
        result = [0.0] * self.dim
        total_weight = 0.0

        for i, tok in enumerate(window):
            pos_from_end = n - 1 - i
            weight = math.exp(-pos_from_end / self.tau_pos)
            emb = self._hash_embed(tok)
            for d in range(self.dim):
                result[d] += weight * emb[d]
            total_weight += weight

        if total_weight > 0:
            result = [v / total_weight for v in result]
        return _layer_norm(result)


# ── Token Difficulty Estimator ───────────────────────────────────────────────

class TokenDifficultyEstimator:
    """
    Full TDE pipeline:
      1. ContextEncoder: token_ids → embedding
      2. MLP: embedding → log_ppl estimate
      3. Router: ppl estimate → route decision

    Adaptive threshold calibration via EMA over estimation errors.
    """

    def __init__(self, cfg: TDEConfig | None = None):
        self.cfg = cfg or TDEConfig()
        self.encoder = _ContextEncoder(dim=self.cfg.context_dim)
        self.mlp = _MLP(self.cfg)
        self.tau_easy = self.cfg.tau_easy
        self.tau_hard = self.cfg.tau_hard

        # Stats tracking
        self._total_routed = 0
        self._draft_count = 0
        self._partial_count = 0
        self._full_count = 0
        self._avg_ppl = 0.0

        # Calibration buffer
        self._calib_buffer: list[tuple[float, float]] = []
        self._calib_window = 500

    # ── Core API ─────────────────────────────────────────────────────────────

    def estimate(self, token_ids: list) -> RoutingDecision:
        """Main entry point. Given token context, return routing decision."""
        t0 = time.perf_counter()

        emb = self.encoder.encode(token_ids)
        est_ppl = self.mlp.predict_ppl(emb)
        route, compute_saved = self._route(est_ppl)
        confidence = self._compute_confidence(est_ppl)
        latency_ms = (time.perf_counter() - t0) * 1000

        self._update_stats(route, est_ppl)

        return RoutingDecision(
            route=route,
            estimated_ppl=est_ppl,
            confidence=confidence,
            latency_ms=latency_ms,
            compute_saved_pct=compute_saved,
        )

    def _route(self, est_ppl: float) -> tuple[RouteTarget, float]:
        if est_ppl < self.tau_easy:
            return RouteTarget.DRAFT, 85.0
        elif est_ppl < self.tau_hard:
            return RouteTarget.PARTIAL, 35.0
        else:
            return RouteTarget.FULL, 0.0

    def _compute_confidence(self, est_ppl: float) -> float:
        d_easy = abs(est_ppl - self.tau_easy)
        d_hard = abs(est_ppl - self.tau_hard)
        d_min = min(d_easy, d_hard)
        return _sigmoid(d_min - 1.0)

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self, est_ppl: float, actual_ppl: float):
        """Online calibration: adjust thresholds based on estimation error."""
        self._calib_buffer.append((est_ppl, actual_ppl))
        if len(self._calib_buffer) >= self._calib_window:
            self._run_calibration()
            self._calib_buffer.clear()

    def _run_calibration(self):
        errors = [actual - est for est, actual in self._calib_buffer]
        bias = sum(errors) / len(errors)
        alpha = self.cfg.ema_alpha
        self.tau_easy = alpha * self.tau_easy + (1 - alpha) * (self.tau_easy + bias)
        self.tau_hard = alpha * self.tau_hard + (1 - alpha) * (self.tau_hard + bias)
        self.tau_easy = max(1.0, self.tau_easy)
        self.tau_hard = max(self.tau_easy + 1.0, self.tau_hard)

    # ── Training ─────────────────────────────────────────────────────────────

    def train_step(self, token_ids: list, actual_ppl: float) -> float:
        emb = self.encoder.encode(token_ids)
        target_log_ppl = math.log(max(actual_ppl, 1e-6))
        _, cache = self.mlp.forward(emb, training=True)
        loss = self.mlp.backward(cache, target_log_ppl)
        self.mlp.step()
        return loss

    def train(self, dataset: list, epochs: int = 3) -> list[float]:
        """Train on (token_ids, perplexity) pairs. Returns per-epoch losses."""
        epoch_losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            for toks, ppl in dataset:
                total_loss += self.train_step(toks, ppl)
            avg = total_loss / max(len(dataset), 1)
            epoch_losses.append(avg)
        return epoch_losses

    # ── Stats ────────────────────────────────────────────────────────────────

    def _update_stats(self, route: RouteTarget, est_ppl: float):
        self._total_routed += 1
        if route == RouteTarget.DRAFT:
            self._draft_count += 1
        elif route == RouteTarget.PARTIAL:
            self._partial_count += 1
        else:
            self._full_count += 1
        n = self._total_routed
        self._avg_ppl = (self._avg_ppl * (n - 1) + est_ppl) / n

    def stats(self) -> dict:
        n = max(self._total_routed, 1)
        return {
            "total_routed": self._total_routed,
            "avg_estimated_ppl": round(self._avg_ppl, 3),
            "tau_easy": round(self.tau_easy, 3),
            "tau_hard": round(self.tau_hard, 3),
            "route_pct": {
                "draft": round(100 * self._draft_count / n, 1),
                "partial": round(100 * self._partial_count / n, 1),
                "full": round(100 * self._full_count / n, 1),
            },
            "estimated_compute_saved_pct": round(
                (0.85 * self._draft_count + 0.35 * self._partial_count) / n * 100, 1),
        }
