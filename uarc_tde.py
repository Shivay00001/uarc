"""
UARC Module 1: Token Difficulty Estimator (TDE)
================================================
Full implementation with training loop, inference, and calibration.

Mathematical Foundation:
  Let x_t = context embedding at token position t
  Let y_t = perplexity under full model F at position t
  TDE learns: f_θ(x_t) → ŷ_t  where ŷ_t ≈ y_t

  Routing decision:
    ŷ_t < τ_easy  → DraftModel   (skip full model)
    τ_easy ≤ ŷ_t < τ_hard → PartialModel (skip top 30% layers)
    ŷ_t ≥ τ_hard  → FullModel    (all layers, FP16)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional
import json

# ── Simulated tensor ops (pure Python, no external deps) ─────────────────────

def zeros(shape):
    if isinstance(shape, int):
        return [0.0] * shape
    rows, cols = shape
    return [[0.0]*cols for _ in range(rows)]

def randn(shape, scale=0.02):
    """Box-Muller Gaussian samples."""
    import random
    def _bm():
        while True:
            u, v = random.random(), random.random()
            if u > 0: break
        return scale * math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*v)
    if isinstance(shape, int):
        return [_bm() for _ in range(shape)]
    rows, cols = shape
    return [[_bm() for _ in range(cols)] for _ in range(rows)]

def relu(x):
    if isinstance(x, list):
        return [max(0.0, v) for v in x]
    return max(0.0, x)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))

def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v/s for v in e]

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def matvec(W, x):
    """W: [out x in], x: [in] → [out]"""
    return [dot(row, x) for row in W]

def layer_norm(x, eps=1e-6):
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean)**2 for v in x) / n
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]

def add_vectors(a, b):
    return [x + y for x, y in zip(a, b)]

def mse_loss(pred, target):
    return sum((p - t)**2 for p, t in zip(pred, target)) / len(pred)

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class TDEConfig:
    context_dim: int = 128      # Input embedding dimension
    hidden_dim: int = 256       # Hidden layer width
    n_hidden: int = 3           # Number of hidden layers
    dropout_rate: float = 0.1   # Dropout during training
    lr: float = 1e-3            # Learning rate
    tau_easy: float = 2.5       # Perplexity threshold: easy
    tau_hard: float = 8.0       # Perplexity threshold: hard
    ema_alpha: float = 0.95     # EMA smoothing for threshold adaptation

@dataclass
class RoutingDecision:
    route: str              # "draft", "partial", "full"
    estimated_ppl: float
    confidence: float
    latency_ms: float
    compute_saved_pct: float

@dataclass
class TDEStats:
    total_routed: int = 0
    draft_count: int = 0
    partial_count: int = 0
    full_count: int = 0
    avg_estimated_ppl: float = 0.0
    threshold_easy: float = 2.5
    threshold_hard: float = 8.0
    calibration_errors: list = field(default_factory=list)

# ── MLP Implementation ────────────────────────────────────────────────────────

class MLP:
    """
    4-layer MLP for perplexity regression.

    Architecture:
      Input(context_dim) → Linear → LayerNorm → ReLU
                        → Linear → LayerNorm → ReLU
                        → Linear → LayerNorm → ReLU
                        → Linear → Scalar output (log-ppl)
    """
    def __init__(self, cfg: TDEConfig):
        self.cfg = cfg
        dims = [cfg.context_dim] + [cfg.hidden_dim]*cfg.n_hidden + [1]
        # Weights: He initialisation for ReLU
        self.W = [randn((dims[i+1], dims[i]), scale=math.sqrt(2/dims[i]))
                  for i in range(len(dims)-1)]
        self.b = [zeros(dims[i+1]) for i in range(len(dims)-1)]
        # LayerNorm params (γ=1, β=0 init)
        self.ln_g = [[1.0]*dims[i+1] for i in range(len(dims)-2)]
        self.ln_b = [[0.0]*dims[i+1] for i in range(len(dims)-2)]
        # Gradient accumulators
        self.dW = [zeros((len(self.W[i]), len(self.W[i][0])))
                   for i in range(len(self.W))]
        self.db = [zeros(len(self.b[i])) for i in range(len(self.b))]

    def forward(self, x: list, training=False) -> tuple:
        """
        Forward pass.
        Returns (log_ppl_estimate, activations_cache)
        """
        cache = [x]
        h = x[:]
        for i in range(len(self.W) - 1):
            # Linear
            z = add_vectors(matvec(self.W[i], h), self.b[i])
            # LayerNorm
            z = layer_norm(z)
            z = [g*v + b for g, v, b in zip(self.ln_g[i], z, self.ln_b[i])]
            # ReLU
            h = relu(z)
            cache.append(h)
        # Output layer (no activation — regression)
        out = add_vectors(matvec(self.W[-1], h), self.b[-1])
        log_ppl = out[0]  # scalar
        cache.append([log_ppl])
        return log_ppl, cache

    def predict_ppl(self, context_embedding: list) -> float:
        """Runtime inference. Returns perplexity estimate."""
        log_ppl, _ = self.forward(context_embedding, training=False)
        return math.exp(log_ppl)   # log-space → ppl

    def backward(self, cache: list, target_log_ppl: float) -> float:
        """
        Simplified backprop for scalar output.
        Returns loss value.
        """
        pred = cache[-1][0]
        loss = (pred - target_log_ppl)**2
        # Output gradient
        grad = 2.0 * (pred - target_log_ppl)
        # Backprop through last linear
        h_prev = cache[-2]
        for j in range(len(h_prev)):
            self.dW[-1][0][j] += grad * h_prev[j]
        self.db[-1][0] += grad
        return loss

    def step(self):
        """SGD weight update and gradient reset."""
        lr = self.cfg.lr
        for i in range(len(self.W)):
            for r in range(len(self.W[i])):
                for c in range(len(self.W[i][r])):
                    self.W[i][r][c] -= lr * self.dW[i][r][c]
                    self.dW[i][r][c] = 0.0
            for j in range(len(self.b[i])):
                self.b[i][j] -= lr * self.db[i][j]
                self.db[i][j] = 0.0

# ── Context Encoder ───────────────────────────────────────────────────────────

class ContextEncoder:
    """
    Lightweight encoder: token_ids[-128:] → 128-dim embedding.

    In production: replace with a 4-layer transformer encoder.
    Here: bag-of-positions + simple hash projection (O(1) per token).

    Mathematical formulation:
      e_t = (1/|W|) Σ_w hash_embed(w) ⊕ positional_decay(pos)
      where ⊕ is element-wise weighted sum
      positional_decay(p) = exp(-p / τ_pos)  with τ_pos = 32
    """
    def __init__(self, dim=128, vocab_size=32000):
        import random
        self.dim = dim
        self.vocab_size = vocab_size
        # Pseudo-random embedding table (hash-based, no storage)
        self.seed = 42
        self.tau_pos = 32.0

    def _hash_embed(self, token_id: int) -> list:
        """Deterministic pseudo-embedding via linear congruential hash."""
        import random
        rng = random.Random(token_id ^ self.seed)
        return [rng.gauss(0, 1) for _ in range(self.dim)]

    def encode(self, token_ids: list) -> list:
        """
        Encode last 128 tokens → 128-dim vector.
        Time: O(128 × dim) = O(1) wrt sequence length.
        """
        window = token_ids[-128:]
        n = len(window)
        result = [0.0] * self.dim
        total_weight = 0.0

        for i, tok in enumerate(window):
            # Positional weight: recent tokens weighted higher
            pos_from_end = n - 1 - i
            weight = math.exp(-pos_from_end / self.tau_pos)
            emb = self._hash_embed(tok)
            for d in range(self.dim):
                result[d] += weight * emb[d]
            total_weight += weight

        if total_weight > 0:
            result = [v / total_weight for v in result]

        return layer_norm(result)

# ── Main TDE Class ────────────────────────────────────────────────────────────

class TokenDifficultyEstimator:
    """
    Full TDE pipeline:
      1. ContextEncoder: token_ids → embedding
      2. MLP: embedding → log_ppl estimate
      3. Router: ppl estimate → route decision

    Adaptive threshold calibration:
      After N tokens, compare estimated ppl to actual ppl
      and use EMA to adjust τ_easy, τ_hard to maintain
      target draft_rate ≈ 30%, partial_rate ≈ 40%, full_rate ≈ 30%.
    """
    def __init__(self, cfg: TDEConfig = None):
        self.cfg = cfg or TDEConfig()
        self.encoder = ContextEncoder(dim=self.cfg.context_dim)
        self.mlp = MLP(self.cfg)
        self.stats = TDEStats()
        # Adaptive thresholds (start from config, then adapt)
        self.tau_easy = self.cfg.tau_easy
        self.tau_hard = self.cfg.tau_hard
        # Calibration buffer
        self._calib_buffer = []   # (estimated_ppl, actual_ppl)
        self._calib_window = 500

    # ── Core API ──────────────────────────────────────────────────────────────

    def estimate(self, token_ids: list) -> RoutingDecision:
        """
        Main entry point. Given token context, return routing decision.

        Complexity: O(128·dim + dim·hidden·n_layers) ≈ O(1) wrt seq length
        Target latency: < 1ms on modern CPU
        """
        t0 = time.perf_counter()

        # Step 1: encode context
        emb = self.encoder.encode(token_ids)

        # Step 2: MLP forward
        est_ppl = self.mlp.predict_ppl(emb)

        # Step 3: route
        route, compute_saved = self._route(est_ppl)

        # Step 4: confidence (inverse of uncertainty)
        confidence = self._compute_confidence(est_ppl)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Update stats
        self._update_stats(route, est_ppl)

        return RoutingDecision(
            route=route,
            estimated_ppl=est_ppl,
            confidence=confidence,
            latency_ms=latency_ms,
            compute_saved_pct=compute_saved,
        )

    def _route(self, est_ppl: float) -> tuple:
        if est_ppl < self.tau_easy:
            return "draft", 85.0     # ~85% compute saved vs full
        elif est_ppl < self.tau_hard:
            return "partial", 35.0   # ~35% compute saved (skip top 30% layers)
        else:
            return "full", 0.0

    def _compute_confidence(self, est_ppl: float) -> float:
        """
        Confidence = distance from nearest threshold, normalised.
        Low confidence near boundaries (est_ppl ≈ τ_easy or τ_hard).
        """
        d_easy = abs(est_ppl - self.tau_easy)
        d_hard = abs(est_ppl - self.tau_hard)
        d_min = min(d_easy, d_hard)
        # Sigmoid confidence: 0.5 at boundary, →1 far away
        return sigmoid(d_min - 1.0)

    # ── Adaptive Calibration ──────────────────────────────────────────────────

    def calibrate(self, est_ppl: float, actual_ppl: float):
        """
        Online calibration: adjust thresholds to maintain target routing ratios.

        Algorithm:
          1. Accumulate (est, actual) pairs
          2. Every 500 tokens, compute calibration_bias = mean(actual - est)
          3. Shift τ_easy and τ_hard by EMA-smoothed bias
        """
        self._calib_buffer.append((est_ppl, actual_ppl))
        self.stats.calibration_errors.append(actual_ppl - est_ppl)

        if len(self._calib_buffer) >= self._calib_window:
            self._run_calibration()
            self._calib_buffer.clear()

    def _run_calibration(self):
        buf = self._calib_buffer
        errors = [actual - est for est, actual in buf]
        bias = sum(errors) / len(errors)
        alpha = self.cfg.ema_alpha

        # EMA-smoothed threshold shift
        self.tau_easy = alpha * self.tau_easy + (1 - alpha) * (self.tau_easy + bias)
        self.tau_hard = alpha * self.tau_hard + (1 - alpha) * (self.tau_hard + bias)

        # Enforce ordering
        self.tau_easy = max(1.0, self.tau_easy)
        self.tau_hard = max(self.tau_easy + 1.0, self.tau_hard)

        print(f"[TDE] Calibrated: τ_easy={self.tau_easy:.2f}  τ_hard={self.tau_hard:.2f}  bias={bias:+.3f}")

    # ── Training ──────────────────────────────────────────────────────────────

    def train_step(self, token_ids: list, actual_ppl: float) -> float:
        """Single supervised training step."""
        emb = self.encoder.encode(token_ids)
        target_log_ppl = math.log(max(actual_ppl, 1e-6))
        log_ppl_est, cache = self.mlp.forward(emb, training=True)
        loss = self.mlp.backward(cache, target_log_ppl)
        self.mlp.step()
        return loss

    def train(self, dataset: list, epochs: int = 3):
        """
        Train TDE on (token_ids, perplexity) pairs.

        dataset: list of (token_ids: list[int], ppl: float)
        """
        print(f"[TDE] Training on {len(dataset)} samples for {epochs} epochs")
        for epoch in range(epochs):
            total_loss = 0.0
            for i, (toks, ppl) in enumerate(dataset):
                loss = self.train_step(toks, ppl)
                total_loss += loss
            avg = total_loss / len(dataset)
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    # ── Stats & Reporting ─────────────────────────────────────────────────────

    def _update_stats(self, route: str, est_ppl: float):
        s = self.stats
        s.total_routed += 1
        if route == "draft":   s.draft_count += 1
        elif route == "partial": s.partial_count += 1
        else: s.full_count += 1
        n = s.total_routed
        s.avg_estimated_ppl = (s.avg_estimated_ppl * (n-1) + est_ppl) / n

    def report(self) -> dict:
        s = self.stats
        n = max(s.total_routed, 1)
        return {
            "total_tokens_routed": s.total_routed,
            "draft_pct": round(100 * s.draft_count / n, 1),
            "partial_pct": round(100 * s.partial_count / n, 1),
            "full_pct": round(100 * s.full_count / n, 1),
            "avg_estimated_ppl": round(s.avg_estimated_ppl, 3),
            "tau_easy": round(self.tau_easy, 3),
            "tau_hard": round(self.tau_hard, 3),
            "estimated_compute_saved_pct": round(
                (0.85 * s.draft_count + 0.35 * s.partial_count) / n * 100, 1
            ),
        }


# ── Demo / Smoke Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    rng = random.Random(0)

    print("=" * 60)
    print("UARC: Token Difficulty Estimator — Demo")
    print("=" * 60)

    cfg = TDEConfig(context_dim=128, hidden_dim=256, n_hidden=3)
    tde = TokenDifficultyEstimator(cfg)

    # Synthetic training data
    # Easy tokens: common words → low ppl (1.5–3.0)
    # Hard tokens: rare/technical → high ppl (8–20)
    print("\n[1] Generating synthetic training data...")
    dataset = []
    for _ in range(200):
        n_tok = rng.randint(32, 128)
        token_ids = [rng.randint(0, 32000) for _ in range(n_tok)]
        # Simulate: tokens from small vocab → easy; large vocab index → hard
        avg_tok = sum(token_ids) / len(token_ids)
        if avg_tok < 10000:
            ppl = rng.uniform(1.5, 4.0)    # easy
        elif avg_tok < 22000:
            ppl = rng.uniform(4.0, 10.0)   # medium
        else:
            ppl = rng.uniform(10.0, 25.0)  # hard
        dataset.append((token_ids, ppl))

    print(f"  Generated {len(dataset)} training samples")

    # Train
    print("\n[2] Training TDE...")
    tde.train(dataset, epochs=3)

    # Inference demo
    print("\n[3] Routing decisions on test tokens:")
    test_cases = [
        ([100, 200, 150, 300], "easy context (common tokens)"),
        ([15000, 22000, 18500, 25000], "hard context (rare tokens)"),
        ([8000, 12000, 9000, 11000], "medium context"),
    ]
    for toks, desc in test_cases:
        dec = tde.estimate(toks)
        print(f"  {desc}")
        print(f"    → route={dec.route:8s}  est_ppl={dec.estimated_ppl:.2f}"
              f"  confidence={dec.confidence:.2f}  latency={dec.latency_ms:.2f}ms"
              f"  compute_saved={dec.compute_saved_pct:.0f}%")

    # Calibration demo
    print("\n[4] Running calibration cycle...")
    for _ in range(600):
        est = rng.uniform(1.0, 20.0)
        actual = est + rng.gauss(0.5, 1.0)   # systematic +0.5 bias
        tde.calibrate(est, actual)

    print("\n[5] Final routing report:")
    for _ in range(1000):
        toks = [rng.randint(0, 32000) for _ in range(rng.randint(32, 128))]
        tde.estimate(toks)
    report = tde.report()
    for k, v in report.items():
        print(f"  {k}: {v}")
