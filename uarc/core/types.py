"""
UARC Shared Types
==================
Enums, data classes, and type definitions shared across all modules.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────────

class Precision(Enum):
    """Quantization precision levels."""
    INT4 = 4
    INT8 = 8
    FP16 = 16
    FP32 = 32


class MemoryTier(IntEnum):
    """Three-tier memory hierarchy."""
    VRAM = 0
    RAM = 1
    NVME = 2


class RouteTarget(Enum):
    """TDE routing targets."""
    DRAFT = "draft"
    PARTIAL = "partial"
    FULL = "full"


class RequestPriority(Enum):
    """Inference request priority levels."""
    REALTIME = 0   # Interactive, <100ms SLA
    STANDARD = 1   # Normal, <2s SLA
    BATCH = 2      # Background, <60s SLA


class DeviceType(Enum):
    """Compute device target."""
    CPU = "cpu"
    GPU = "gpu"


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """Output of the Token Difficulty Estimator."""
    route: RouteTarget
    estimated_ppl: float
    confidence: float
    latency_ms: float
    compute_saved_pct: float


@dataclass
class InferenceRequest:
    """A single inference request submitted to the runtime."""
    request_id: str = ""
    prompt: str = ""
    token_ids: list[int] = field(default_factory=list)
    max_new_tokens: int = 256
    priority: RequestPriority = RequestPriority.STANDARD
    deadline_ts: float = field(default_factory=lambda: time.time() + 2.0)
    estimated_tokens: int = 100
    difficulty_score: float = 5.0
    prefix_hash: str = ""
    submitted_ts: float = field(default_factory=time.time)
    stream: bool = False

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())

    def sla_urgency(self) -> float:
        remaining = max(self.deadline_ts - time.time(), 1e-6)
        return 1.0 / remaining

    def priority_score(self, w_sla=3.0, w_short=1.0, w_batch=2.0,
                       batch_bonus=0.0) -> float:
        base = -int(self.priority.value)
        return (base
                + w_sla * self.sla_urgency()
                + w_short * (1.0 / max(self.estimated_tokens, 1))
                + w_batch * batch_bonus)


@dataclass
class InferenceResponse:
    """Response from UARC inference pipeline."""
    request_id: str = ""
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    route_taken: str = ""
    cache_hit: bool = False
    precision_plan_summary: dict = field(default_factory=dict)
    compute_saved_pct: float = 0.0


@dataclass
class Batch:
    """A batch of inference requests formed by ACS."""
    requests: list[InferenceRequest] = field(default_factory=list)
    batch_id: str = ""
    formed_ts: float = field(default_factory=time.time)

    @property
    def size(self):
        return len(self.requests)

    @property
    def total_tokens(self):
        return sum(r.estimated_tokens for r in self.requests)

    @property
    def avg_difficulty(self):
        if not self.requests:
            return 0.0
        return sum(r.difficulty_score for r in self.requests) / len(self.requests)


@dataclass
class UARCStats:
    """Runtime-level performance statistics."""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    draft_routes: int = 0
    partial_routes: int = 0
    full_routes: int = 0

    def to_dict(self) -> dict:
        n = max(self.total_requests, 1)
        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_latency_ms": round(self.total_latency_ms / n, 2),
            "avg_tokens_per_second": round(
                self.total_tokens_generated / max(self.total_latency_ms / 1000, 1e-6), 1),
            "cache_hit_rate": round(self.cache_hits / n, 3),
            "route_distribution": {
                "draft": self.draft_routes,
                "partial": self.partial_routes,
                "full": self.full_routes,
                "cache": self.cache_hits,
            },
        }
