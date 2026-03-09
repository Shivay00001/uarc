"""
UARC Module 2+4: AI Virtual Memory Manager + Predictive Layer Loader
=====================================================================
AI-VM: Three-tier unified memory with CLOCK-Pro eviction, KV-cache paging.
PLL: Lookahead prefetch scheduler with EMA timing.
"""
from __future__ import annotations

import heapq
import math
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from uarc.core.config import AIVMConfig, PLLConfig
from uarc.core.types import MemoryTier

# ── Constants ────────────────────────────────────────────────────────────────
TIER_NAMES = {MemoryTier.VRAM: "VRAM", MemoryTier.RAM: "RAM", MemoryTier.NVME: "NVMe"}
BW_TABLE = {
    (MemoryTier.VRAM, MemoryTier.RAM): 900.0, (MemoryTier.RAM, MemoryTier.VRAM): 900.0,
    (MemoryTier.RAM, MemoryTier.NVME): 12.0, (MemoryTier.NVME, MemoryTier.RAM): 7.0,
    (MemoryTier.VRAM, MemoryTier.NVME): 12.0, (MemoryTier.NVME, MemoryTier.VRAM): 7.0,
}
PAGE_SIZE_MB = 2; DECAY_TAU = 30.0
SCORE_ALPHA = 4.0; SCORE_BETA = 2.0; SCORE_GAMMA = 1.0
EVICT_THRESH = 0.5; PRESSURE_HI = 0.85; MIN_FREE_PCT = 0.10

# ── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class _TierConfig:
    tier: MemoryTier; capacity_mb: float; used_mb: float = 0.0
    @property
    def free_mb(self): return self.capacity_mb - self.used_mb
    @property
    def pressure(self): return self.used_mb / self.capacity_mb if self.capacity_mb > 0 else 0.0

@dataclass
class _AIPage:
    page_id: str; size_mb: float; tier: MemoryTier; data_type: str = "weight"
    layer_id: Optional[int] = None; seq_id: Optional[str] = None
    access_count: int = 0; last_access_ts: float = field(default_factory=time.time)
    created_ts: float = field(default_factory=time.time)
    pinned: bool = False; dirty: bool = False; referenced: bool = False
    transfer_in_progress: bool = False; target_tier: Optional[MemoryTier] = None
    def touch(self):
        self.access_count += 1; self.last_access_ts = time.time(); self.referenced = True
    def age_seconds(self) -> float: return time.time() - self.last_access_ts
    def score(self) -> float:
        return (SCORE_ALPHA * math.log(1 + self.access_count)
                + SCORE_BETA * math.exp(-self.age_seconds() / DECAY_TAU)
                - SCORE_GAMMA * float(self.tier))
    def transfer_time_ms(self, dst: MemoryTier) -> float:
        if self.tier == dst: return 0.0
        return (self.size_mb / 1024.0) / BW_TABLE.get((self.tier, dst), 1.0) * 1000.0

@dataclass
class _KVBlock:
    block_id: int; seq_id: str; logical_idx: int; page_id: str
    tokens: int = 16; ref_count: int = 0
    def is_shared(self) -> bool: return self.ref_count > 1

# ── CLOCK-Pro Eviction ───────────────────────────────────────────────────────
class _CLOCKPro:
    def __init__(self):
        self._hot: list[str] = []; self._cold: list[str] = []
    def add(self, pid: str, is_hot=False):
        (self._hot if is_hot else self._cold).append(pid)
    def remove(self, pid: str):
        if pid in self._hot: self._hot.remove(pid)
        if pid in self._cold: self._cold.remove(pid)
    def promote_to_hot(self, pid: str):
        if pid in self._cold: self._cold.remove(pid)
        if pid not in self._hot: self._hot.append(pid)
    def scan_cold(self, pages: dict, n: int) -> list[str]:
        evict = []
        for pid in self._cold[:]:
            if len(evict) >= n: break
            if pid not in pages: self._cold.remove(pid); continue
            p = pages[pid]
            if p.pinned or p.transfer_in_progress: continue
            if p.referenced: p.referenced = False; self.promote_to_hot(pid)
            elif p.score() < EVICT_THRESH: evict.append(pid)
        return evict
    def scan_hot(self, pages: dict, thr=1.5):
        demote = [pid for pid in self._hot[:] if pid in pages and pages[pid].score() < thr]
        for pid in demote: self._hot.remove(pid); self._cold.append(pid)

# ── KV Page Table ────────────────────────────────────────────────────────────
class _KVPageTable:
    def __init__(self):
        self._logical: dict[tuple, int] = {}; self._blocks: dict[int, _KVBlock] = {}; self._next = 0
    def allocate_block(self, seq_id, idx, page_id) -> _KVBlock:
        bid = self._next; self._next += 1
        blk = _KVBlock(block_id=bid, seq_id=seq_id, logical_idx=idx, page_id=page_id, ref_count=1)
        self._blocks[bid] = blk; self._logical[(seq_id, idx)] = bid; return blk
    def lookup(self, seq_id, idx) -> Optional[_KVBlock]:
        bid = self._logical.get((seq_id, idx)); return self._blocks.get(bid) if bid is not None else None
    def share_prefix(self, src, dst, n) -> int:
        shared = 0
        for i in range(n):
            bid = self._logical.get((src, i))
            if bid is None: break
            self._blocks[bid].ref_count += 1; self._logical[(dst, i)] = bid; shared += 1
        return shared
    def free_sequence(self, seq_id) -> list[str]:
        keys = [k for k in self._logical if k[0] == seq_id]; release = []
        for k in keys:
            bid = self._logical.pop(k); blk = self._blocks.get(bid)
            if blk:
                blk.ref_count -= 1
                if blk.ref_count <= 0: release.append(blk.page_id); del self._blocks[bid]
        return release

# ── AI Virtual Memory Manager ────────────────────────────────────────────────
class AIVirtualMemoryManager:
    """Unified three-tier memory manager with CLOCK-Pro eviction and KV-cache paging."""
    def __init__(self, cfg: AIVMConfig | None = None):
        cfg = cfg or AIVMConfig()
        self._tiers = {
            MemoryTier.VRAM: _TierConfig(MemoryTier.VRAM, cfg.vram_mb),
            MemoryTier.RAM: _TierConfig(MemoryTier.RAM, cfg.ram_mb),
            MemoryTier.NVME: _TierConfig(MemoryTier.NVME, cfg.nvme_mb),
        }
        self._pages: dict[str, _AIPage] = {}
        self._clock = _CLOCKPro(); self._kv = _KVPageTable()
        self._lock = threading.Lock()
        self._pf_queue: list = []
        self._pf_thread = threading.Thread(target=self._pf_worker, daemon=True)
        self._pf_thread.start()
        self.stats = defaultdict(int)

    def allocate(self, page_id, size_mb=2.0, data_type="weight", layer_id=None,
                 seq_id=None, preferred_tier=MemoryTier.NVME, pinned=False) -> _AIPage:
        with self._lock:
            for tier in [preferred_tier, MemoryTier.RAM, MemoryTier.NVME]:
                t = self._tiers[tier]
                if t.free_mb >= size_mb:
                    page = _AIPage(page_id=page_id, size_mb=size_mb, tier=tier,
                                   data_type=data_type, layer_id=layer_id, seq_id=seq_id, pinned=pinned)
                    self._pages[page_id] = page; t.used_mb += size_mb
                    self._clock.add(page_id, is_hot=(tier == MemoryTier.VRAM))
                    self.stats[f"alloc_{TIER_NAMES[tier]}"] += 1; return page
            raise MemoryError(f"Cannot allocate {size_mb}MB: all tiers full")

    def locate(self, page_id) -> Optional[MemoryTier]:
        p = self._pages.get(page_id); return p.tier if p else None

    def access(self, page_id) -> Optional[_AIPage]:
        p = self._pages.get(page_id)
        if p is None: return None
        p.touch(); self.stats["access_total"] += 1
        k = {MemoryTier.VRAM: "tier0_hits", MemoryTier.RAM: "tier1_hits"}.get(p.tier, "tier2_hits")
        self.stats[k] += 1; return p

    def promote(self, page_id, target_tier=MemoryTier.VRAM) -> float:
        p = self._pages.get(page_id)
        if p is None or p.tier <= target_tier: return 0.0
        src = p.tier; dst_t = self._tiers[target_tier]
        if dst_t.free_mb < p.size_mb:
            if self._evict(target_tier, p.size_mb) < p.size_mb: return -1.0
        ms = p.transfer_time_ms(target_tier)
        with self._lock: p.transfer_in_progress = True
        time.sleep(min(ms / 1e6, 0.001))
        with self._lock:
            self._tiers[src].used_mb -= p.size_mb; dst_t.used_mb += p.size_mb
            p.tier = target_tier; p.transfer_in_progress = False
            self._clock.promote_to_hot(page_id)
        self.stats[f"promote_to_{TIER_NAMES[target_tier]}"] += 1; return ms

    def demote(self, page_id, target_tier=MemoryTier.RAM) -> float:
        p = self._pages.get(page_id)
        if p is None or p.tier >= target_tier or p.pinned: return 0.0
        src = p.tier; ms = p.transfer_time_ms(target_tier)
        time.sleep(min(ms / 1e7, 0.0005))
        with self._lock:
            self._tiers[src].used_mb -= p.size_mb; self._tiers[target_tier].used_mb += p.size_mb
            p.tier = target_tier; p.referenced = False
        self.stats[f"demote_to_{TIER_NAMES[target_tier]}"] += 1; return ms

    def free(self, page_id):
        with self._lock:
            p = self._pages.pop(page_id, None)
            if p: self._tiers[p.tier].used_mb -= p.size_mb; self._clock.remove(page_id)

    # KV-Cache API
    def kv_allocate(self, seq_id, block_idx) -> _KVBlock:
        pid = f"kv_{seq_id}_{block_idx}_{int(time.time()*1e6)}"
        self.allocate(pid, 2.0, "kv_cache", seq_id=seq_id, preferred_tier=MemoryTier.VRAM)
        return self._kv.allocate_block(seq_id, block_idx, pid)

    def kv_lookup(self, seq_id, idx):
        return self._kv.lookup(seq_id, idx)

    def kv_share_prefix(self, src, dst, n) -> int:
        s = self._kv.share_prefix(src, dst, n); self.stats["prefix_shares"] += 1; return s

    def kv_free(self, seq_id):
        for pid in self._kv.free_sequence(seq_id): self.free(pid)

    def schedule_prefetch(self, page_id, target_tier, priority=0.0):
        heapq.heappush(self._pf_queue, (priority, page_id, target_tier))

    def _pf_worker(self):
        while True:
            if self._pf_queue:
                _, pid, tier = heapq.heappop(self._pf_queue); self.promote(pid, tier)
            else: time.sleep(0.001)

    def _evict(self, tier, need_mb) -> float:
        freed = 0.0
        for pid in self._clock.scan_cold(self._pages, math.ceil(need_mb / PAGE_SIZE_MB) + 1):
            p = self._pages.get(pid)
            if p is None or p.tier != tier or p.pinned: continue
            nxt = MemoryTier(p.tier + 1) if p.tier < MemoryTier.NVME else None
            if nxt: self.demote(pid, nxt); freed += p.size_mb
            if freed >= need_mb: break
        return freed

    def run_eviction_cycle(self):
        for tier in [MemoryTier.VRAM, MemoryTier.RAM]:
            t = self._tiers[tier]
            if t.pressure > PRESSURE_HI:
                need = t.capacity_mb * MIN_FREE_PCT - t.free_mb
                if need > 0: self._evict(tier, need); self.stats[f"evict_{TIER_NAMES[tier]}"] += 1
        self._clock.scan_hot(self._pages)

    def status(self) -> dict:
        r = {}
        for tier, t in self._tiers.items():
            r[TIER_NAMES[tier]] = {"capacity_mb": round(t.capacity_mb, 1), "used_mb": round(t.used_mb, 1),
                                    "free_mb": round(t.free_mb, 1), "pressure": round(t.pressure, 3)}
        r["total_pages"] = len(self._pages); r["stats"] = dict(self.stats)
        th = sum(self.stats.get(k, 0) for k in ("tier0_hits", "tier1_hits", "tier2_hits"))
        if th > 0: r["tier0_hit_rate"] = round(self.stats.get("tier0_hits", 0) / th, 3)
        return r


# ══════════════════════════════════════════════════════════════════════════════
# Predictive Layer Loader (PLL)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PrefetchOrder:
    layer_id: int; urgency: float; target_tier: MemoryTier; estimated_load_ms: float

class PredictiveLayerLoader:
    """Lookahead prefetch scheduler with EMA timing and deadline analysis."""
    def __init__(self, cfg: PLLConfig | None = None, n_layers=32,
                 layer_sizes_mb=None, aivm: AIVirtualMemoryManager | None = None):
        self.cfg = cfg or PLLConfig()
        self.n_layers = n_layers; self.layer_sizes = layer_sizes_mb or [2.0]*n_layers
        self.aivm = aivm
        self._exec_ema = [50.0]*n_layers; self._load_ema = [200.0]*n_layers
        self._alpha = 0.9; self._current_layer = 0
        self._t_layer_start = time.perf_counter()
        self._issued: set[int] = set()
        self._pf_queue: list = []; self._lock = threading.Lock()
        self.on_prefetch: callable = lambda order: None
        self._n_issued = 0; self._n_hit = 0

    def on_layer_start(self, layer_id):
        self._current_layer = layer_id; self._t_layer_start = time.perf_counter()

    def on_layer_complete(self, layer_id):
        t_exec = (time.perf_counter() - self._t_layer_start) * 1000
        self._exec_ema[layer_id] = self._alpha * self._exec_ema[layer_id] + (1-self._alpha) * t_exec
        self._schedule(layer_id)

    def _schedule(self, completed):
        t_now = time.perf_counter() * 1000; cum = 0.0
        for off in range(1, self.cfg.lookahead_k + 1):
            tgt = completed + off
            if tgt >= self.n_layers: break
            for mid in range(completed + 1, tgt): cum += self._exec_ema[mid]
            slack = (t_now + cum) - self._load_ema[tgt] - t_now
            if slack > self.cfg.slack_ms and tgt not in self._issued:
                tier = MemoryTier.VRAM if off <= 2 else MemoryTier.RAM
                order = PrefetchOrder(tgt, slack, tier, self._load_ema[tgt])
                with self._lock:
                    heapq.heappush(self._pf_queue, (slack, tgt, order)); self._issued.add(tgt)
                self._n_issued += 1; self.on_prefetch(order)

    def pop_prefetch(self):
        with self._lock:
            if self._pf_queue: _, _, o = heapq.heappop(self._pf_queue); return o
        return None

    def reset(self):
        self._current_layer = 0; self._issued.clear()
        with self._lock: self._pf_queue.clear()

    def stats_report(self) -> dict:
        return {"prefetch_issued": self._n_issued, "hit_rate": round(self._n_hit / max(self._n_issued, 1), 3),
                "avg_exec_ms": round(sum(self._exec_ema) / self.n_layers, 2),
                "avg_load_ms": round(sum(self._load_ema) / self.n_layers, 2)}
