"""
UARC Module 2: AI Virtual Memory Manager (AI-VM)
=================================================
Three-tier unified memory: VRAM (Tier 0) → RAM (Tier 1) → NVMe (Tier 2)

Mathematical Foundation:
  Page Score Function:
    S(p) = α·log(1 + f(p)) + β·exp(-age(p)/τ) - γ·cost(tier(p))

    where:
      f(p)     = access frequency count
      age(p)   = seconds since last access
      τ        = decay time constant (30s)
      cost(t)  = promotion cost in bytes/s for tier t
      α=4, β=2, γ=1

  Eviction Threshold:
    Evict page p if S(p) < S_threshold AND tier_pressure > 0.85

  DMA Transfer Time Model:
    t_transfer(p, Δtier) = size(p) / BW(src_tier → dst_tier)
    BW(0→1) = 900 GB/s (NVLink)   BW(1→0) = 900 GB/s
    BW(1→2) = 12 GB/s  (NVMe)     BW(2→1) = 7 GB/s
"""

import time
import math
import heapq
import threading
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from enum import IntEnum

# ── Constants & Enums ────────────────────────────────────────────────────────

class Tier(IntEnum):
    VRAM = 0   # GPU HBM / VRAM
    RAM  = 1   # System DRAM
    NVME = 2   # NVMe SSD

TIER_NAMES = {Tier.VRAM: "VRAM", Tier.RAM: "RAM", Tier.NVME: "NVMe"}

# Simulated bandwidth table (GB/s)
BW_TABLE = {
    (Tier.VRAM, Tier.RAM):  900.0,
    (Tier.RAM,  Tier.VRAM): 900.0,
    (Tier.RAM,  Tier.NVME):  12.0,
    (Tier.NVME, Tier.RAM):    7.0,
    (Tier.VRAM, Tier.NVME):  12.0,   # via RAM
    (Tier.NVME, Tier.VRAM):   7.0,
}

PAGE_SIZE_MB = 2          # 2 MB per AI page
DECAY_TAU    = 30.0       # seconds
SCORE_ALPHA  = 4.0        # frequency weight
SCORE_BETA   = 2.0        # recency weight
SCORE_GAMMA  = 1.0        # promotion cost penalty
EVICT_THRESH = 0.5        # pages below this score are evicted
PRESSURE_HI  = 0.85       # tier pressure threshold to trigger eviction
MIN_FREE_PCT = 0.10       # keep at least 10% free in each tier

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class TierConfig:
    tier: Tier
    capacity_mb: float    # Total capacity
    used_mb: float = 0.0

    @property
    def free_mb(self):
        return self.capacity_mb - self.used_mb

    @property
    def pressure(self):
        return self.used_mb / self.capacity_mb if self.capacity_mb > 0 else 0.0

@dataclass
class AIPage:
    """A 2MB allocation unit in the AI-VM address space."""
    page_id: str
    size_mb: float
    tier: Tier
    data_type: str          # "weight", "kv_cache", "activation"
    layer_id: Optional[int] # For weight pages
    seq_id: Optional[str]   # For KV-cache pages

    # Access tracking
    access_count: int = 0
    last_access_ts: float = field(default_factory=time.time)
    created_ts: float = field(default_factory=time.time)

    # State flags
    pinned: bool = False    # Cannot be evicted (e.g., embedding table)
    dirty: bool = False     # Modified since last sync
    referenced: bool = False  # CLOCK-Pro hand flag

    # Transfer state
    transfer_in_progress: bool = False
    target_tier: Optional[Tier] = None

    def touch(self):
        """Record an access."""
        self.access_count += 1
        self.last_access_ts = time.time()
        self.referenced = True

    def age_seconds(self) -> float:
        return time.time() - self.last_access_ts

    def score(self) -> float:
        """
        AI-aware page score.
        Higher = more valuable = less likely to be evicted.

        S(p) = α·log(1+f) + β·exp(-age/τ) - γ·tier_cost
        """
        freq_score    = SCORE_ALPHA * math.log(1 + self.access_count)
        recency_score = SCORE_BETA  * math.exp(-self.age_seconds() / DECAY_TAU)
        tier_cost     = SCORE_GAMMA * float(self.tier)   # higher tier = more cost
        return freq_score + recency_score - tier_cost

    def transfer_time_ms(self, dst_tier: Tier) -> float:
        """Estimated DMA transfer time in milliseconds."""
        if self.tier == dst_tier:
            return 0.0
        bw = BW_TABLE.get((self.tier, dst_tier), 1.0)  # GB/s
        return (self.size_mb / 1024.0) / bw * 1000.0   # ms


@dataclass
class KVBlock:
    """16-token KV-cache block (vLLM-style paging)."""
    block_id: int
    seq_id: str
    logical_idx: int       # Position in logical sequence
    page_id: str           # Backing AI page
    tokens: int = 16
    ref_count: int = 0     # For prefix sharing (copy-on-write)

    def is_shared(self) -> bool:
        return self.ref_count > 1


# ── CLOCK-Pro Eviction ────────────────────────────────────────────────────────

class CLOCKProEviction:
    """
    CLOCK-Pro: Two-hand clock with hot/cold distinction.

    Hot ring: recently promoted pages (referenced recently)
    Cold ring: candidate pages for eviction

    Algorithm:
      cold_hand scans cold ring:
        if referenced: move to hot ring
        else if score < threshold: evict
      hot_hand scans hot ring:
        if score drops below HOT_THRESHOLD: demote to cold ring

    Advantage over LRU: O(1) amortised eviction,
    correctly handles one-time-access pages (cold) without
    polluting the hot set.
    """
    def __init__(self):
        self._hot  = []    # list of page_ids (hot ring)
        self._cold = []    # list of page_ids (cold ring)
        self._hot_idx  = 0
        self._cold_idx = 0
        HOT_THRESHOLD  = EVICT_THRESH * 1.5

    def add(self, page_id: str, is_hot: bool = False):
        if is_hot:
            self._hot.append(page_id)
        else:
            self._cold.append(page_id)

    def remove(self, page_id: str):
        if page_id in self._hot:
            self._hot.remove(page_id)
        if page_id in self._cold:
            self._cold.remove(page_id)

    def promote_to_hot(self, page_id: str):
        if page_id in self._cold:
            self._cold.remove(page_id)
        if page_id not in self._hot:
            self._hot.append(page_id)

    def scan_cold(self, pages: dict, n_evict: int) -> list:
        """
        Scan cold ring and return up to n_evict page_ids to evict.
        """
        evict_list = []
        scanned = 0
        ring = self._cold[:]   # snapshot

        for pid in ring:
            if len(evict_list) >= n_evict:
                break
            if pid not in pages:
                self._cold.remove(pid)
                continue
            page = pages[pid]
            if page.pinned or page.transfer_in_progress:
                scanned += 1
                continue
            if page.referenced:
                page.referenced = False
                self.promote_to_hot(pid)
            elif page.score() < EVICT_THRESH:
                evict_list.append(pid)
            scanned += 1

        return evict_list

    def scan_hot(self, pages: dict, hot_threshold: float = 1.5) -> list:
        """Demote cool hot pages to cold ring."""
        demote = []
        for pid in self._hot[:]:
            if pid not in pages:
                self._hot.remove(pid)
                continue
            page = pages[pid]
            if page.score() < hot_threshold:
                demote.append(pid)
        for pid in demote:
            self._hot.remove(pid)
            self._cold.append(pid)
        return demote


# ── KV-Cache Page Table ───────────────────────────────────────────────────────

class KVPageTable:
    """
    Two-level page table for KV-cache:
      Logical:  (seq_id, block_idx) → block_id
      Physical: block_id → AIPage

    Supports copy-on-write prefix sharing:
      Multiple sequences can share the same physical block
      until one writes → triggers block copy.
    """
    def __init__(self):
        self._logical: dict[tuple, int] = {}   # (seq_id, block_idx) → block_id
        self._blocks:  dict[int, KVBlock] = {}
        self._next_block_id = 0

    def allocate_block(self, seq_id: str, logical_idx: int, page_id: str) -> KVBlock:
        bid = self._next_block_id
        self._next_block_id += 1
        blk = KVBlock(block_id=bid, seq_id=seq_id,
                      logical_idx=logical_idx, page_id=page_id, ref_count=1)
        self._blocks[bid] = blk
        self._logical[(seq_id, logical_idx)] = bid
        return blk

    def lookup(self, seq_id: str, logical_idx: int) -> Optional[KVBlock]:
        bid = self._logical.get((seq_id, logical_idx))
        return self._blocks.get(bid) if bid is not None else None

    def share_prefix(self, src_seq: str, dst_seq: str, n_blocks: int):
        """
        Copy-on-write prefix sharing: dst_seq shares first n_blocks
        of src_seq without copying data. Ref counts incremented.
        """
        for i in range(n_blocks):
            bid = self._logical.get((src_seq, i))
            if bid is None:
                break
            blk = self._blocks[bid]
            blk.ref_count += 1
            self._logical[(dst_seq, i)] = bid

    def free_sequence(self, seq_id: str) -> list:
        """Free all blocks for seq_id. Returns page_ids to release."""
        keys = [k for k in self._logical if k[0] == seq_id]
        release_pages = []
        for k in keys:
            bid = self._logical.pop(k)
            blk = self._blocks.get(bid)
            if blk:
                blk.ref_count -= 1
                if blk.ref_count <= 0:
                    release_pages.append(blk.page_id)
                    del self._blocks[bid]
        return release_pages

    def cow_on_write(self, seq_id: str, block_idx: int) -> Optional[str]:
        """
        Copy-on-write: if block is shared, allocate a new page and copy.
        Returns new page_id if copy was triggered, else None.
        """
        bid = self._logical.get((seq_id, block_idx))
        if bid is None:
            return None
        blk = self._blocks[bid]
        if blk.ref_count > 1:
            blk.ref_count -= 1
            new_page_id = f"kv_{seq_id}_{block_idx}_cow_{int(time.time()*1e6)}"
            new_blk = KVBlock(
                block_id=self._next_block_id,
                seq_id=seq_id, logical_idx=block_idx,
                page_id=new_page_id, ref_count=1
            )
            self._next_block_id += 1
            self._blocks[new_blk.block_id] = new_blk
            self._logical[(seq_id, block_idx)] = new_blk.block_id
            return new_page_id
        return None


# ── AI Virtual Memory Manager ─────────────────────────────────────────────────

class AIVirtualMemoryManager:
    """
    Unified three-tier memory manager for AI inference.

    Responsibilities:
      1. Allocate AI pages across tiers
      2. Track all page metadata (access patterns, scores)
      3. Manage KV-cache blocks with copy-on-write sharing
      4. Run CLOCK-Pro eviction when pressure > threshold
      5. Schedule async DMA promotions/demotions
      6. Provide locate() / promote() / demote() primitives

    Simulates DMA timing with sleep() for fidelity.
    In production: replace with cudaMemcpyAsync / io_uring.
    """

    def __init__(self,
                 vram_mb: float = 8192,    # 8 GB VRAM
                 ram_mb:  float = 32768,   # 32 GB RAM
                 nvme_mb: float = 524288): # 512 GB NVMe
        self._tiers = {
            Tier.VRAM: TierConfig(Tier.VRAM, vram_mb),
            Tier.RAM:  TierConfig(Tier.RAM,  ram_mb),
            Tier.NVME: TierConfig(Tier.NVME, nvme_mb),
        }
        self._pages:  dict[str, AIPage] = {}
        self._clock   = CLOCKProEviction()
        self._kv_table = KVPageTable()
        self._lock    = threading.Lock()

        # Async prefetch queue: (priority, page_id, target_tier)
        self._prefetch_queue: list = []
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

        # Stats
        self.stats = defaultdict(int)

    # ── Public API ─────────────────────────────────────────────────────────────

    def allocate(self, page_id: str, size_mb: float,
                 data_type: str = "weight",
                 layer_id: int = None, seq_id: str = None,
                 preferred_tier: Tier = Tier.NVME,
                 pinned: bool = False) -> AIPage:
        """
        Allocate a new AI page. Tries preferred_tier first,
        then falls back to lower tiers if capacity exhausted.
        """
        with self._lock:
            for tier in [preferred_tier, Tier.RAM, Tier.NVME]:
                t = self._tiers[tier]
                if t.free_mb >= size_mb:
                    page = AIPage(
                        page_id=page_id, size_mb=size_mb,
                        tier=tier, data_type=data_type,
                        layer_id=layer_id, seq_id=seq_id,
                        pinned=pinned,
                    )
                    self._pages[page_id] = page
                    t.used_mb += size_mb
                    self._clock.add(page_id, is_hot=(tier == Tier.VRAM))
                    self.stats[f"alloc_{TIER_NAMES[tier]}"] += 1
                    return page

            raise MemoryError(f"Cannot allocate {size_mb}MB: all tiers full")

    def locate(self, page_id: str) -> Optional[Tier]:
        """Return current tier of a page, or None if not found."""
        page = self._pages.get(page_id)
        return page.tier if page else None

    def access(self, page_id: str) -> Optional[AIPage]:
        """
        Touch a page (record access). Auto-promotes to Tier 0 if needed.
        Returns the page, or None if not found.
        """
        page = self._pages.get(page_id)
        if page is None:
            return None
        page.touch()
        self.stats["access_total"] += 1
        if page.tier == Tier.VRAM:
            self.stats["tier0_hits"] += 1
        elif page.tier == Tier.RAM:
            self.stats["tier1_hits"] += 1
        else:
            self.stats["tier2_hits"] += 1
        return page

    def promote(self, page_id: str, target_tier: Tier = Tier.VRAM) -> float:
        """
        Move page to a higher (faster) tier. Simulates DMA.
        Returns actual transfer time in ms.
        """
        page = self._pages.get(page_id)
        if page is None or page.tier <= target_tier:
            return 0.0

        src_tier = page.tier
        # Check capacity
        t_dst = self._tiers[target_tier]
        if t_dst.free_mb < page.size_mb:
            # Need to evict first
            freed = self._evict(target_tier, need_mb=page.size_mb)
            if freed < page.size_mb:
                return -1.0   # Failed: not enough space even after eviction

        transfer_ms = page.transfer_time_ms(target_tier)

        # Simulate async DMA (in production: cudaMemcpyAsync / io_uring)
        with self._lock:
            page.transfer_in_progress = True
            page.target_tier = target_tier

        # Simulated transfer (wall-clock: very fast in simulation)
        time.sleep(min(transfer_ms / 1e6, 0.001))  # cap at 1ms sim

        with self._lock:
            self._tiers[src_tier].used_mb  -= page.size_mb
            self._tiers[target_tier].used_mb += page.size_mb
            page.tier = target_tier
            page.transfer_in_progress = False
            page.target_tier = None
            self._clock.promote_to_hot(page_id)

        self.stats[f"promote_to_{TIER_NAMES[target_tier]}"] += 1
        return transfer_ms

    def demote(self, page_id: str, target_tier: Tier = Tier.RAM) -> float:
        """Move page to a slower tier. Returns transfer time ms."""
        page = self._pages.get(page_id)
        if page is None or page.tier >= target_tier or page.pinned:
            return 0.0

        src_tier = page.tier
        t_dst = self._tiers[target_tier]
        t_dst_obj = self._tiers[target_tier]

        transfer_ms = page.transfer_time_ms(target_tier)
        time.sleep(min(transfer_ms / 1e7, 0.0005))

        with self._lock:
            self._tiers[src_tier].used_mb  -= page.size_mb
            t_dst_obj.used_mb += page.size_mb
            page.tier = target_tier
            page.referenced = False

        self.stats[f"demote_to_{TIER_NAMES[target_tier]}"] += 1
        return transfer_ms

    def free(self, page_id: str):
        """Release a page entirely (all tiers)."""
        with self._lock:
            page = self._pages.pop(page_id, None)
            if page:
                self._tiers[page.tier].used_mb -= page.size_mb
                self._clock.remove(page_id)

    # ── KV-Cache API ──────────────────────────────────────────────────────────

    def kv_allocate_block(self, seq_id: str, block_idx: int) -> KVBlock:
        pid = f"kv_{seq_id}_{block_idx}_{int(time.time()*1e6)}"
        # KV blocks are 16 tokens × ~128KB per head → ~2MB per block at 70B
        self.allocate(pid, size_mb=2.0, data_type="kv_cache",
                      seq_id=seq_id, preferred_tier=Tier.VRAM)
        return self._kv_table.allocate_block(seq_id, block_idx, pid)

    def kv_lookup(self, seq_id: str, block_idx: int) -> Optional[KVBlock]:
        return self._kv_table.lookup(seq_id, block_idx)

    def kv_share_prefix(self, src_seq: str, dst_seq: str, n_blocks: int):
        """Share first n_blocks of src_seq with dst_seq (zero-copy)."""
        self._kv_table.share_prefix(src_seq, dst_seq, n_blocks)
        self.stats["prefix_shares"] += 1

    def kv_free_sequence(self, seq_id: str):
        page_ids = self._kv_table.free_sequence(seq_id)
        for pid in page_ids:
            self.free(pid)

    # ── Prefetch Scheduling ───────────────────────────────────────────────────

    def schedule_prefetch(self, page_id: str, target_tier: Tier,
                          priority: float = 0.0):
        """
        Schedule async prefetch. priority = urgency (lower = sooner).
        In production: backed by io_uring or CUDA streams.
        """
        heapq.heappush(self._prefetch_queue, (priority, page_id, target_tier))

    def _prefetch_worker(self):
        """Background thread: drains prefetch queue."""
        while True:
            if self._prefetch_queue:
                _, page_id, target_tier = heapq.heappop(self._prefetch_queue)
                self.promote(page_id, target_tier)
            else:
                time.sleep(0.001)

    # ── Eviction Engine ───────────────────────────────────────────────────────

    def _evict(self, tier: Tier, need_mb: float) -> float:
        """
        Evict pages from tier until need_mb is available.
        Uses CLOCK-Pro to select victims.
        Returns MB freed.
        """
        freed_mb = 0.0
        tier_obj = self._tiers[tier]

        # How many pages to evict
        n_pages = math.ceil(need_mb / PAGE_SIZE_MB) + 1
        victims = self._clock.scan_cold(self._pages, n_pages)

        for pid in victims:
            page = self._pages.get(pid)
            if page is None or page.tier != tier or page.pinned:
                continue
            # Demote to next tier
            next_tier = Tier(page.tier + 1) if page.tier < Tier.NVME else None
            if next_tier is not None:
                self.demote(pid, next_tier)
                freed_mb += page.size_mb

            if freed_mb >= need_mb:
                break

        return freed_mb

    def run_eviction_cycle(self):
        """
        Proactive eviction: called periodically (every 100ms in production).
        Evicts from all tiers exceeding pressure threshold.
        """
        for tier in [Tier.VRAM, Tier.RAM]:
            t = self._tiers[tier]
            if t.pressure > PRESSURE_HI:
                target_free = t.capacity_mb * MIN_FREE_PCT
                need_mb = target_free - t.free_mb
                if need_mb > 0:
                    freed = self._evict(tier, need_mb)
                    self.stats[f"evict_cycles_{TIER_NAMES[tier]}"] += 1

        # Hot→Cold demotion
        self._clock.scan_hot(self._pages)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def status(self) -> dict:
        result = {}
        total_pages = len(self._pages)
        for tier, t in self._tiers.items():
            result[TIER_NAMES[tier]] = {
                "capacity_mb": round(t.capacity_mb, 1),
                "used_mb":     round(t.used_mb, 1),
                "free_mb":     round(t.free_mb, 1),
                "pressure":    round(t.pressure, 3),
            }
        result["total_pages"] = total_pages
        result["stats"] = dict(self.stats)

        # Hit rate
        total_hits = (self.stats.get("tier0_hits", 0) +
                      self.stats.get("tier1_hits", 0) +
                      self.stats.get("tier2_hits", 0))
        if total_hits > 0:
            result["tier0_hit_rate"] = round(
                self.stats.get("tier0_hits", 0) / total_hits, 3)
        return result


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    rng = random.Random(42)

    print("=" * 60)
    print("UARC: AI Virtual Memory Manager — Demo")
    print("=" * 60)

    vm = AIVirtualMemoryManager(
        vram_mb=1024,   # 1 GB VRAM (tiny for demo)
        ram_mb=4096,    # 4 GB RAM
        nvme_mb=16384,  # 16 GB NVMe
    )

    # [1] Load model layers (80 layers × 2MB each on NVMe)
    print("\n[1] Allocating model weight pages on NVMe...")
    layer_pages = []
    for i in range(80):
        pid = f"layer_{i}_weights"
        vm.allocate(pid, size_mb=2.0, data_type="weight",
                    layer_id=i, preferred_tier=Tier.NVME)
        layer_pages.append(pid)
    print(f"  Allocated 80 layers (160MB) on NVMe")

    # [2] Promote first 4 layers to VRAM (prefetch)
    print("\n[2] Prefetching layers 0-3 to VRAM...")
    for i in range(4):
        ms = vm.promote(f"layer_{i}_weights", Tier.VRAM)
        print(f"  layer_{i} promoted: {ms:.3f}ms simulated transfer")

    # [3] KV-cache blocks for 3 sequences
    print("\n[3] Allocating KV-cache blocks...")
    for seq in ["seq_A", "seq_B"]:
        for blk in range(8):
            vm.kv_allocate_block(seq, blk)
    print("  seq_A and seq_B: 8 KV blocks each (16MB)")

    # [4] Prefix sharing
    print("\n[4] Prefix sharing: seq_C shares seq_A's first 4 blocks...")
    vm.kv_share_prefix("seq_A", "seq_C", n_blocks=4)

    # [5] Access pattern simulation
    print("\n[5] Simulating access patterns...")
    for _ in range(200):
        lid = rng.choices(range(80), weights=[
            20 if i < 10 else (5 if i < 40 else 1)
            for i in range(80)
        ])[0]
        vm.access(f"layer_{lid}_weights")

    # [6] Run eviction cycle
    print("\n[6] Running eviction cycle...")
    vm.run_eviction_cycle()

    # [7] Status report
    print("\n[7] Memory status:")
    status = vm.status()
    for tier_name in ["VRAM", "RAM", "NVMe"]:
        t = status[tier_name]
        bar_len = 30
        fill = int(t["pressure"] * bar_len)
        bar = "█"*fill + "░"*(bar_len-fill)
        print(f"  {tier_name:6s} [{bar}] {t['pressure']*100:.1f}%  "
              f"{t['used_mb']:.0f}/{t['capacity_mb']:.0f} MB")

    print(f"\n  Total pages tracked: {status['total_pages']}")
    tier0_rate = status.get('tier0_hit_rate', 0)
    print(f"  Tier-0 hit rate: {tier0_rate:.1%}")
    print(f"  Prefix shares: {vm.stats.get('prefix_shares', 0)}")
