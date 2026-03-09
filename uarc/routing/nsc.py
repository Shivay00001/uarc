"""
UARC Module 6: Neural Semantic Cache (NSC)
============================================
Embedding-based semantic cache for LLM inference deduplication.

Mathematical Foundation:
  Embed prompt p → vector e(p) ∈ ℝ^dim via bi-encoder.
  Cache hit: cosine_similarity(e(query), e(cached)) > θ
  HNSW ANN search: O(log N) expected for N cached items.
  Adaptive threshold: θ ← θ + η × (fpr - fpr_target)
"""
from __future__ import annotations

import hashlib
import heapq
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from uarc.core.config import NSCConfig


# ── Similarity Functions ─────────────────────────────────────────────────────

def _cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    denom = na * nb
    return dot / denom if denom > 1e-9 else 0.0


# ── HNSW Index ───────────────────────────────────────────────────────────────

class _HNSWIndex:
    """
    Simplified HNSW (Hierarchical Navigable Small World) index.
    Single-layer NSW for clarity; preserves O(log N) search behaviour.
    Production: replace with faiss.IndexHNSWFlat.
    """

    def __init__(self, dim: int, M: int = 16, ef: int = 50):
        self.dim = dim
        self.M = M
        self.ef = ef
        self._vectors: list[list] = []
        self._ids: list[str] = []
        self._graph: list[list] = []

    def __len__(self):
        return len(self._vectors)

    def add(self, vector: list, entry_id: str) -> int:
        idx = len(self._vectors)
        self._vectors.append(vector)
        self._ids.append(entry_id)
        self._graph.append([])

        if idx == 0:
            return idx

        neighbours = self._greedy_search(vector, k=self.M, entry_idx=0)
        self._graph[idx] = [n for n, _ in neighbours]

        for n_idx, _ in neighbours:
            if len(self._graph[n_idx]) < self.M * 2:
                self._graph[n_idx].append(idx)
        return idx

    def search(self, query: list, k: int = 1) -> list:
        if not self._vectors:
            return []
        neighbours = self._greedy_search(query, k=k, entry_idx=0)
        return [(self._ids[idx], sim) for idx, sim in neighbours[:k]]

    def _greedy_search(self, query: list, k: int, entry_idx: int) -> list:
        visited = {entry_idx}
        entry_sim = _cosine_sim(query, self._vectors[entry_idx])
        candidates = [(-entry_sim, entry_idx)]
        results = [(entry_sim, entry_idx)]

        while candidates:
            neg_sim, cur = heapq.heappop(candidates)
            cur_sim = -neg_sim
            if results and cur_sim < results[-1][0] and len(results) >= self.ef:
                break
            for nb in self._graph[cur]:
                if nb in visited:
                    continue
                visited.add(nb)
                nb_sim = _cosine_sim(query, self._vectors[nb])
                heapq.heappush(candidates, (-nb_sim, nb))
                results.append((nb_sim, nb))
            if len(visited) > self.ef * 2:
                break

        results.sort(reverse=True)
        return [(idx, sim) for sim, idx in results[:k]]


# ── Bi-Encoder ───────────────────────────────────────────────────────────────

class _MiniBiEncoder:
    """
    Lightweight prompt encoder: token_ids → dim-dimensional embedding.
    Simulates a MiniLM bi-encoder via deterministic hash projection.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, token_ids: list) -> list:
        result = [0.0] * self.dim
        for i, tok in enumerate(token_ids[-256:]):
            w = math.exp(-i / 64.0)
            rng = random.Random(tok ^ 0xDEADBEEF)
            for d in range(self.dim):
                result[d] += w * rng.gauss(0, 1)
        norm = math.sqrt(sum(x * x for x in result))
        if norm > 1e-9:
            result = [x / norm for x in result]
        return result

    def encode_text(self, text: str) -> list:
        """Encode text string via simple hash tokenization."""
        rng = random.Random(hash(text))
        token_ids = [rng.randint(10, 32000) for _ in text.split()]
        return self.encode(token_ids)


# ── Cache Entry ──────────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    entry_id: str
    prompt_embedding: list
    completion_tokens: list[int]
    completion_text: str
    prompt_hash: str
    created_ts: float = field(default_factory=time.time)
    last_access_ts: float = field(default_factory=time.time)
    access_count: int = 0

    def touch(self):
        self.last_access_ts = time.time()
        self.access_count += 1

    def age_seconds(self) -> float:
        return time.time() - self.last_access_ts


# ── Neural Semantic Cache ────────────────────────────────────────────────────

class NeuralSemanticCache:
    """
    Embedding-based semantic cache for LLM inference.

    Hit path (<2ms):
      1. Encode query prompt → embedding
      2. HNSW ANN search → nearest cached embedding
      3. If cosine_sim > threshold: return cached completion
      4. Else: miss → proceed to full inference

    Store path:
      1. Encode prompt → embedding
      2. Insert into HNSW index
      3. Store (embedding, completion) in cache dict
      4. Evict if over capacity (LRU + TTL)
    """

    def __init__(self, cfg: NSCConfig | None = None):
        self.cfg = cfg or NSCConfig()
        self.dim = self.cfg.embedding_dim
        self.threshold = self.cfg.similarity_threshold
        self.max_entries = self.cfg.max_entries
        self.ttl = self.cfg.ttl_seconds

        self.encoder = _MiniBiEncoder(self.dim)
        self.index = _HNSWIndex(self.dim)
        self._store: dict[str, _CacheEntry] = {}
        self._id_counter = 0

        # Stats
        self.n_lookups = 0
        self.n_hits = 0
        self.n_misses = 0
        self.n_false_positives = 0
        self.n_stored = 0
        self.n_evictions = 0
        self._total_lookup_ms = 0.0

    # ── Core API ─────────────────────────────────────────────────────────────

    def lookup(self, token_ids: list, prompt: str = "") -> Optional[tuple[list[int], str]]:
        """
        Semantic cache lookup.
        Returns (completion_tokens, completion_text) on hit, None on miss.
        """
        t0 = time.perf_counter()

        # Encode using tokens if available, else text
        if token_ids:
            query_emb = self.encoder.encode(token_ids)
        elif prompt:
            query_emb = self.encoder.encode_text(prompt)
        else:
            self.n_lookups += 1
            self.n_misses += 1
            return None

        results = self.index.search(query_emb, k=1)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.n_lookups += 1
        self._total_lookup_ms += latency_ms

        if not results:
            self.n_misses += 1
            return None

        entry_id, sim = results[0]
        entry = self._store.get(entry_id)

        if entry is None or entry.age_seconds() > self.ttl:
            self.n_misses += 1
            if entry:
                self._evict_entry(entry_id)
            return None

        if sim >= self.threshold:
            entry.touch()
            self.n_hits += 1
            return (entry.completion_tokens, entry.completion_text)
        else:
            self.n_misses += 1
            return None

    def store(self, prompt_token_ids: list, prompt_text: str,
              completion_token_ids: list, completion_text: str):
        """Store a (prompt, completion) pair in the cache."""
        if prompt_token_ids:
            emb = self.encoder.encode(prompt_token_ids)
        else:
            emb = self.encoder.encode_text(prompt_text)

        prompt_hash = hashlib.md5(
            str(prompt_token_ids[:64]).encode()).hexdigest()

        self._id_counter += 1
        entry_id = f"nsc_{self._id_counter:08d}"

        entry = _CacheEntry(
            entry_id=entry_id,
            prompt_embedding=emb,
            completion_tokens=completion_token_ids,
            completion_text=completion_text,
            prompt_hash=prompt_hash,
        )
        self.index.add(emb, entry_id)
        self._store[entry_id] = entry
        self.n_stored += 1

        if len(self._store) > self.max_entries:
            self._evict_lru(batch=max(1, len(self._store) - self.max_entries + 50))

    # ── Threshold Adaptation ─────────────────────────────────────────────────

    def report_false_positive(self, entry_id: str = None):
        """Signal that a cache hit was incorrect."""
        self.n_false_positives += 1
        self._adapt_threshold()

    def _adapt_threshold(self, target_fpr: float = 0.02, lr: float = 0.005):
        hits = max(self.n_hits, 1)
        fpr_obs = self.n_false_positives / hits
        delta = lr * (fpr_obs - target_fpr)
        self.threshold = max(0.80, min(0.99, self.threshold + delta))

    # ── Eviction ─────────────────────────────────────────────────────────────

    def _evict_lru(self, batch: int = 100):
        entries_sorted = sorted(
            self._store.items(), key=lambda kv: kv[1].last_access_ts)
        for entry_id, _ in entries_sorted[:batch]:
            self._evict_entry(entry_id)
        self.n_evictions += batch

    def _evict_entry(self, entry_id: str):
        self._store.pop(entry_id, None)

    # ── Reporting ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = max(self.n_lookups, 1)
        return {
            "lookups": self.n_lookups,
            "hits": self.n_hits,
            "misses": self.n_misses,
            "hit_rate": round(self.n_hits / total, 3),
            "false_positives": self.n_false_positives,
            "size": len(self._store),
            "threshold": round(self.threshold, 4),
            "avg_lookup_ms": round(self._total_lookup_ms / total, 3),
            "stored": self.n_stored,
            "evictions": self.n_evictions,
        }
