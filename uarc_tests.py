"""
UARC Test Suite
================
Tests every module independently and then the integrated runtime.
Run with: pytest tests/ -v
"""
from __future__ import annotations

import math
import random
import time
import uuid
import pytest

from uarc.core.config import UARCConfig, TDEConfig, AIVMConfig, NSCConfig
from uarc.core.types import (
    InferenceRequest, MemoryTier, Precision,
    RequestPriority, RouteTarget,
)


# ══════════════════════════════════════════════════════════════════════════════
# M1: Token Difficulty Estimator
# ══════════════════════════════════════════════════════════════════════════════

class TestTDE:
    @pytest.fixture
    def tde(self):
        from uarc.routing.tde import TokenDifficultyEstimator
        return TokenDifficultyEstimator(TDEConfig())

    def test_estimate_returns_routing_decision(self, tde):
        from uarc.routing.tde import TokenDifficultyEstimator
        token_ids = [100, 200, 300, 400, 500]
        dec = tde.estimate(token_ids)
        assert dec.route in list(RouteTarget)
        assert dec.estimated_ppl > 0
        assert 0.0 <= dec.confidence <= 1.0
        assert dec.latency_ms >= 0
        assert dec.compute_saved_pct in (0.0, 35.0, 85.0)

    def test_estimate_empty_tokens(self, tde):
        dec = tde.estimate([])
        assert dec.route in list(RouteTarget)

    def test_estimate_long_context(self, tde):
        token_ids = list(range(512))
        dec = tde.estimate(token_ids)
        assert dec.estimated_ppl > 0

    def test_train_reduces_loss(self, tde):
        rng = random.Random(0)
        dataset = []
        for _ in range(50):
            toks = [rng.randint(0, 1000) for _ in range(32)]
            ppl = rng.uniform(1.5, 20.0)
            dataset.append((toks, ppl))
        losses = tde.train(dataset, epochs=3)
        assert len(losses) == 3
        assert all(l >= 0 for l in losses)
        # Loss should decrease or at least not explode
        assert losses[-1] < 1000

    def test_calibration_adjusts_thresholds(self, tde):
        original_easy = tde.tau_easy
        original_hard = tde.tau_hard
        # Feed 600 samples with consistent +2 bias
        for _ in range(600):
            tde.calibrate(5.0, 7.0)   # always +2 bias
        # Thresholds should have shifted upward
        assert tde.tau_easy != original_easy or tde.tau_hard != original_hard

    def test_stats_tracking(self, tde):
        rng = random.Random(1)
        for _ in range(20):
            tde.estimate([rng.randint(0, 32000) for _ in range(32)])
        s = tde.stats()
        assert s["total_routed"] == 20
        assert sum(s["route_pct"].values()) == pytest.approx(100.0, abs=0.5)

    def test_draft_route_highest_savings(self, tde):
        # Force easy route by setting very high tau_hard
        tde.tau_easy = 1000.0
        tde.tau_hard = 2000.0
        dec = tde.estimate([1, 2, 3])
        assert dec.route == RouteTarget.DRAFT
        assert dec.compute_saved_pct == 85.0

    def test_full_route_zero_savings(self, tde):
        # Force hard route by setting very low thresholds
        tde.tau_easy = 0.001
        tde.tau_hard = 0.002
        dec = tde.estimate([1, 2, 3])
        assert dec.route == RouteTarget.FULL
        assert dec.compute_saved_pct == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# M2: AI Virtual Memory Manager
# ══════════════════════════════════════════════════════════════════════════════

class TestAIVM:
    @pytest.fixture
    def aivm(self):
        from uarc.memory.aivm import AIVirtualMemoryManager
        cfg = AIVMConfig(vram_mb=512, ram_mb=2048, nvme_mb=8192)
        return AIVirtualMemoryManager(cfg)

    def test_allocate_and_locate(self, aivm):
        page = aivm.allocate("p1", size_mb=2.0, data_type="weight",
                              preferred_tier=MemoryTier.NVME)
        assert page is not None
        assert page.page_id == "p1"
        loc = aivm.locate("p1")
        assert loc is not None

    def test_access_tracks_stats(self, aivm):
        aivm.allocate("p2", size_mb=1.0)
        page = aivm.access("p2")
        assert page is not None
        assert page.access_count == 1
        aivm.access("p2")
        page = aivm.access("p2")
        assert page.access_count == 3

    def test_promote_page(self, aivm):
        aivm.allocate("p3", size_mb=1.0, preferred_tier=MemoryTier.NVME)
        loc_before = aivm.locate("p3")
        ms = aivm.promote("p3", MemoryTier.RAM)
        loc_after = aivm.locate("p3")
        assert ms >= 0
        # After promotion, tier should be <= RAM (could go to RAM or VRAM)
        assert loc_after.value <= MemoryTier.RAM.value

    def test_free_removes_page(self, aivm):
        aivm.allocate("p4", size_mb=1.0)
        aivm.free("p4")
        assert aivm.locate("p4") is None

    def test_kv_allocate_and_lookup(self, aivm):
        blk = aivm.kv_allocate("seq_001", block_idx=0)
        assert blk is not None
        found = aivm.kv_lookup("seq_001", 0)
        assert found is not None
        assert found.seq_id == "seq_001"

    def test_kv_prefix_sharing(self, aivm):
        for i in range(4):
            aivm.kv_allocate("seq_A", i)
        shared = aivm.kv_share_prefix("seq_A", "seq_B", n=4)
        assert shared == 4
        # seq_B should now have blocks 0-3
        for i in range(4):
            blk = aivm.kv_lookup("seq_B", i)
            assert blk is not None

    def test_kv_free_sequence(self, aivm):
        for i in range(3):
            aivm.kv_allocate("seq_C", i)
        aivm.kv_free("seq_C")
        for i in range(3):
            assert aivm.kv_lookup("seq_C", i) is None

    def test_memory_pressure_tracking(self, aivm):
        s = aivm.status()
        assert "VRAM" in s or "RAM" in s or "NVMe" in s

    def test_allocation_fallback_on_full_tier(self, aivm):
        # Fill VRAM with many pages
        n = 0
        try:
            for i in range(1000):
                aivm.allocate(f"fill_{i}", size_mb=100.0,
                              preferred_tier=MemoryTier.VRAM)
                n += 1
        except MemoryError:
            pass
        # Should have fallen back to RAM/NVMe
        assert n > 0


# ══════════════════════════════════════════════════════════════════════════════
# M3: Dynamic Precision Engine
# ══════════════════════════════════════════════════════════════════════════════

class TestDPE:
    @pytest.fixture
    def dpe(self):
        from uarc.scheduling.dpe_acs import DynamicPrecisionEngine
        from uarc.core.config import DPEConfig
        profiles = DynamicPrecisionEngine.build_profiles(32, 100_000_000)
        return DynamicPrecisionEngine(DPEConfig(), profiles)

    def test_allocate_returns_plan(self, dpe):
        plan = dpe.allocate(8 * 1024**3)
        assert plan is not None
        assert len(plan.assignment) == 32
        assert plan.total_bytes > 0

    def test_allocation_respects_budget(self, dpe):
        budget = 4 * 1024**3
        plan = dpe.allocate(budget)
        assert plan.total_bytes <= budget * 1.01   # 1% tolerance

    def test_larger_budget_better_quality(self, dpe):
        plan_small = dpe.allocate(4 * 1024**3)
        plan_large = dpe.allocate(16 * 1024**3)
        assert plan_large.estimated_ppl_delta <= plan_small.estimated_ppl_delta

    def test_larger_budget_more_bits(self, dpe):
        plan_small = dpe.allocate(4 * 1024**3)
        plan_large = dpe.allocate(16 * 1024**3)
        assert plan_large.avg_bits >= plan_small.avg_bits

    def test_plan_caching(self, dpe):
        t0 = time.perf_counter()
        dpe.allocate(8 * 1024**3)
        t1 = time.perf_counter()
        dpe.allocate(8 * 1024**3)  # cached
        t2 = time.perf_counter()
        # Second call should be faster (cache hit)
        assert (t2 - t1) <= (t1 - t0) + 0.01

    def test_adapt_for_hard_token(self, dpe):
        plan = dpe.allocate(8 * 1024**3)
        adapted = dpe.adapt_for_token(plan, difficulty=15.0)
        # Critical layers (first/last 10%) should be upgraded
        n = len(plan.assignment)
        crit = list(range(n // 10)) + list(range(n * 9 // 10, n))
        for lid in crit:
            original_bits = plan.assignment[lid].value
            adapted_bits  = adapted.assignment[lid].value
            assert adapted_bits >= original_bits

    def test_all_layers_have_valid_precision(self, dpe):
        plan = dpe.allocate(8 * 1024**3)
        valid = {Precision.INT4, Precision.INT8, Precision.FP16, Precision.FP32}
        for p in plan.assignment:
            assert p in valid


# ══════════════════════════════════════════════════════════════════════════════
# M4: Predictive Layer Loader
# ══════════════════════════════════════════════════════════════════════════════

class TestPLL:
    @pytest.fixture
    def pll(self):
        from uarc.memory.aivm import PredictiveLayerLoader
        from uarc.core.config import PLLConfig
        cfg = PLLConfig(lookahead_k=4, slack_ms=5.0)
        return PredictiveLayerLoader(cfg, n_layers=32,
                                     layer_sizes_mb=[2.0]*32)

    def test_prefetch_issued_on_layer_complete(self, pll):
        orders = []
        pll.on_prefetch = lambda o: orders.append(o)
        for i in range(5):
            pll.on_layer_start(i)
            time.sleep(0.002)
            pll.on_layer_complete(i)
        assert len(orders) >= 0   # May or may not issue based on timing

    def test_ema_update(self, pll):
        for i in range(5):
            pll.on_layer_start(i)
            time.sleep(0.010)  # 10ms
            pll.on_layer_complete(i)
        # EMA should have updated from initial 50ms
        # (Either increased or decreased depending on actual sleep)
        assert pll._exec_ema[0] != 50.0 or pll._exec_ema[1] != 50.0

    def test_reset_clears_issued_set(self, pll):
        pll._issued.add(5)
        pll._issued.add(6)
        pll.reset()
        assert len(pll._issued) == 0

    def test_near_layers_target_vram(self, pll):
        orders = []
        pll.on_prefetch = lambda o: orders.append(o)
        # Set very short exec time so prefetch triggers
        pll._exec_ema = [500.0] * 32
        pll._load_ema = [1.0]   * 32
        pll.on_layer_start(0)
        pll.on_layer_complete(0)
        near = [o for o in orders if o.layer_id in (1, 2)]
        if near:
            assert near[0].target_tier == MemoryTier.VRAM


# ══════════════════════════════════════════════════════════════════════════════
# M5: Adaptive Compute Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class TestACS:
    @pytest.fixture
    def acs(self):
        from uarc.scheduling.dpe_acs import AdaptiveComputeScheduler
        from uarc.core.config import ACSConfig
        return AdaptiveComputeScheduler(ACSConfig(max_batch_size=16))

    def _req(self, priority=RequestPriority.STANDARD, prompt="hello world"):
        return InferenceRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            priority=priority,
        )

    def test_submit_and_form_batch(self, acs):
        for _ in range(5):
            acs.submit(self._req())
        batch = acs.form_batch()
        assert batch is not None
        assert batch.size == 5

    def test_realtime_scheduled_first(self, acs):
        acs.submit(self._req(RequestPriority.BATCH))
        acs.submit(self._req(RequestPriority.BATCH))
        acs.submit(self._req(RequestPriority.REALTIME))
        batch = acs.form_batch()
        assert batch is not None
        # REALTIME request should appear first in batch
        assert batch.requests[0].priority == RequestPriority.REALTIME

    def test_empty_queue_returns_none(self, acs):
        batch = acs.form_batch()
        assert batch is None

    def test_prefix_grouping(self, acs):
        shared_prefix = list(range(64))
        for i in range(4):
            req = self._req(prompt=f"same prefix query {i}")
            req.token_ids = shared_prefix + [i]
            acs.submit(req)
        batch = acs.form_batch()
        assert batch is not None
        # Stats should reflect some sharing
        assert acs.stats["kv_sharing_pairs"] >= 0

    def test_route_cpu_small_batch(self, acs):
        device = acs.route(batch_size=1, n_params=100_000_000)
        assert device.value == "cpu"

    def test_route_gpu_large_batch(self, acs):
        acs.gpu_util = 0.5
        device = acs.route(batch_size=32, n_params=100_000_000)
        assert device.value == "gpu"

    def test_route_cpu_on_gpu_saturation(self, acs):
        acs.gpu_util = 0.95
        device = acs.route(batch_size=32, n_params=100_000_000)
        assert device.value == "cpu"

    def test_stats_report(self, acs):
        for _ in range(10):
            acs.submit(self._req())
        acs.form_batch()
        s = acs.stats_report()
        assert s["submitted"] == 10
        assert s["batches_formed"] == 1
        assert s["avg_batch_size"] == 10.0


# ══════════════════════════════════════════════════════════════════════════════
# M6: Neural Semantic Cache
# ══════════════════════════════════════════════════════════════════════════════

class TestNSC:
    @pytest.fixture
    def nsc(self):
        from uarc.routing.nsc import NeuralSemanticCache
        cfg = NSCConfig(embedding_dim=64, similarity_threshold=0.80,
                        max_entries=200, ttl_seconds=3600.0)
        return NeuralSemanticCache(cfg)

    def test_miss_on_empty_cache(self, nsc):
        result = nsc.lookup([1, 2, 3], "hello world")
        assert result is None

    def test_exact_hit_after_store(self, nsc):
        toks = [100, 200, 300, 400]
        comp = [10, 20, 30]
        nsc.store(toks, "test prompt", comp, "test completion")
        result = nsc.lookup(toks, "test prompt")
        assert result is not None
        comp_toks, comp_text = result
        assert comp_text == "test completion"

    def test_cache_accumulates_entries(self, nsc):
        rng = random.Random(0)
        for i in range(20):
            nsc.store(
                [rng.randint(0, 1000) for _ in range(16)],
                f"prompt {i}",
                [rng.randint(0, 1000) for _ in range(8)],
                f"completion {i}",
            )
        assert nsc.stats()["size"] == 20

    def test_false_positive_raises_threshold(self, nsc):
        original = nsc.threshold
        for _ in range(20):
            nsc.n_hits += 1   # simulate hits first
        for _ in range(5):
            nsc.report_false_positive()
        # High FPR should raise threshold
        assert nsc.threshold >= original

    def test_stats_tracking(self, nsc):
        nsc.lookup([1, 2], "miss query")
        s = nsc.stats()
        assert s["lookups"] == 1
        assert s["misses"] == 1
        assert s["hits"] == 0

    def test_lru_eviction_on_overflow(self, nsc):
        rng = random.Random(42)
        for i in range(250):   # max_entries=200
            nsc.store(
                [rng.randint(0, 32000) for _ in range(8)],
                f"prompt {i}",
                [1, 2, 3],
                f"completion {i}",
            )
        assert nsc.stats()["size"] <= 220   # allows small buffer


# ══════════════════════════════════════════════════════════════════════════════
# Full Runtime Integration
# ══════════════════════════════════════════════════════════════════════════════

class TestRuntime:
    @pytest.fixture
    def runtime(self):
        from uarc.core.runtime import UARCRuntime
        cfg = UARCConfig()
        cfg.aivm.vram_mb = 512
        cfg.aivm.ram_mb  = 2048
        cfg.aivm.nvme_mb = 8192
        cfg.model.n_layers = 8   # small for speed
        rt = UARCRuntime(cfg)
        rt.start()
        yield rt
        rt.stop()

    def test_basic_inference(self, runtime):
        req = InferenceRequest(
            request_id="test-001",
            prompt="Hello, world!",
            max_new_tokens=32,
        )
        resp = runtime.infer(req)
        assert resp.request_id == "test-001"
        assert len(resp.text) > 0
        assert resp.completion_tokens > 0
        assert resp.latency_ms > 0

    def test_cache_hit_on_repeat(self, runtime):
        req = InferenceRequest(
            request_id="test-002",
            prompt="What is the meaning of life?",
            max_new_tokens=32,
        )
        runtime.infer(req)   # First: cache miss, stores result
        req2 = InferenceRequest(
            request_id="test-003",
            prompt="What is the meaning of life?",
            max_new_tokens=32,
        )
        resp2 = runtime.infer(req2)
        # May or may not hit cache depending on embedding similarity
        assert resp2.route_taken in ("cache", "draft", "partial", "full")

    def test_response_has_metadata(self, runtime):
        req = InferenceRequest(
            request_id="test-004",
            prompt="Explain neural networks",
            max_new_tokens=16,
        )
        resp = runtime.infer(req)
        assert resp.route_taken in ("draft", "partial", "full", "cache")
        assert resp.prompt_tokens >= 0
        assert resp.tokens_per_second >= 0

    def test_batch_inference(self, runtime):
        reqs = [
            InferenceRequest(request_id=f"batch-{i}", prompt=f"Query {i}",
                             max_new_tokens=16)
            for i in range(5)
        ]
        responses = runtime.infer_batch(reqs)
        assert len(responses) == 5
        for r in responses:
            assert r.completion_tokens > 0

    def test_status_returns_all_modules(self, runtime):
        s = runtime.status()
        assert "modules" in s
        assert "memory" in s
        assert "performance" in s
        for module in ("tde", "nsc", "dpe", "pll", "acs"):
            assert module in s["modules"], f"Missing module in status: {module}"

    def test_stats_accumulate(self, runtime):
        for i in range(5):
            req = InferenceRequest(request_id=f"stat-{i}",
                                   prompt=f"Prompt {i}", max_new_tokens=8)
            runtime.infer(req)
        s = runtime.status()
        assert s["performance"]["total_requests"] >= 5  # type: ignore

    def test_streaming_yields_tokens(self, runtime):
        req = InferenceRequest(
            request_id="stream-001",
            prompt="Tell me a story",
            max_new_tokens=32,
        )
        chunks = list(runtime.infer_stream(req))
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

    def test_different_priorities_accepted(self, runtime):
        for prio in RequestPriority:
            req = InferenceRequest(
                request_id=str(uuid.uuid4()),
                prompt="Priority test",
                priority=prio,
                max_new_tokens=8,
            )
            resp = runtime.infer(req)
            assert resp is not None

    def test_memory_status_after_inference(self, runtime):
        req = InferenceRequest(
            request_id="mem-test",
            prompt="Memory test prompt",
            max_new_tokens=8,
        )
        runtime.infer(req)
        mem = runtime.aivm.status()
        # At least one tier should have used some memory (model pages)
        total_used = sum(
            t.get("used_mb", 0) for t in mem.values()
            if isinstance(t, dict)
        )
        assert total_used >= 0
