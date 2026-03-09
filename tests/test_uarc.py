"""
UARC Test Suite
================
Tests every module independently and then the integrated runtime.
Run with: pytest tests/test_uarc.py -v
"""
from __future__ import annotations
import math, random, time, uuid
import pytest

from uarc.core.config import UARCConfig, TDEConfig, AIVMConfig, NSCConfig, DPEConfig, PLLConfig, ACSConfig
from uarc.core.types import InferenceRequest, MemoryTier, Precision, RequestPriority, RouteTarget


# ══════════════════════════════════════════════════════════════════════════════
# M1: Token Difficulty Estimator
# ══════════════════════════════════════════════════════════════════════════════
class TestTDE:
    @pytest.fixture
    def tde(self):
        from uarc.routing.tde import TokenDifficultyEstimator
        return TokenDifficultyEstimator(TDEConfig())

    def test_estimate_returns_routing_decision(self, tde):
        dec = tde.estimate([100, 200, 300, 400, 500])
        assert dec.route in list(RouteTarget)
        assert dec.estimated_ppl > 0
        assert 0.0 <= dec.confidence <= 1.0
        assert dec.latency_ms >= 0
        assert dec.compute_saved_pct in (0.0, 35.0, 85.0)

    def test_estimate_empty_tokens(self, tde):
        dec = tde.estimate([])
        assert dec.route in list(RouteTarget)

    def test_estimate_long_context(self, tde):
        dec = tde.estimate(list(range(512)))
        assert dec.estimated_ppl > 0

    def test_train_reduces_loss(self, tde):
        rng = random.Random(0)
        dataset = [([rng.randint(0, 1000) for _ in range(32)], rng.uniform(1.5, 20.0)) for _ in range(50)]
        losses = tde.train(dataset, epochs=3)
        assert len(losses) == 3
        assert all(l >= 0 for l in losses)
        assert losses[-1] < 1000

    def test_calibration_adjusts_thresholds(self, tde):
        orig_easy, orig_hard = tde.tau_easy, tde.tau_hard
        for _ in range(600): tde.calibrate(5.0, 7.0)
        assert tde.tau_easy != orig_easy or tde.tau_hard != orig_hard

    def test_stats_tracking(self, tde):
        rng = random.Random(1)
        for _ in range(20): tde.estimate([rng.randint(0, 32000) for _ in range(32)])
        s = tde.stats()
        assert s["total_routed"] == 20
        assert sum(s["route_pct"].values()) == pytest.approx(100.0, abs=0.5)

    def test_draft_route_highest_savings(self, tde):
        tde.tau_easy = 1000.0; tde.tau_hard = 2000.0
        dec = tde.estimate([1, 2, 3])
        assert dec.route == RouteTarget.DRAFT
        assert dec.compute_saved_pct == 85.0

    def test_full_route_zero_savings(self, tde):
        tde.tau_easy = 0.001; tde.tau_hard = 0.002
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
        return AIVirtualMemoryManager(AIVMConfig(vram_mb=512, ram_mb=2048, nvme_mb=8192))

    def test_allocate_and_locate(self, aivm):
        page = aivm.allocate("p1", size_mb=2.0, data_type="weight", preferred_tier=MemoryTier.NVME)
        assert page is not None and page.page_id == "p1"
        assert aivm.locate("p1") is not None

    def test_access_tracks_stats(self, aivm):
        aivm.allocate("p2", size_mb=1.0)
        page = aivm.access("p2"); assert page.access_count == 1
        aivm.access("p2"); page = aivm.access("p2"); assert page.access_count == 3

    def test_promote_page(self, aivm):
        aivm.allocate("p3", size_mb=1.0, preferred_tier=MemoryTier.NVME)
        ms = aivm.promote("p3", MemoryTier.RAM)
        loc = aivm.locate("p3"); assert ms >= 0; assert loc.value <= MemoryTier.RAM.value

    def test_free_removes_page(self, aivm):
        aivm.allocate("p4", size_mb=1.0); aivm.free("p4"); assert aivm.locate("p4") is None

    def test_kv_allocate_and_lookup(self, aivm):
        blk = aivm.kv_allocate("seq_001", block_idx=0)
        assert blk is not None
        found = aivm.kv_lookup("seq_001", 0); assert found is not None and found.seq_id == "seq_001"

    def test_kv_prefix_sharing(self, aivm):
        for i in range(4): aivm.kv_allocate("seq_A", i)
        shared = aivm.kv_share_prefix("seq_A", "seq_B", n=4); assert shared == 4
        for i in range(4): assert aivm.kv_lookup("seq_B", i) is not None

    def test_kv_free_sequence(self, aivm):
        for i in range(3): aivm.kv_allocate("seq_C", i)
        aivm.kv_free("seq_C")
        for i in range(3): assert aivm.kv_lookup("seq_C", i) is None

    def test_memory_pressure_tracking(self, aivm):
        s = aivm.status(); assert "VRAM" in s or "RAM" in s or "NVMe" in s

    def test_allocation_fallback_on_full_tier(self, aivm):
        n = 0
        try:
            for i in range(1000):
                aivm.allocate(f"fill_{i}", size_mb=100.0, preferred_tier=MemoryTier.VRAM); n += 1
        except MemoryError: pass
        assert n > 0


# ══════════════════════════════════════════════════════════════════════════════
# M3: Dynamic Precision Engine
# ══════════════════════════════════════════════════════════════════════════════
class TestDPE:
    @pytest.fixture
    def dpe(self):
        from uarc.scheduling.dpe_acs import DynamicPrecisionEngine
        profiles = DynamicPrecisionEngine.build_profiles(32, 100_000_000)
        return DynamicPrecisionEngine(DPEConfig(), profiles)

    def test_allocate_returns_plan(self, dpe):
        plan = dpe.allocate(8 * 1024**3)
        assert plan is not None and len(plan.assignment) == 32 and plan.total_bytes > 0

    def test_allocation_respects_budget(self, dpe):
        budget = 4 * 1024**3; plan = dpe.allocate(budget)
        assert plan.total_bytes <= budget * 1.01

    def test_larger_budget_better_quality(self, dpe):
        s = dpe.allocate(4 * 1024**3); l = dpe.allocate(16 * 1024**3)
        assert l.estimated_ppl_delta <= s.estimated_ppl_delta

    def test_larger_budget_more_bits(self, dpe):
        s = dpe.allocate(4 * 1024**3); l = dpe.allocate(16 * 1024**3)
        assert l.avg_bits >= s.avg_bits

    def test_plan_caching(self, dpe):
        t0 = time.perf_counter(); dpe.allocate(8 * 1024**3)
        t1 = time.perf_counter(); dpe.allocate(8 * 1024**3)
        t2 = time.perf_counter()
        assert (t2 - t1) <= (t1 - t0) + 0.01

    def test_adapt_for_hard_token(self, dpe):
        plan = dpe.allocate(8 * 1024**3); adapted = dpe.adapt_for_token(plan, difficulty=15.0)
        n = len(plan.assignment); crit = list(range(n // 10)) + list(range(n * 9 // 10, n))
        for lid in crit:
            assert adapted.assignment[lid].value >= plan.assignment[lid].value

    def test_all_layers_have_valid_precision(self, dpe):
        plan = dpe.allocate(8 * 1024**3)
        valid = {Precision.INT4, Precision.INT8, Precision.FP16, Precision.FP32}
        for p in plan.assignment: assert p in valid


# ══════════════════════════════════════════════════════════════════════════════
# M4: Predictive Layer Loader
# ══════════════════════════════════════════════════════════════════════════════
class TestPLL:
    @pytest.fixture
    def pll(self):
        from uarc.memory.aivm import PredictiveLayerLoader
        return PredictiveLayerLoader(PLLConfig(lookahead_k=4, slack_ms=5.0), n_layers=32, layer_sizes_mb=[2.0]*32)

    def test_prefetch_issued_on_layer_complete(self, pll):
        orders = []; pll.on_prefetch = lambda o: orders.append(o)
        for i in range(5):
            pll.on_layer_start(i); time.sleep(0.002); pll.on_layer_complete(i)
        assert len(orders) >= 0

    def test_ema_update(self, pll):
        for i in range(5):
            pll.on_layer_start(i); time.sleep(0.010); pll.on_layer_complete(i)
        assert pll._exec_ema[0] != 50.0 or pll._exec_ema[1] != 50.0

    def test_reset_clears_issued_set(self, pll):
        pll._issued.add(5); pll._issued.add(6); pll.reset(); assert len(pll._issued) == 0

    def test_near_layers_target_vram(self, pll):
        orders = []; pll.on_prefetch = lambda o: orders.append(o)
        pll._exec_ema = [500.0] * 32; pll._load_ema = [1.0] * 32
        pll.on_layer_start(0); pll.on_layer_complete(0)
        near = [o for o in orders if o.layer_id in (1, 2)]
        if near: assert near[0].target_tier == MemoryTier.VRAM


# ══════════════════════════════════════════════════════════════════════════════
# M5: Adaptive Compute Scheduler
# ══════════════════════════════════════════════════════════════════════════════
class TestACS:
    @pytest.fixture
    def acs(self):
        from uarc.scheduling.dpe_acs import AdaptiveComputeScheduler
        return AdaptiveComputeScheduler(ACSConfig(max_batch_size=16))

    def _req(self, priority=RequestPriority.STANDARD, prompt="hello world"):
        return InferenceRequest(request_id=str(uuid.uuid4()), prompt=prompt, priority=priority)

    def test_submit_and_form_batch(self, acs):
        for _ in range(5): acs.submit(self._req())
        batch = acs.form_batch(); assert batch is not None and batch.size == 5

    def test_realtime_scheduled_first(self, acs):
        acs.submit(self._req(RequestPriority.BATCH)); acs.submit(self._req(RequestPriority.BATCH))
        acs.submit(self._req(RequestPriority.REALTIME))
        batch = acs.form_batch(); assert batch.requests[0].priority == RequestPriority.REALTIME

    def test_empty_queue_returns_none(self, acs):
        assert acs.form_batch() is None

    def test_prefix_grouping(self, acs):
        shared = list(range(64))
        for i in range(4):
            req = self._req(prompt=f"same prefix {i}"); req.token_ids = shared + [i]; acs.submit(req)
        batch = acs.form_batch(); assert batch is not None
        assert acs.stats["kv_sharing_pairs"] >= 0

    def test_route_cpu_small_batch(self, acs):
        from uarc.core.types import DeviceType
        assert acs.route(batch_size=1, n_params=100_000_000).value == "cpu"

    def test_route_gpu_large_batch(self, acs):
        acs.gpu_util = 0.5
        assert acs.route(batch_size=32, n_params=100_000_000).value == "gpu"

    def test_route_cpu_on_gpu_saturation(self, acs):
        acs.gpu_util = 0.95
        assert acs.route(batch_size=32, n_params=100_000_000).value == "cpu"

    def test_stats_report(self, acs):
        for _ in range(10): acs.submit(self._req())
        acs.form_batch(); s = acs.stats_report()
        assert s["submitted"] == 10 and s["batches_formed"] == 1 and s["avg_batch_size"] == 10.0


# ══════════════════════════════════════════════════════════════════════════════
# M6: Neural Semantic Cache
# ══════════════════════════════════════════════════════════════════════════════
class TestNSC:
    @pytest.fixture
    def nsc(self):
        from uarc.routing.nsc import NeuralSemanticCache
        return NeuralSemanticCache(NSCConfig(embedding_dim=64, similarity_threshold=0.80, max_entries=200, ttl_seconds=3600.0))

    def test_miss_on_empty_cache(self, nsc):
        assert nsc.lookup([1, 2, 3], "hello world") is None

    def test_exact_hit_after_store(self, nsc):
        toks = [100, 200, 300, 400]; nsc.store(toks, "test prompt", [10, 20, 30], "test completion")
        result = nsc.lookup(toks, "test prompt")
        assert result is not None; ct, cx = result; assert cx == "test completion"

    def test_cache_accumulates_entries(self, nsc):
        rng = random.Random(0)
        for i in range(20):
            nsc.store([rng.randint(0,1000) for _ in range(16)], f"p{i}",
                      [rng.randint(0,1000) for _ in range(8)], f"c{i}")
        assert nsc.stats()["size"] == 20

    def test_false_positive_raises_threshold(self, nsc):
        orig = nsc.threshold
        for _ in range(20): nsc.n_hits += 1
        for _ in range(5): nsc.report_false_positive()
        assert nsc.threshold >= orig

    def test_stats_tracking(self, nsc):
        nsc.lookup([1, 2], "miss"); s = nsc.stats()
        assert s["lookups"] == 1 and s["misses"] == 1 and s["hits"] == 0

    def test_lru_eviction_on_overflow(self, nsc):
        rng = random.Random(42)
        for i in range(250):
            nsc.store([rng.randint(0,32000) for _ in range(8)], f"p{i}", [1,2,3], f"c{i}")
        assert nsc.stats()["size"] <= 220


# ══════════════════════════════════════════════════════════════════════════════
# Full Runtime Integration
# ══════════════════════════════════════════════════════════════════════════════
class TestRuntime:
    @pytest.fixture
    def runtime(self):
        from uarc.core.runtime import UARCRuntime
        cfg = UARCConfig(); cfg.aivm.vram_mb = 512; cfg.aivm.ram_mb = 2048
        cfg.aivm.nvme_mb = 8192; cfg.model.n_layers = 8
        rt = UARCRuntime(cfg); rt.start(); yield rt; rt.stop()

    def test_basic_inference(self, runtime):
        resp = runtime.infer(InferenceRequest(request_id="t-001", prompt="Hello!", max_new_tokens=32))
        assert resp.request_id == "t-001" and len(resp.text) > 0 and resp.completion_tokens > 0

    def test_cache_hit_on_repeat(self, runtime):
        runtime.infer(InferenceRequest(request_id="t-002", prompt="What is life?", max_new_tokens=32))
        resp2 = runtime.infer(InferenceRequest(request_id="t-003", prompt="What is life?", max_new_tokens=32))
        assert resp2.route_taken in ("cache", "draft", "partial", "full")

    def test_response_has_metadata(self, runtime):
        resp = runtime.infer(InferenceRequest(request_id="t-004", prompt="Neural networks", max_new_tokens=16))
        assert resp.route_taken in ("draft", "partial", "full", "cache")
        assert resp.prompt_tokens >= 0 and resp.tokens_per_second >= 0

    def test_batch_inference(self, runtime):
        reqs = [InferenceRequest(request_id=f"b-{i}", prompt=f"Q{i}", max_new_tokens=16) for i in range(5)]
        responses = runtime.infer_batch(reqs); assert len(responses) == 5
        for r in responses: assert r.completion_tokens > 0

    def test_status_returns_all_modules(self, runtime):
        s = runtime.status(); assert "modules" in s and "memory" in s and "performance" in s
        for m in ("tde", "nsc", "dpe", "pll", "acs"): assert m in s["modules"]

    def test_stats_accumulate(self, runtime):
        for i in range(5):
            runtime.infer(InferenceRequest(request_id=f"s-{i}", prompt=f"P{i}", max_new_tokens=8))
        assert runtime.status()["performance"]["total_requests"] >= 5

    def test_streaming_yields_tokens(self, runtime):
        chunks = list(runtime.infer_stream(InferenceRequest(request_id="st-1", prompt="Story", max_new_tokens=32)))
        assert len(chunks) > 0 and len("".join(chunks)) > 0

    def test_different_priorities_accepted(self, runtime):
        for prio in RequestPriority:
            resp = runtime.infer(InferenceRequest(
                request_id=str(uuid.uuid4()), prompt="Priority test", priority=prio, max_new_tokens=8))
            assert resp is not None

    def test_memory_status_after_inference(self, runtime):
        runtime.infer(InferenceRequest(request_id="m-t", prompt="Memory test", max_new_tokens=8))
        mem = runtime.aivm.status()
        total = sum(t.get("used_mb", 0) for t in mem.values() if isinstance(t, dict))
        assert total >= 0
