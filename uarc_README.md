# UARC — Unified Adaptive Runtime Core

> Next-generation AI inference engine: 7 adaptive modules, portable, OpenAI-compatible.

```
pip install uarc
```

## Modules

| # | Module | Purpose | Key Algorithm |
|---|--------|---------|---------------|
| M1 | **TDE** — Token Difficulty Estimator | Per-token compute routing | 4-layer MLP → perplexity → draft/partial/full |
| M2 | **AI-VM** — Virtual Memory Manager | 3-tier memory (VRAM→RAM→NVMe) | CLOCK-Pro eviction + KV page table |
| M3 | **DPE** — Dynamic Precision Engine | Per-layer bit-width allocation | Greedy knapsack (INT4→INT8→FP16→FP32) |
| M4 | **PLL** — Predictive Layer Loader | Lookahead prefetch scheduling | EMA timing + DAG deadline analysis |
| M5 | **ACS** — Adaptive Compute Scheduler | Priority batch formation + device routing | 3-heap priority queue + roofline routing |
| M6 | **NSC** — Neural Semantic Cache | Embedding-based prompt deduplication | HNSW ANN + bi-encoder + adaptive threshold |
| M7 | **Router** — Hybrid CPU/GPU | Arithmetic-intensity routing | Roofline model + pipeline parallelism |

---

## Quick Start

### 1. Single Inference (CLI)
```bash
uarc run "Explain quantum computing in simple terms"
uarc run "Write Python code to sort a list" --max-tokens 256 --stream
uarc run "Quick question" --priority realtime --json
```

### 2. Inference Server (OpenAI-compatible)
```bash
uarc serve --port 8000 --vram 8 --ram 32

# Then call it like any OpenAI API:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "uarc-sim-7b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 3. Benchmark
```bash
uarc bench --requests 100 --vram 8 --seed-cache
```

### 4. Python API
```python
from uarc import UARCRuntime, UARCConfig, InferenceRequest

cfg = UARCConfig.from_env()   # reads UARC_VRAM_GB, UARC_RAM_GB, etc.
rt  = UARCRuntime(cfg)
rt.start()

resp = rt.infer(InferenceRequest(
    request_id="req-001",
    prompt="What is machine learning?",
    max_new_tokens=256,
))

print(resp.text)
print(f"Route: {resp.route_taken}")          # draft / partial / full / cache
print(f"Latency: {resp.latency_ms:.1f}ms")
print(f"Compute saved: {resp.compute_saved_pct:.0f}%")

rt.stop()
```

### 5. Streaming
```python
for chunk in rt.infer_stream(request):
    print(chunk, end="", flush=True)
```

### 6. Batch
```python
responses = rt.infer_batch([req1, req2, req3])
```

---

## Configuration

```python
from uarc import UARCConfig

# Presets
cfg = UARCConfig.for_edge()          # 4GB RAM, no GPU
cfg = UARCConfig.for_gpu(vram_gb=24) # 24GB VRAM

# From environment variables
# UARC_VRAM_GB, UARC_RAM_GB, UARC_MAX_BATCH, UARC_PORT,
# UARC_NSC_THRESHOLD, UARC_DEVICE
cfg = UARCConfig.from_env()

# Manual
cfg = UARCConfig()
cfg.aivm.vram_mb = 8192         # 8 GB VRAM
cfg.tde.tau_easy = 2.5          # PPL threshold: easy tokens
cfg.tde.tau_hard = 8.0          # PPL threshold: hard tokens
cfg.nsc.similarity_threshold = 0.92
cfg.acs.max_batch_size = 32
```

---

## Server Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI chat (streaming supported) |
| POST | `/v1/completions` | OpenAI completion (streaming supported) |
| GET  | `/v1/models` | Model list |
| GET  | `/health` | Health check |
| GET  | `/status` | Full module status JSON |
| GET  | `/metrics` | Prometheus metrics |
| POST | `/admin/cache/clear` | Flush NSC cache |
| POST | `/admin/evict` | Trigger memory eviction |

---

## Architecture

```
InferenceRequest
     │
     ▼
  M6 NSC ──── hit ────────────────────────────► Response (cache)
     │ miss
     ▼
  M1 TDE ──── estimate PPL ──────► route: draft / partial / full
     │
     ▼
  M5 ACS ──── form batch ─────► prefix-sharing reorder
     │
     ▼
  M3 DPE ──── precision plan ──► INT4/INT8/FP16/FP32 per layer
     │
     ▼
  M4 PLL ──── prefetch ────────► async layer loading from NVMe
     │
     ▼
  M2 AI-VM ── VRAM/RAM/NVMe ──► CLOCK-Pro eviction
     │
     ▼
  M7 Router ─ CPU or GPU? ────► intensity < 4 → CPU, else GPU
     │
     ▼
  Model forward pass
     │
     ▼
  M6 NSC store
     │
     ▼
  Response
```

---

## Development

```bash
git clone <repo>
cd uarc
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Performance Targets

| Scenario | Baseline | UARC Target |
|----------|----------|-------------|
| 70B on 8GB VRAM | OOM | 85 tok/s |
| 70B on 24GB GPU | 12 tok/s | 38 tok/s (+217%) |
| TTFT | 1.0× | 0.38× (−62%) |
| Concurrent requests | 4 | 18–32 |
| Compute per token | 1.0× | 0.30–0.70× |
