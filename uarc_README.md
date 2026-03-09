# UARC — Unified Adaptive Runtime Core 🚀

[![PyPI version](https://badge.fury.io/py/uarc.svg)](https://badge.fury.io/py/uarc)
[![Python Versions](https://img.shields.io/pypi/pyversions/uarc.svg)](https://pypi.org/project/uarc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**UARC** is a lightweight, production-ready AI inference engine for Python. It provides a **single, unified gateway** for running Large Language Models locally, seamlessly bridging the gap between different backends like Ollama and Llama.cpp.

Stop rewriting your inference code every time you switch backends. With UARC, you get a zero-config CLI and an instant OpenAI-compatible server out of the box.

---

## ⚡ Quick Start

Install UARC globally via pip:

```bash
pip install uarc
```

### 1. Instant CLI Inference

Run models directly from your terminal. UARC auto-detects your backend (Ollama, local weights, etc.) and streams the response.

```bash
uarc run "Explain quantum computing in simple terms" --model llama3.2 --stream
```

### 2. Drop-in OpenAI Server

Need an API? Spin up an OpenAI-compatible server in one command. Point tools like AutoGen, LangChain, or your custom apps to `localhost:8000`.

```bash
uarc serve --port 8000 --model llama3.2
```

Test it immediately:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 3. Built-in Benchmarking

Test your hardware or model quantizations instantly. Get P50/P99 latencies, tokens/sec, and hardware stats down to the millisecond.

```bash
uarc bench --requests 100 --model llama3.2
```

---

## 🧠 Why UARC? (The Core Architecture)

UARC goes beyond just being a wrapper. It features a pipeline of experimental adaptive modules designed to maximize efficiency on consumer hardware:

| Module | Name | Purpose |
|---|---|---|
| **TDE** | Token Difficulty Estimator | Predicts token difficulty to route between draft/full models to save compute. |
| **AI-VM** | Virtual Memory Manager | Intelligent 3-tier memory management (VRAM → RAM → NVMe). |
| **DPE** | Dynamic Precision Engine | (Experimental) Per-layer bit-width allocation for memory constraints. |
| **PLL** | Predictive Layer Loader | Async layer loading from NVMe preventing pipeline stalls. |
| **NSC** | Neural Semantic Cache | Embedding-based prompt deduplication to eliminate redundant inference. |

*(Note: Adaptive routing and caching are actively being developed for the `uarc` core package).*

---

## 💻 Python API

Integrate UARC directly into your Python applications for maximum control:

```python
from uarc import UARCRuntime, UARCConfig, InferenceRequest

# 1. Configure
cfg = UARCConfig()
cfg.backend = "auto" # Auto-detects Ollama or llama_cpp
cfg.model_name = "llama3.2:1b"

# 2. Initialize
rt = UARCRuntime(cfg)
rt.start()

# 3. Infer
req = InferenceRequest(
    request_id="req-001",
    prompt="Write a python script to reverse a string.",
    max_new_tokens=256
)

response = rt.infer(req)

print(f"Output: {response.text}")
print(f"Speed:  {response.tokens_per_second:.1f} tok/s")

rt.stop()
```

---

## 🛠️ Development & Contributing

Want to help build the ultimate unified inference engine? We'd love your contributions!

```bash
git clone https://github.com/Shivay00001/uarc.git
cd uarc
pip install -e ".[dev]"
pytest tests/ -v
```

## 📄 License

UARC is licensed under the MIT License.
