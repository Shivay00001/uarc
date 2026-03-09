"""
UARC CLI — Command-line interface for the inference engine.
Usage:
  python -m uarc.cli run "prompt text" [--max-tokens N] [--stream] [--json]
  python -m uarc.cli serve [--port 8000] [--vram N] [--ram N]
  python -m uarc.cli bench [--requests N] [--vram N]
  python -m uarc.cli status
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid


def cmd_run(args):
    from uarc.core.config import UARCConfig
    from uarc.core.types import InferenceRequest
    from uarc.core.runtime import UARCRuntime

    cfg = UARCConfig()
    cfg.backend = args.backend
    cfg.model_name = args.model
    if args.vram: cfg.aivm.vram_mb = args.vram * 1024
    if args.ram: cfg.aivm.ram_mb = args.ram * 1024
    cfg.model.n_layers = args.layers

    rt = UARCRuntime(cfg)
    rt.start()

    req = InferenceRequest(
        request_id=str(uuid.uuid4()),
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
    )

    if args.stream:
        sys.stdout.write("\n")
        for chunk in rt.infer_stream(req):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        resp = rt.infer(req)
        if args.json:
            result = {
                "id": resp.request_id,
                "text": resp.text,
                "route": resp.route_taken,
                "latency_ms": resp.latency_ms,
                "tokens_per_second": resp.tokens_per_second,
                "compute_saved_pct": resp.compute_saved_pct,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"  UARC Inference Result")
            print(f"{'='*60}")
            print(f"  Prompt:  {args.prompt}")
            print(f"  Model:   {rt.backend.model_name}")
            print(f"  Output:  {resp.text}")
            print(f"  Route:   {resp.route_taken}")
            print(f"  Latency: {resp.latency_ms:.1f}ms")
            print(f"  Speed:   {resp.tokens_per_second:.0f} tok/s")
            print(f"  Saved:   {resp.compute_saved_pct:.0f}%")
            print(f"{'='*60}\n")

    rt.stop()


def cmd_serve(args):
    from uarc.server import run_server
    print(f"\n{'='*60}")
    print(f"  UARC Inference Server v0.2.0")
    print(f"  Backend: {args.backend}  Model: {args.model}")
    print(f"  Port: {args.port}")
    print(f"  VRAM: {args.vram}GB  RAM: {args.ram}GB")
    print(f"{'='*60}\n")
    run_server(port=args.port, vram_gb=args.vram, ram_gb=args.ram,
               n_layers=args.layers, backend=args.backend,
               model_name=args.model)


def cmd_bench(args):
    from uarc.core.config import UARCConfig
    from uarc.core.types import InferenceRequest
    from uarc.core.runtime import UARCRuntime
    import random

    cfg = UARCConfig()
    cfg.backend = args.backend
    cfg.model_name = args.model
    if args.vram: cfg.aivm.vram_mb = args.vram * 1024
    cfg.model.n_layers = args.layers
    rt = UARCRuntime(cfg); rt.start()

    rng = random.Random(args.seed)
    prompts = [
        "Explain quantum computing", "What is machine learning",
        "Write a Python function", "Describe neural networks",
        "How does DNA work", "Explain relativity",
        "What is blockchain", "Describe photosynthesis",
    ]

    print(f"\n{'='*60}")
    print(f"  UARC Benchmark — {args.requests} requests")
    print(f"{'='*60}\n")

    latencies = []; routes = {"draft": 0, "partial": 0, "full": 0, "cache": 0}
    t_start = time.perf_counter()

    for i in range(args.requests):
        req = InferenceRequest(
            request_id=f"bench-{i:06d}",
            prompt=rng.choice(prompts),
            max_new_tokens=rng.randint(16, 64),
        )
        resp = rt.infer(req)
        latencies.append(resp.latency_ms)
        routes[resp.route_taken] = routes.get(resp.route_taken, 0) + 1

    total_s = time.perf_counter() - t_start
    n = len(latencies)
    avg_lat = sum(latencies) / n
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[n // 2]
    p95 = sorted_lat[int(n * 0.95)]
    p99 = sorted_lat[int(n * 0.99)]

    status = rt.status()
    total_toks = status["performance"]["total_tokens_generated"]

    print(f"  Total time:   {total_s:.2f}s")
    print(f"  Throughput:   {n / total_s:.1f} req/s")
    print(f"  Tokens:       {total_toks} ({total_toks / total_s:.0f} tok/s)")
    print(f"\n  Latency (ms):")
    print(f"    avg={avg_lat:.1f}  p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}")
    print(f"\n  Routes: {dict(routes)}")
    print(f"\n{'='*60}\n")

    rt.stop()


def main():
    parser = argparse.ArgumentParser(
        prog="uarc",
        description="UARC — Unified Adaptive Runtime Core")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Single inference")
    p_run.add_argument("prompt", type=str, help="Input prompt")
    p_run.add_argument("--max-tokens", type=int, default=256)
    p_run.add_argument("--stream", action="store_true")
    p_run.add_argument("--json", action="store_true")
    p_run.add_argument("--backend", type=str, default="auto",
                       choices=["auto", "ollama", "llama_cpp", "simulated"])
    p_run.add_argument("--model", type=str, default="llama3.2:1b")
    p_run.add_argument("--vram", type=float, default=8)
    p_run.add_argument("--ram", type=float, default=32)
    p_run.add_argument("--layers", type=int, default=32)

    # serve
    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--backend", type=str, default="auto",
                         choices=["auto", "ollama", "llama_cpp", "simulated"])
    p_serve.add_argument("--model", type=str, default="llama3.2:1b")
    p_serve.add_argument("--vram", type=float, default=8)
    p_serve.add_argument("--ram", type=float, default=32)
    p_serve.add_argument("--layers", type=int, default=32)

    # bench
    p_bench = sub.add_parser("bench", help="Run benchmark")
    p_bench.add_argument("--requests", type=int, default=100)
    p_bench.add_argument("--backend", type=str, default="auto",
                         choices=["auto", "ollama", "llama_cpp", "simulated"])
    p_bench.add_argument("--model", type=str, default="llama3.2:1b")
    p_bench.add_argument("--vram", type=float, default=8)
    p_bench.add_argument("--layers", type=int, default=32)
    p_bench.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "run": cmd_run(args)
    elif args.command == "serve": cmd_serve(args)
    elif args.command == "bench": cmd_bench(args)
    else: parser.print_help()


if __name__ == "__main__":
    main()
