"""
UARC OpenAI-Compatible HTTP Server
====================================
Pure stdlib server (http.server) — no Flask, no FastAPI needed.
Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /health, /status, /metrics
"""
from __future__ import annotations

import json
import time
import uuid
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from uarc.core.config import UARCConfig
from uarc.core.types import InferenceRequest
from uarc.core.runtime import UARCRuntime


# ── Global runtime (set by run_server) ───────────────────────────────────────
_runtime: Optional[UARCRuntime] = None
_start_time = time.time()


class UARCRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, data: dict, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "uptime_s": round(time.time() - _start_time, 1)})
        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": "uarc-sim-7b", "object": "model", "created": int(_start_time),
                    "owned_by": "uarc", "permission": [], "root": "uarc-sim-7b",
                }]
            })
        elif self.path == "/status":
            self._send_json(_runtime.status() if _runtime else {"error": "not started"})
        elif self.path == "/metrics":
            self._send_json(_runtime.status()["performance"] if _runtime else {})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/v1/completions"):
            self._handle_completion()
        elif self.path == "/admin/cache/clear":
            if _runtime:
                _runtime.nsc = type(_runtime.nsc)(_runtime.cfg.nsc)
            self._send_json({"status": "cache cleared"})
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_completion(self):
        body = self._read_body()
        messages = body.get("messages", [])
        prompt = body.get("prompt", "")
        if messages:
            prompt = " ".join(m.get("content", "") for m in messages)
        max_tokens = body.get("max_tokens", 256)
        stream = body.get("stream", False)

        req = InferenceRequest(
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            max_new_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            self._handle_stream(req, body)
        else:
            resp = _runtime.infer(req)
            result = {
                "id": resp.request_id,
                "object": "chat.completion" if "/chat/" in self.path else "text_completion",
                "created": int(time.time()),
                "model": body.get("model", "uarc-sim-7b"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": resp.text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "total_tokens": resp.prompt_tokens + resp.completion_tokens,
                },
                "uarc_metadata": {
                    "route": resp.route_taken,
                    "latency_ms": resp.latency_ms,
                    "tokens_per_second": resp.tokens_per_second,
                    "compute_saved_pct": resp.compute_saved_pct,
                    "cache_hit": resp.cache_hit,
                },
            }
            self._send_json(result)

    def _handle_stream(self, req, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        model = body.get("model", "uarc-sim-7b")
        for chunk in _runtime.infer_stream(req):
            event_data = {
                "id": req.request_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            line = f"data: {json.dumps(event_data)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        done_data = {
            "id": req.request_id, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(done_data)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def run_server(port: int = 8000, vram_gb: float = 8,
               ram_gb: float = 32, n_layers: int = 32,
               backend: str = "auto", model_name: str = "llama3.2:1b"):
    """Start the UARC inference server."""
    global _runtime, _start_time

    cfg = UARCConfig()
    cfg.backend = backend
    cfg.model_name = model_name
    cfg.aivm.vram_mb = vram_gb * 1024
    cfg.aivm.ram_mb = ram_gb * 1024
    cfg.model.n_layers = n_layers
    cfg.port = port

    _runtime = UARCRuntime(cfg)
    _runtime.start()
    _start_time = time.time()

    server = HTTPServer(("0.0.0.0", port), UARCRequestHandler)
    print(f"  Listening on http://0.0.0.0:{port}")
    print(f"  Endpoints:")
    print(f"    POST /v1/chat/completions  — OpenAI chat")
    print(f"    POST /v1/completions       — OpenAI completion")
    print(f"    GET  /v1/models            — Model list")
    print(f"    GET  /health               — Health check")
    print(f"    GET  /status               — Full status")
    print(f"    GET  /metrics              — Prometheus metrics")
    print(f"\n  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        _runtime.stop()
        server.server_close()
