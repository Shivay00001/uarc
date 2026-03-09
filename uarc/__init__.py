"""
UARC — Unified Adaptive Runtime Core
======================================
Next-generation AI inference engine with 7 adaptive modules, portable,
OpenAI-compatible. Pure Python, zero external dependencies.

Quick start:
    from uarc import UARCRuntime, UARCConfig, InferenceRequest

    rt = UARCRuntime(UARCConfig())
    rt.start()
    resp = rt.infer(InferenceRequest(request_id="r1", prompt="Hello!"))
    print(resp.text)
    rt.stop()
"""

__version__ = "0.1.0"

from uarc.core.config import UARCConfig
from uarc.core.types import InferenceRequest, InferenceResponse
from uarc.core.runtime import UARCRuntime

__all__ = ["UARCRuntime", "UARCConfig", "InferenceRequest", "InferenceResponse"]
