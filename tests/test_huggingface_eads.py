import pytest
import torch
from uarc.core.config import UARCConfig
from uarc.core.runtime import UARCRuntime
from uarc.core.types import InferenceRequest, RouteTarget

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for real model testing")
def test_huggingface_speculative_integration():
    """
    Verifies that the UARC Runtime can load a HuggingFace backend with a draft model
    and execute a speculative inference request.
    """
    # Use extremely small models for testing
    cfg = UARCConfig(
        backend="hf",
        model_name="sshleifer/tiny-gpt2",
        draft_model_name="sshleifer/tiny-gpt2",
        enable_eads=True
    )
    
    runtime = UARCRuntime(cfg)
    runtime.start()
    
    try:
        # Request with RouteTarget.DRAFT to trigger the speculative engine
        request = InferenceRequest(
            prompt="Hello UARC speculative engine,",
            max_new_tokens=10,
            request_id="test_spec_0"
        )
        # Manually force the difficulty and route if TDE isn't loaded with a real model
        # Normally TDE does this, but here we want to test the execution path
        
        # We need to ensure the runtime 'infer' loop hits the DRAFT path
        # Let's mock a high difficulty but keep route as DRAFT
        from uarc.core.types import RoutingDecision
        
        # We'll use a wrapper to force the decision
        original_estimate = runtime.tde.estimate
        runtime.tde.estimate = lambda ids: RoutingDecision(
            route=RouteTarget.DRAFT,
            estimated_ppl=2.0,
            confidence=1.0, 
            latency_ms=0.0,
            compute_saved_pct=50.0
        )
        
        response = runtime.infer(request)
        
        assert response.text is not None
        assert response.route_taken == "draft"
        assert response.completion_tokens > 0
        
        # Verify EADS stats actually recorded something
        stats = runtime.eads.stats()
        assert stats["active_tracking"] > 0
        
    finally:
        runtime.stop()

if __name__ == "__main__":
    # If run manually
    test_huggingface_speculative_integration()
