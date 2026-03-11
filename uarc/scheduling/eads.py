"""
UARC Module 7 (Replacement): Entropy-Aware Dynamic Speculator (EADS)
=====================================================================
Replaces naive routing with Information-Theory-backed dynamic speculative decoding.
Calculates optimal lookahead depth (K) dynamically based on token complexity limits
and historically smoothed Acceptance Rates to maximize GPU batch efficiency.
"""
from __future__ import annotations

import math
import logging
from typing import Dict, Any, Tuple, List
from collections import defaultdict

from uarc.core.config import EADSConfig

logger = logging.getLogger("uarc.eads")

class EADSScheduler:
    """
    Stateful scheduler for speculative continuous batching.
    Tracks the moving average of Acceptance Rate and approximated Entropy (via PPR)
    to dynamically increase or decrease the draft depth (K).
    """
    def __init__(self, cfg: EADSConfig | None = None):
        self.cfg = cfg or EADSConfig()
        # State tracking per active sequence UUID
        # { seq_id: {'current_k': int, 'ema_acceptance': float, 'ema_entropy': float} }
        self.seq_states: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._total_drafts = 0
        self._total_accepted = 0
        self._wasted_drafts = 0
        self._oracle_fallbacks = 0

    def init_sequence(self, req_id: str):
        """Register a new request for EADS tracking."""
        self.seq_states[req_id] = {
            'current_k': self.cfg.base_k,
            'ema_acceptance_rate': 0.8,
            # Since raw SoftMax entropy is abstracted by the backend, 
            # we use TokenDifficultyEstimator's Perplexity approximation
            'ema_entropy': 2.0 
        }

    def update_and_get_k(self, req_id: str, drafted_k: int, accepted_k: int, estimated_ppl: float) -> int:
        """
        The dynamic decision matrix: 
        Evaluates draft efficiency and token difficulty to adjust K for the next cascade step.
        """
        if req_id not in self.seq_states:
            self.init_sequence(req_id)
            
        state = self.seq_states[req_id]
        
        # 1. Update Acceptance Rate EMA
        step_acceptance = (accepted_k / drafted_k) if drafted_k > 0 else 1.0
        state['ema_acceptance_rate'] = (
            self.cfg.ema_alpha * step_acceptance + 
            (1 - self.cfg.ema_alpha) * state['ema_acceptance_rate']
        )
        
        # 2. Update Entropy (Perplexity) EMA
        state['ema_entropy'] = (
            self.cfg.ema_alpha * estimated_ppl + 
            (1 - self.cfg.ema_alpha) * state['ema_entropy']
        )
        
        # 3. Dynamic Routing Decision Matrix
        
        # CASE A: High Entropy OR Terrible Historical Acceptance (e.g. Code, Math)
        # Action: Cut K to avoid validating garbage draft tokens
        if state['ema_entropy'] > self.cfg.entropy_threshold or state['ema_acceptance_rate'] < 0.3:
            new_k = max(self.cfg.min_k, state['current_k'] - 2)
            if new_k == self.cfg.min_k:
                self._oracle_fallbacks += 1
                
        # CASE B: Low Entropy AND Excellent Historical Acceptance (e.g. Boilerplate, Greetings)
        # Action: Expand K to sprint ahead
        elif state['ema_entropy'] < (self.cfg.entropy_threshold * 0.5) and state['ema_acceptance_rate'] > 0.8:
            new_k = min(self.cfg.max_k, state['current_k'] + 1)
            
        # CASE C: Ambiguous Zone
        # Action: Hold steady
        else:
            new_k = state['current_k']
            
        state['current_k'] = new_k
        
        # Update metrics
        self._total_drafts += drafted_k
        self._total_accepted += accepted_k
        self._wasted_drafts += (drafted_k - accepted_k)
        
        return new_k

    def stats(self) -> dict:
        """Report efficiency metrics."""
        return {
            "total_drafted_tokens": self._total_drafts,
            "total_accepted_tokens": self._total_accepted,
            "wasted_draft_tokens": self._wasted_drafts,
            "global_acceptance_rate": round(self._total_accepted / max(self._total_drafts, 1), 3),
            "oracle_fallbacks (K=1)": self._oracle_fallbacks,
            "active_tracking": len(self.seq_states)
        }
