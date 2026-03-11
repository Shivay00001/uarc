import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Any
from uarc.scheduling.eads import EADSScheduler
from uarc.core.config import EADSConfig
import logging

logger = logging.getLogger("uarc.eads_engine")

def crop_cache(cache: Any, keep_length: int) -> Any:
    """
    Slices the HuggingFace KV cache to rollback rejected speculative tokens.
    Supports both traditional tuple of tuples and newer DynamicCache objects.
    """
    if cache is None:
        return None
    
    # HF transformers >= 4.36 DynamicCache
    if hasattr(cache, "crop"):
        cache.crop(keep_length)
        return cache
        
    # Legacy tuple format: ( (k0, v0), (k1, v1), ... )
    if isinstance(cache, tuple) or isinstance(cache, list):
        new_cache = []
        for layer in cache:
            k, v = layer
            new_k = k[:, :, :keep_length, :]
            new_v = v[:, :, :keep_length, :]
            new_cache.append((new_k, new_v))
        return tuple(new_cache)
        
    return cache

class EADSSpeculativeEngine:
    """
    Production-grade Entropy-Aware Dynamic Speculative Decoding Engine.
    Uses a true draft model and target model to accelerate inference while 
    using the EADS scheduler to dynamically resize the draft depth (K).
    """
    def __init__(self, target_model: Any, draft_model: Any, scheduler: Optional[EADSScheduler] = None):
        self.target = target_model
        self.draft = draft_model
        # Configure EADS default values for Speculative Decoding tracking 
        self.scheduler = scheduler or EADSScheduler(EADSConfig(base_k=4, min_k=1, max_k=12))
        
        self.device = target_model.device
        self.target.eval()
        self.draft.eval()
        
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, req_id: str = "req_spec_0", debug: bool = False) -> torch.Tensor:
        """
        Generates tokens using speculative decoding + EADS.
        Returns the output sequence.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        self.scheduler.init_sequence(req_id)
        
        n_generated = 0
        N_current = input_ids.size(1)
        
        # 1. Initial Prefill
        N_initial = input_ids.size(1)
        target_outputs = self.target(input_ids, use_cache=True)
        target_kv = target_outputs.past_key_values
        
        draft_outputs = self.draft(input_ids, use_cache=True)
        draft_kv = draft_outputs.past_key_values
        
        # First next token from target prefill
        next_tok = torch.argmax(target_outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        n_generated += 1
        
        while n_generated < max_new_tokens:
            state = self.scheduler.seq_states[req_id]
            current_k = min(state['current_k'], max_new_tokens - n_generated)
            
            if current_k <= 0:
                break
                
            # --- Phase A: Drafting ---
            draft_tokens = []
            draft_input = next_tok
            
            for _ in range(current_k):
                d_out = self.draft(draft_input, past_key_values=draft_kv, use_cache=True)
                draft_kv = d_out.past_key_values
                y_i = torch.argmax(d_out.logits[:, -1, :], dim=-1).unsqueeze(-1)
                draft_tokens.append(y_i)
                draft_input = y_i
                
            draft_tensor = torch.cat(draft_tokens, dim=1)  # [1, K]
            
            # --- Phase B: Verification ---
            # Target sees the last validated token plus all K drafted tokens
            target_input = torch.cat([next_tok, draft_tensor], dim=1)
            t_out = self.target(target_input, past_key_values=target_kv, use_cache=True)
            target_kv = t_out.past_key_values
            
            target_logits = t_out.logits  # [1, K+1, vocab_size]
            
            # Compute Mean Sequence Entropy 
            probs = F.softmax(target_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
            
            # --- Phase C: Greedy Matching ---
            m = 0
            for i in range(current_k):
                expected = torch.argmax(target_logits[:, i, :], dim=-1).unsqueeze(-1)
                actual = draft_tokens[i]
                if torch.equal(expected, actual):
                    m += 1
                else:
                    break
                    
            # True next token from the first mismatch
            true_next = torch.argmax(target_logits[:, m, :], dim=-1).unsqueeze(-1)
            
            # Push efficiency stats back to the EADS Engine to resize K
            self.scheduler.update_and_get_k(req_id, current_k, m, entropy)
            
            if debug:
                print(f"EADS [{req_id}] | K:{current_k} | Acc:{m} | Entropy:{entropy:.2f} -> Next K: {self.scheduler.seq_states[req_id]['current_k']}")
            
            # Append valid sequence
            if m > 0:
                valid_draft = draft_tensor[:, :m]
                accepted_seq = torch.cat([valid_draft, true_next], dim=1)
            else:
                accepted_seq = true_next
                
            input_ids = torch.cat([input_ids, accepted_seq], dim=1)
            
            next_tok = true_next
            n_generated += (m + 1)
            
            # --- Phase D: Cache Cropping / State Rollback ---
            keep_len_target = N_current + 1 + m
            target_kv = crop_cache(target_kv, keep_len_target)
            
            keep_len_draft = N_current + 1 + m
            draft_kv = crop_cache(draft_kv, keep_len_draft)
            
            N_current = keep_len_target
            
        return input_ids[:, :N_initial + max_new_tokens]
