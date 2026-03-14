"""
Token sampling for autoregressive generation.

Supports temperature, top-k, and top-p (nucleus) sampling.
Runs on CPU (downloads logits from GPU, samples in numpy).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    max_tokens: int = 256
    stop_token_ids: Optional[list] = None


def sample_token(logits: np.ndarray, params: SamplingParams,
                 past_tokens: Optional[list] = None) -> int:
    """Sample a single token from logits.

    Args:
        logits: [vocab_size] FP16 or FP32 array
        params: sampling parameters
        past_tokens: list of previously generated token IDs (for repetition penalty)

    Returns:
        Selected token ID
    """
    logits = logits.astype(np.float32).copy()

    # Repetition penalty
    if past_tokens and params.repetition_penalty != 1.0:
        for tid in set(past_tokens):
            if tid < len(logits):
                if logits[tid] > 0:
                    logits[tid] /= params.repetition_penalty
                else:
                    logits[tid] *= params.repetition_penalty

    # Temperature
    if params.temperature != 1.0 and params.temperature > 0:
        logits /= params.temperature

    # Greedy
    if params.temperature == 0:
        return int(np.argmax(logits))

    # Top-k filtering
    if params.top_k > 0 and params.top_k < len(logits):
        top_k_idx = np.argpartition(logits, -params.top_k)[-params.top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_idx] = logits[top_k_idx]
        logits = mask

    # Softmax
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    # Top-p (nucleus) filtering
    if params.top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, params.top_p) + 1
        # Zero out everything below cutoff
        mask = np.zeros_like(probs)
        mask[sorted_idx[:cutoff]] = probs[sorted_idx[:cutoff]]
        probs = mask
        probs /= probs.sum()

    # Sample
    return int(np.random.choice(len(probs), p=probs))
