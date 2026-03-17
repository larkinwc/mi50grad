"""
End-to-end text generation for Qwen 3.5 27B on MI50.

Ties together: tokenizer, embedding, inference engine, LM head, and sampling.
Supports both prefill (prompt processing) and decode (token generation).
"""

import ctypes
import numpy as np
import time
from pathlib import Path
from typing import Optional, Generator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.engine import InferenceEngine
from src.inference.sampler import SamplingParams, sample_token
from src.model.qwen import QwenConfig
from src.runtime.hip_dispatch import GPUDevice


class TextGenerator:
    """End-to-end text generation with prefill + decode."""

    def __init__(self, engine: InferenceEngine,
                 embed_weight: np.ndarray,
                 lm_head_weight: np.ndarray,
                 tokenizer=None):
        """
        Args:
            engine: initialized InferenceEngine with loaded weights
            embed_weight: [vocab_size, hidden_dim] FP16 embedding table
            lm_head_weight: [vocab_size, hidden_dim] FP16 LM head weight
            tokenizer: HuggingFace tokenizer (or compatible)
        """
        self.engine = engine
        self.config = engine.config
        self.device = engine.device
        self.tokenizer = tokenizer

        # Upload embedding table to GPU
        self.d_embed = self.device.malloc(embed_weight.nbytes)
        self.device.upload(self.d_embed, embed_weight.tobytes())
        self.embed_weight = embed_weight  # keep CPU copy for lookup

        # Upload LM head weight to GPU
        self.d_lm_head = self.device.malloc(lm_head_weight.nbytes)
        self.device.upload(self.d_lm_head, lm_head_weight.tobytes())
        self.lm_head_weight = lm_head_weight  # keep CPU copy

        # Scratch for LM head output
        self.d_logits = self.device.malloc(self.config.vocab_size * 2)  # FP16

    def _embed_token(self, token_id: int) -> np.ndarray:
        """Look up token embedding. Returns [hidden_dim] FP16."""
        return self.embed_weight[token_id].copy()

    def _embed_tokens(self, token_ids: list) -> np.ndarray:
        """Look up multiple token embeddings. Returns [seq_len, hidden_dim] FP16."""
        return self.embed_weight[token_ids].copy()

    def _lm_head(self, hidden_state: np.ndarray) -> np.ndarray:
        """Project hidden state to vocab logits using GPU GEMV.

        Args:
            hidden_state: [hidden_dim] FP16

        Returns:
            [vocab_size] FP32 logits
        """
        return self.engine.compute_logits(hidden_state)

    def prefill(self, token_ids: list) -> np.ndarray:
        """Process prompt tokens through all layers.

        Uses the prefill path (GEMM + FlashAttention) for efficiency.
        Falls back to sequential decode steps if prefill kernels aren't available.

        Args:
            token_ids: list of prompt token IDs

        Returns:
            [hidden_dim] FP16 hidden state of last token
        """
        if hasattr(self.engine, 'prefill_step'):
            # Use optimized prefill path
            embeddings = self._embed_tokens(token_ids)
            return self.engine.prefill_step(embeddings)

        # Fallback: sequential decode steps
        hidden = None
        for i, tid in enumerate(token_ids):
            emb = self._embed_token(tid)
            hidden = self.engine.decode_step(emb, i)
        return hidden

    def generate(self, prompt: str,
                 params: Optional[SamplingParams] = None,
                 stream: bool = False) -> str:
        """Generate text from a prompt.

        Args:
            prompt: input text
            params: sampling parameters
            stream: if True, yield tokens as they are generated

        Returns:
            Generated text (excluding prompt)
        """
        if params is None:
            params = SamplingParams()

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set. Pass tokenizer to TextGenerator.")

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        generated_ids = []

        # Prefill
        hidden = self.prefill(input_ids)

        # Decode loop
        position = len(input_ids)
        for _ in range(params.max_tokens):
            # LM head: hidden -> logits
            logits = self._lm_head(hidden)

            # Sample
            token_id = sample_token(logits, params,
                                    past_tokens=input_ids + generated_ids)

            # Check stop conditions
            if params.stop_token_ids and token_id in params.stop_token_ids:
                break
            if token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)

            # Decode next token
            emb = self._embed_token(token_id)
            hidden = self.engine.decode_step(emb, position)
            position += 1

        return self.tokenizer.decode(generated_ids)

    def generate_streaming(self, prompt: str,
                           params: Optional[SamplingParams] = None
                           ) -> Generator[str, None, None]:
        """Generate text token by token, yielding each new token's text.

        Args:
            prompt: input text
            params: sampling parameters

        Yields:
            Each generated token as decoded text
        """
        if params is None:
            params = SamplingParams()

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set.")

        input_ids = self.tokenizer.encode(prompt)
        generated_ids = []

        # Prefill
        hidden = self.prefill(input_ids)

        # Decode loop
        position = len(input_ids)
        for _ in range(params.max_tokens):
            logits = self._lm_head(hidden)
            token_id = sample_token(logits, params,
                                    past_tokens=input_ids + generated_ids)

            if params.stop_token_ids and token_id in params.stop_token_ids:
                break
            if token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            token_text = self.tokenizer.decode([token_id])
            yield token_text

            emb = self._embed_token(token_id)
            hidden = self.engine.decode_step(emb, position)
            position += 1

    def benchmark(self, prompt: str, max_tokens: int = 100) -> dict:
        """Benchmark generation speed.

        Returns dict with: prefill_time, prefill_toks, decode_time,
        decode_toks, prefill_tok_s, decode_tok_s, ttft
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set.")

        input_ids = self.tokenizer.encode(prompt)
        params = SamplingParams(temperature=0, max_tokens=max_tokens)

        # Prefill timing
        t0 = time.perf_counter()
        hidden = self.prefill(input_ids)
        self.device.synchronize()
        t_prefill = time.perf_counter() - t0

        # First token
        logits = self._lm_head(hidden)
        token_id = sample_token(logits, params)
        t_ttft = time.perf_counter() - t0

        generated_ids = [token_id]
        position = len(input_ids)

        # Decode timing
        t_decode_start = time.perf_counter()
        for _ in range(max_tokens - 1):
            emb = self._embed_token(token_id)
            hidden = self.engine.decode_step(emb, position)
            self.device.synchronize()

            logits = self._lm_head(hidden)
            token_id = sample_token(logits, params,
                                    past_tokens=input_ids + generated_ids)

            if token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            position += 1

        t_decode = time.perf_counter() - t_decode_start
        n_decode = len(generated_ids) - 1  # exclude first token (counted in prefill)

        return {
            "prefill_time_ms": t_prefill * 1000,
            "prefill_tokens": len(input_ids),
            "prefill_tok_s": len(input_ids) / t_prefill if t_prefill > 0 else 0,
            "decode_time_ms": t_decode * 1000,
            "decode_tokens": n_decode,
            "decode_tok_s": n_decode / t_decode if t_decode > 0 else 0,
            "ttft_ms": t_ttft * 1000,
            "total_generated": len(generated_ids),
        }

    def cleanup(self):
        """Free GPU resources."""
        self.device.free(self.d_embed)
        self.device.free(self.d_lm_head)
        self.device.free(self.d_logits)


def load_and_generate(model_dir: str, prompt: str,
                      device_ids: Optional[list] = None,
                      max_tokens: int = 256,
                      temperature: float = 0.7) -> str:
    """Convenience function: load model and generate text.

    Args:
        model_dir: path to GPTQ safetensors model directory
        prompt: input text
        device_ids: list of GPU IDs (default: [0]). If >1 device, uses tensor parallelism.
        max_tokens: maximum tokens to generate
        temperature: sampling temperature

    Returns:
        Generated text
    """
    from src.model.weight_loader import QwenWeightLoader
    from src.model.qwen import QwenConfig, load_config_from_json

    if device_ids is None:
        device_ids = [0]

    # Load config from model directory
    config = load_config_from_json(model_dir)
    qwen_loader = QwenWeightLoader(model_dir, config=config,
                                    bits=config.bits,
                                    group_size=config.group_size)

    # Initialize engine (single-GPU or tensor-parallel)
    if len(device_ids) > 1:
        from src.inference.tp_engine import TPInferenceEngine
        engine = TPInferenceEngine(config, device_ids=device_ids)
    else:
        engine = InferenceEngine(config, device_id=device_ids[0])

    # Load weights layer by layer
    for layer_idx in range(config.num_hidden_layers):
        weights = qwen_loader.load_layer(layer_idx)
        engine.load_layer_weights(layer_idx, weights)

    # Load final RMSNorm
    final_norm = qwen_loader.load_final_norm()
    engine.load_final_norm(final_norm)

    # Load embedding and LM head
    embed_weight = qwen_loader.load_embedding()
    lm_head_weight = qwen_loader.load_lm_head()

    # Tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except ImportError:
        raise ImportError("transformers package required for tokenizer")

    # Generate
    generator = TextGenerator(engine, embed_weight, lm_head_weight, tokenizer)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    try:
        return generator.generate(prompt, params)
    finally:
        generator.cleanup()
        engine.cleanup()


def load_and_generate_eagle(model_dir: str, prompt: str,
                             device_ids: Optional[list] = None,
                             max_tokens: int = 256,
                             temperature: float = 0.0,
                             k_draft: int = 4,
                             verbose: bool = False) -> Tuple[str, dict]:
    """Load model and generate text using EAGLE speculative decoding.
    
    EAGLE uses the target model's own hidden states and lm_head to generate
    draft tokens, eliminating the need for a separate draft model.
    
    Args:
        model_dir: path to GPTQ safetensors model directory
        prompt: input text
        device_ids: list of GPU IDs (default: [0])
        max_tokens: maximum tokens to generate
        temperature: sampling temperature (0.0 for greedy)
        k_draft: number of draft tokens per iteration (default: 4)
        verbose: if True, print debugging info
    
    Returns:
        Tuple of (generated_text, stats_dict)
    """
    from src.model.weight_loader import QwenWeightLoader
    from src.model.qwen import QwenConfig, load_config_from_json
    from src.inference.speculative import EagleSpeculativeGenerator
    
    if device_ids is None:
        device_ids = [0]
    
    # Load config from model directory
    config = load_config_from_json(model_dir)
    qwen_loader = QwenWeightLoader(model_dir, config=config,
                                    bits=config.bits,
                                    group_size=config.group_size)
    
    # Initialize engine (single-GPU or tensor-parallel)
    if len(device_ids) > 1:
        from src.inference.tp_engine import TPInferenceEngine
        engine = TPInferenceEngine(config, device_ids=device_ids)
    else:
        engine = InferenceEngine(config, device_id=device_ids[0])
    
    # Load weights layer by layer
    for layer_idx in range(config.num_hidden_layers):
        weights = qwen_loader.load_layer(layer_idx)
        engine.load_layer_weights(layer_idx, weights)
    
    # Load final RMSNorm
    final_norm = qwen_loader.load_final_norm()
    engine.load_final_norm(final_norm)
    
    # Load embedding and LM head
    embed_weight = qwen_loader.load_embedding()
    lm_head_weight = qwen_loader.load_lm_head()
    
    # Tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except ImportError:
        raise ImportError("transformers package required for tokenizer")
    
    # Generate with EAGLE speculative decoding
    generator = EagleSpeculativeGenerator(
        engine, embed_weight, lm_head_weight, tokenizer,
        k_draft=k_draft, temperature=temperature
    )
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    
    try:
        return generator.generate(prompt, params, verbose=verbose)
    finally:
        generator.cleanup()
        engine.cleanup()
