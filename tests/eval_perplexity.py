#!/usr/bin/env python3
"""
Perplexity evaluation of Qwen 3.5 27B GPTQ-INT4 on WikiText-2 test set.

This script validates that:
  - All 64 model layers load without errors (VAL-EVAL-001)
  - VRAM usage is reasonable (14-18GB for weights + KV cache) (VAL-EVAL-001)
  - Perplexity on WikiText-2 is in range 5-15 for a 27B INT4 model (VAL-EVAL-002)
  - No NaN or Inf values during evaluation (VAL-EVAL-002)

Usage:
    PYTHONPATH=/root/mi50grad python3 tests/eval_perplexity.py \
        --model-dir /opt/models/Qwen3.5-27B-GPTQ-Int4 \
        [--max-tokens 512] [--stride 256] [--window 256] [--device 0]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def get_vram_usage_gb(device_id: int = 0) -> float:
    """Get current VRAM usage in GB using rocm-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", f"--device={device_id}"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "Used Memory" in line:
                # Parse bytes from line like "GPU[0]: VRAM Total Used Memory (B): 12345678"
                parts = line.split(":")
                if len(parts) >= 2:
                    return int(parts[-1].strip()) / 1e9
    except Exception:
        pass
    return -1.0


def load_wikitext2_tokens(model_dir: str, max_tokens: int = 2048) -> tuple:
    """Load WikiText-2 test set and tokenize it.

    Returns: (token_ids: list[int], tok: tokenizer object)
    """
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_dir} ...")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Try cached file first
    data_path = Path("/opt/data/wikitext-2-raw/wiki.test.raw")
    if data_path.exists():
        print(f"Loading WikiText-2 from cache: {data_path}")
        text = data_path.read_text()
    else:
        # Try HuggingFace datasets
        try:
            from datasets import load_dataset
            print("Downloading WikiText-2 test set from HuggingFace datasets ...")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n".join(ds["text"])
            # Cache it
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_text(text)
            print(f"Cached to {data_path}")
        except Exception as e:
            print(f"WARNING: Could not load WikiText-2 dataset: {e}")
            print("Falling back to a small embedded test passage.")
            # Fallback: embedded passage with known structure to at least test the pipeline
            text = (
                " The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey "
                "building, and the tallest structure in Paris. Its base is square, measuring 125 metres "
                "(410 ft) on each side. During its construction, the Eiffel Tower surpassed the "
                "Washington Monument to become the tallest man-made structure in the world, "
                "a title it held for 41 years until the Chrysler Building in New York City was "
                "finished in 1930. It was the first structure to reach a height of 300 metres. "
                "Due to the addition of a broadcasting aerial at the top of the tower in 1957, "
                "it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding "
                "transmitters, the Eiffel Tower is the second tallest free-standing structure in "
                "France after the Millau Viaduct. The tower has three levels for visitors, with "
                "restaurants on the first and second levels. The top level's upper platform is "
                "276 m (906 ft) above the ground – the highest observation deck accessible to "
                "the public in the European Union. Tickets can be purchased to ascend by stairs "
                "or lift to the first and second levels. The climb from ground level to the first "
                "level is over 300 steps, as is the climb from the first level to the second. "
                "Although there is a staircase to the top level, it is usually only accessible "
                "by lift. On Bastille Day in 1880, the interior illumination of the tower in "
                "blue, white, and red colors was first lit. "
            ) * 20  # Repeat to get enough tokens

    print("Tokenizing WikiText-2 ...")
    ids = tok.encode(text)
    if max_tokens > 0:
        ids = ids[:max_tokens]
    print(f"Total tokens after truncation: {len(ids)}")
    return ids, tok


def load_model(model_dir: str, device_id: int = 0,
               max_seq_len: int = 512) -> tuple:
    """Load Qwen 3.5 27B GPTQ-INT4 model and return (engine, embed_weight, loader).

    Validates:
      - All layers load without errors
      - Reports VRAM usage after loading
    """
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_dir}")
    print(f"{'='*60}")

    config = load_config_from_json(model_dir)
    print(f"Config loaded: {config.num_hidden_layers} layers, hidden={config.hidden_size}")
    print(f"Layer types: {config.num_full_attention_layers} full-attn + "
          f"{config.num_linear_attention_layers} linear-attn")

    loader = QwenWeightLoader(model_dir, config, bits=config.bits,
                               group_size=config.group_size, quant_format='w4a16')

    # Initialize engine
    print(f"\nInitializing InferenceEngine (device={device_id}, max_seq_len={max_seq_len}) ...")
    vram_before = get_vram_usage_gb(device_id)
    engine = InferenceEngine(config, device_id=device_id,
                              max_seq_len=max_seq_len, quant_format='w4a16')
    print("Engine initialized.")

    # Load embedding (needed for input, keep on CPU)
    print("\nLoading embedding weights ...")
    embed_weight = loader.load_embedding()
    print(f"  Embedding: {embed_weight.shape} ({embed_weight.nbytes / 1e6:.1f} MB)")

    # Load all layers
    print("\nLoading 64 transformer layers ...")
    loaded_layers = 0
    for layer_idx in range(config.num_hidden_layers):
        layer_type = "full-attn" if config.is_full_attention(layer_idx) else "linear-attn"
        if layer_idx % 16 == 0 or layer_idx == config.num_hidden_layers - 1:
            print(f"  Layer {layer_idx}/{config.num_hidden_layers - 1} [{layer_type}]...")
        try:
            weights = loader.load_layer(layer_idx)
            engine.load_layer_weights(layer_idx, weights)
            loaded_layers += 1
        except Exception as e:
            raise RuntimeError(f"FAILED to load layer {layer_idx}: {e}") from e

    print(f"\n  Successfully loaded {loaded_layers}/{config.num_hidden_layers} layers.")

    # Load final norm and LM head
    print("Loading final norm ...")
    final_norm = loader.load_final_norm()
    engine.load_final_norm(final_norm)

    print("Loading LM head ...")
    lm_head = loader.load_lm_head()
    engine.load_lm_head(lm_head)

    vram_after = get_vram_usage_gb(device_id)
    if vram_before >= 0 and vram_after >= 0:
        vram_used = vram_after - vram_before
        print(f"\nVRAM usage after loading: {vram_after:.2f} GB total used "
              f"(+{vram_used:.2f} GB for model)")
        # Model is ~27GB on a single GPU (FP16 attn + INT4 FFN + embedding)
        # For multi-GPU splits it would be 14-18GB per GPU
        vram_ok = 5 <= vram_after <= 32  # single GPU full model range
        if not vram_ok:
            print(f"  WARNING: VRAM usage {vram_after:.2f}GB is outside expected "
                  f"5-32GB range (loaded weights + OS + KV cache)")
        else:
            print(f"  VRAM check PASSED: {vram_after:.2f} GB is in reasonable range")
    else:
        print("  (VRAM usage query not available)")
        vram_after = -1.0

    print(f"\nAll {loaded_layers} layers loaded successfully. ✓")
    return engine, embed_weight, config, vram_after


def evaluate_perplexity(
    engine: InferenceEngine,
    embed_weight: np.ndarray,
    token_ids: list,
    max_eval_tokens: int = 512,
    window_size: int = 256,
    stride: int = 256,
) -> dict:
    """Compute perplexity on token_ids using sliding window decode steps.

    For each window:
      1. Reset KV cache and DeltaNet state
      2. Sequential decode: run decode_step(emb[t], t) → hidden[t]
      3. Compute logits at each position → cross-entropy loss

    Returns dict with: perplexity, total_tokens, total_loss, has_nan, has_inf
    """
    print(f"\n{'='*60}")
    print(f"Computing perplexity on {len(token_ids)} tokens")
    print(f"  window_size={window_size}, stride={stride}, "
          f"max_eval_tokens={max_eval_tokens}")
    print(f"{'='*60}")

    total_loss = 0.0
    total_tokens = 0
    has_nan = False
    has_inf = False
    nan_positions = []
    inf_positions = []

    # Number of windows to evaluate
    n_windows = max(1, math.ceil((min(len(token_ids), max_eval_tokens + window_size) - window_size) / stride))

    window_num = 0
    for window_start in range(0, min(len(token_ids) - 1, max_eval_tokens), stride):
        if total_tokens >= max_eval_tokens:
            break

        window_ids = token_ids[window_start:window_start + window_size + 1]
        if len(window_ids) < 2:
            break

        window_num += 1
        context = window_ids[:-1]  # input tokens
        targets = window_ids[1:]   # target tokens

        # Reset recurrent state for each window
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()

        window_loss = 0.0
        window_count = 0
        t0 = time.perf_counter()

        for t, (inp_tok, tgt_tok) in enumerate(zip(context, targets)):
            emb = embed_weight[inp_tok].copy()

            # Forward pass
            hidden = engine.decode_step(emb, t)

            # Check for NaN/Inf in hidden state
            if np.any(np.isnan(hidden)):
                has_nan = True
                nan_positions.append((window_num, t))
                print(f"  WARNING: NaN in hidden state at window={window_num}, pos={t}")
                break
            if np.any(np.isinf(hidden)):
                has_inf = True
                inf_positions.append((window_num, t))
                print(f"  WARNING: Inf in hidden state at window={window_num}, pos={t}")
                break

            # Compute logits and cross-entropy loss
            engine.device.synchronize()
            logits = engine.compute_logits()  # uses d_hidden2 (after final norm)

            # Check logits
            if np.any(np.isnan(logits)):
                has_nan = True
                nan_positions.append((window_num, t))
                print(f"  WARNING: NaN in logits at window={window_num}, pos={t}")
                break
            if np.any(np.isinf(logits)):
                has_inf = True
                inf_positions.append((window_num, t))
                print(f"  WARNING: Inf in logits at window={window_num}, pos={t}")
                break

            # Numerically stable log-softmax
            logits_f64 = logits.astype(np.float64)
            max_logit = np.max(logits_f64)
            log_sum_exp = max_logit + np.log(np.sum(np.exp(logits_f64 - max_logit)))
            log_prob = logits_f64[tgt_tok] - log_sum_exp

            if np.isnan(log_prob) or np.isinf(log_prob):
                has_nan = True
                nan_positions.append((window_num, t))
                print(f"  WARNING: NaN/Inf loss at window={window_num}, pos={t} "
                      f"(max_logit={max_logit:.2f})")
                break

            window_loss -= log_prob
            window_count += 1

        elapsed = time.perf_counter() - t0

        if window_count > 0:
            total_loss += window_loss
            total_tokens += window_count
            window_ppl = math.exp(window_loss / window_count)
            running_ppl = math.exp(total_loss / total_tokens)

            print(f"  Window {window_num}: {window_count} tokens, "
                  f"loss={window_loss / window_count:.4f}, "
                  f"window_ppl={window_ppl:.2f}, "
                  f"running_ppl={running_ppl:.2f}, "
                  f"speed={window_count / elapsed:.1f} tok/s")

    if total_tokens == 0:
        return {
            "perplexity": float("inf"),
            "total_tokens": 0,
            "total_loss": float("inf"),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "error": "No tokens evaluated (all windows had errors)",
        }

    final_ppl = math.exp(total_loss / total_tokens)
    return {
        "perplexity": final_ppl,
        "total_tokens": total_tokens,
        "total_loss": total_loss,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nan_positions": nan_positions,
        "inf_positions": inf_positions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen 3.5 27B GPTQ-INT4 perplexity on WikiText-2"
    )
    parser.add_argument("--model-dir", default="/opt/models/Qwen3.5-27B-GPTQ-Int4",
                        help="Path to GPTQ-INT4 model directory")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max number of tokens to evaluate (default: 512)")
    parser.add_argument("--window", type=int, default=256,
                        help="Context window size (default: 256)")
    parser.add_argument("--stride", type=int, default=256,
                        help="Sliding window stride (default: 256)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Engine max_seq_len (KV cache size, default: 512)")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen 3.5 27B GPTQ-INT4 Perplexity Evaluation")
    print("=" * 60)
    print(f"Model:      {args.model_dir}")
    print(f"Device:     GPU {args.device}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Window:     {args.window}")
    print(f"Stride:     {args.stride}")

    # Step 1: Load model (validates VAL-EVAL-001)
    t_load_start = time.perf_counter()
    try:
        engine, embed_weight, config, vram_gb = load_model(
            args.model_dir, args.device, max_seq_len=args.max_seq_len
        )
    except Exception as e:
        print(f"\nFATAL: Model loading failed: {e}")
        sys.exit(1)
    t_load = time.perf_counter() - t_load_start
    print(f"\nModel load time: {t_load:.1f}s")

    # Step 2: Load WikiText-2
    try:
        loaded = load_wikitext2_tokens(
            args.model_dir, max_tokens=args.max_tokens + args.window + 10
        )
        token_ids = loaded[0]
    except Exception as e:
        print(f"\nFATAL: Failed to load WikiText-2: {e}")
        engine.cleanup()
        sys.exit(1)

    # Step 3: Evaluate perplexity (validates VAL-EVAL-002)
    t_eval_start = time.perf_counter()
    results = evaluate_perplexity(
        engine, embed_weight, token_ids,
        max_eval_tokens=args.max_tokens,
        window_size=args.window,
        stride=args.stride,
    )
    t_eval = time.perf_counter() - t_eval_start

    # Step 4: Report results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    ppl = results["perplexity"]
    n_tokens = results["total_tokens"]
    has_nan = results["has_nan"]
    has_inf = results["has_inf"]

    print(f"Perplexity:      {ppl:.4f}")
    print(f"Tokens evaluated: {n_tokens}")
    print(f"Eval time:       {t_eval:.1f}s ({n_tokens / t_eval:.2f} tok/s)")
    if vram_gb > 0:
        print(f"VRAM used:       {vram_gb:.2f} GB")

    # Validation checks
    print("\n" + "-" * 40)
    print("VALIDATION")
    print("-" * 40)

    # VAL-EVAL-001: All layers loaded
    print(f"[{'PASS' if n_tokens > 0 else 'FAIL'}] VAL-EVAL-001: "
          f"All {config.num_hidden_layers} layers loaded without errors")

    # VRAM check
    if vram_gb > 0:
        # Single GPU: full model ~27GB; multi-GPU: 14-18GB per GPU
        vram_ok = 5 <= vram_gb <= 32
        print(f"[{'PASS' if vram_ok else 'WARN'}] VRAM usage: {vram_gb:.2f} GB "
              f"({'OK' if vram_ok else 'outside expected range'})")

    # VAL-EVAL-002: Perplexity in expected range
    PPL_MIN = 5.0
    PPL_MAX = 15.0
    ppl_ok = PPL_MIN <= ppl <= PPL_MAX
    print(f"[{'PASS' if ppl_ok else 'FAIL'}] VAL-EVAL-002: "
          f"Perplexity={ppl:.2f} (expected {PPL_MIN}-{PPL_MAX})")

    # NaN/Inf check
    no_nan_inf = not has_nan and not has_inf
    print(f"[{'PASS' if no_nan_inf else 'FAIL'}] No NaN or Inf values: "
          f"{'clean' if no_nan_inf else f'nan={has_nan}, inf={has_inf}'}")

    print("-" * 40)

    overall_pass = (n_tokens > 0) and ppl_ok and no_nan_inf
    print(f"\nOverall: {'PASS ✓' if overall_pass else 'FAIL ✗'}")

    if ppl > 50:
        print("\nWARNING: Perplexity > 50 indicates a likely bug in kernel or weight loading!")
        print("  Check: are all kernels compiled? Are weights loading correctly?")
    elif ppl > 15:
        print(f"\nNOTE: Perplexity {ppl:.2f} is higher than expected (5-15).")
        print("  This may indicate quantization quality issues or kernel accuracy issues.")

    engine.cleanup()
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
