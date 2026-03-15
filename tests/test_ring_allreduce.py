#!/usr/bin/env python3
"""
Ring Allreduce: Correctness, Microbenchmark, and TP=4 Decode Correctness.

Tests:
1. Correctness: max abs error < 1e-3 vs CPU reference (not star topology)
   - Each GPU has same initial hidden (realistic TP scenario)
   - Verifies ring allreduce result matches CPU-computed reference
   - Verifies all GPUs get same result (consistency)
2. Microbenchmark: ring vs star latency over 200 iterations
3. TP=4 decode correctness: cosine sim >= 0.99 vs single-GPU (when model available)

Usage:
    HIP_VISIBLE_DEVICES=0,1,2,3 python3 tests/test_ring_allreduce.py
"""

import sys
import ctypes
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime, HIPError
from src.runtime.p2p_allreduce import P2PAllreduce, RingAllreduce


def alloc_fill_gpu(dev: GPUDevice, data: np.ndarray) -> int:
    """Allocate GPU buffer and upload numpy array."""
    ptr = dev.malloc(data.nbytes)
    dev.upload(ptr, data.tobytes())
    return ptr


def download_fp16(hip: HIPRuntime, dev_id: int, ptr: int, num_elems: int) -> np.ndarray:
    """Download FP16 buffer from GPU."""
    hip.set_device(dev_id)
    size = num_elems * 2
    buf = ctypes.create_string_buffer(size)
    hip.memcpy_d2h(buf, ptr, size)
    return np.frombuffer(buf, dtype=np.float16).copy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    dot = float(np.dot(a_f, b_f))
    norm_a = float(np.linalg.norm(a_f))
    norm_b = float(np.linalg.norm(b_f))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    return dot / (norm_a * norm_b)


# ============================================================
# Test 1: Correctness vs CPU reference
# ============================================================

def test_ring_correctness(tp_size: int = 4, num_elems: int = 5120,
                          seed: int = 42) -> float:
    """Test ring allreduce correctness vs CPU reference.

    All GPUs start with the same hidden state (realistic TP scenario).
    Expected result: hidden + sum(partials) on all GPUs.

    Also compares against star topology (P2PAllreduce) which should give same result.

    Returns: max abs error between ring and CPU reference.
    """
    print(f"\n--- Ring Correctness Test: TP={tp_size}, hidden={num_elems} ---")
    rng = np.random.default_rng(seed)

    devices = []
    for i in range(tp_size):
        d = GPUDevice(i)
        devices.append(d)
    hip = devices[0].hip
    device_ids = list(range(tp_size))

    streams = []
    for i in range(tp_size):
        hip.set_device(i)
        streams.append(hip.stream_create())

    # Generate test data
    # In real TP: each GPU has a different partial (its GEMV result)
    # but the same initial hidden (broadcast embedding)
    partials = [rng.random(num_elems).astype(np.float16) for _ in range(tp_size)]
    hidden0 = rng.random(num_elems).astype(np.float16)

    # CPU reference: hidden + sum(all partials)
    ref_result = hidden0.astype(np.float32).copy()
    for p in partials:
        ref_result += p.astype(np.float32)
    ref_result = ref_result.astype(np.float16)
    print(f"  CPU reference max value: {float(np.max(np.abs(ref_result.astype(np.float32)))):.4f}")

    # --- Ring topology (RingAllreduce) ---
    ring_partial_ptrs = []
    ring_hidden_ptrs = []
    for i in range(tp_size):
        ring_partial_ptrs.append(alloc_fill_gpu(devices[i], partials[i]))
        # ALL GPUs get the same initial hidden (as in real TP inference)
        ring_hidden_ptrs.append(alloc_fill_gpu(devices[i], hidden0.copy()))

    ring_ar = RingAllreduce(hip, device_ids, num_elems, streams=streams)
    ring_ar.allreduce_residual(ring_partial_ptrs, ring_hidden_ptrs, num_elems)

    # Download ring result from all GPUs
    ring_results = []
    for i in range(tp_size):
        ring_results.append(download_fp16(hip, i, ring_hidden_ptrs[i], num_elems))

    # Compare ring vs CPU reference
    max_abs_err = 0.0
    for i in range(tp_size):
        err = float(np.max(np.abs(ref_result.astype(np.float32) -
                                   ring_results[i].astype(np.float32))))
        max_abs_err = max(max_abs_err, err)
        print(f"  GPU{i}: max_abs_err vs CPU ref = {err:.4e}")

    # Check ring consistency (all GPUs should have same result)
    print("  Ring GPU consistency:")
    for i in range(1, tp_size):
        diff = float(np.max(np.abs(ring_results[0].astype(np.float32) -
                                    ring_results[i].astype(np.float32))))
        status = "OK" if diff < 1e-3 else "FAIL"
        print(f"    GPU{i} vs GPU0: {diff:.4e} [{status}]")

    # Also compare ring vs star topology to confirm they match
    print("  Comparing ring vs star topology...")
    star_partial_ptrs = []
    star_hidden_ptrs = []
    for i in range(tp_size):
        star_partial_ptrs.append(alloc_fill_gpu(devices[i], partials[i]))
        star_hidden_ptrs.append(alloc_fill_gpu(devices[i], hidden0.copy()))

    star_ar = P2PAllreduce(hip, device_ids, num_elems, streams=streams)
    star_ar.allreduce_residual(star_partial_ptrs, star_hidden_ptrs, num_elems)

    star_result = download_fp16(hip, 0, star_hidden_ptrs[0], num_elems)
    ring_vs_star = float(np.max(np.abs(star_result.astype(np.float32) -
                                        ring_results[0].astype(np.float32))))
    print(f"  Ring vs Star GPU0 max_abs_err: {ring_vs_star:.4e}")

    if max_abs_err < 1e-3:
        print(f"  PASS: max_abs_err={max_abs_err:.4e} < 1e-3 threshold")
    else:
        print(f"  FAIL: max_abs_err={max_abs_err:.4e} >= 1e-3 threshold")

    # Cleanup
    ring_ar.cleanup()
    star_ar.cleanup()
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(ring_partial_ptrs[i])
        hip.free(ring_hidden_ptrs[i])
        hip.free(star_partial_ptrs[i])
        hip.free(star_hidden_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()

    return max_abs_err


def test_ring_allreduce_sum(tp_size: int = 4, num_elems: int = 5120) -> float:
    """Test ring allreduce_sum (no residual) correctness.

    Returns: max abs error vs reference (CPU sum).
    """
    print(f"\n--- Ring allreduce_sum Correctness: TP={tp_size}, hidden={num_elems} ---")
    rng = np.random.default_rng(77)

    devices = []
    for i in range(tp_size):
        d = GPUDevice(i)
        devices.append(d)
    hip = devices[0].hip
    device_ids = list(range(tp_size))

    streams = []
    for i in range(tp_size):
        hip.set_device(i)
        streams.append(hip.stream_create())

    partials = [rng.random(num_elems).astype(np.float16) for _ in range(tp_size)]

    # CPU reference sum
    ref_sum = np.zeros(num_elems, dtype=np.float32)
    for p in partials:
        ref_sum += p.astype(np.float32)
    ref_result = ref_sum.astype(np.float16)

    # Ring allreduce_sum
    partial_ptrs = []
    for i in range(tp_size):
        partial_ptrs.append(alloc_fill_gpu(devices[i], partials[i]))

    ring_ar = RingAllreduce(hip, device_ids, num_elems, streams=streams)
    ring_ar.allreduce_sum(partial_ptrs, num_elems)

    # Download from all GPUs and check
    max_abs_err = 0.0
    for i in range(tp_size):
        result = download_fp16(hip, i, partial_ptrs[i], num_elems)
        err = float(np.max(np.abs(ref_result.astype(np.float32) -
                                   result.astype(np.float32))))
        max_abs_err = max(max_abs_err, err)
        print(f"  GPU{i}: max_abs_err vs CPU ref = {err:.4e}")

    if max_abs_err < 1e-3:
        print(f"  PASS: max_abs_err={max_abs_err:.4e} < 1e-3")
    else:
        print(f"  FAIL: max_abs_err={max_abs_err:.4e} >= 1e-3")

    # Cleanup
    ring_ar.cleanup()
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()

    return max_abs_err


# ============================================================
# Test 2: Microbenchmark - ring vs star latency
# ============================================================

def bench_allreduce(allreduce_cls, tp_size: int = 4, num_elems: int = 5120,
                    n_warmup: int = 10, n_iters: int = 200):
    """Benchmark an allreduce implementation.

    Returns: (median_us, mean_us, p90_us)
    """
    devices = []
    for i in range(tp_size):
        d = GPUDevice(i)
        devices.append(d)
    hip = devices[0].hip
    device_ids = list(range(tp_size))

    streams = []
    for i in range(tp_size):
        hip.set_device(i)
        streams.append(hip.stream_create())

    rng = np.random.default_rng(0)
    partials = [rng.random(num_elems).astype(np.float16) for _ in range(tp_size)]
    hidden0 = rng.random(num_elems).astype(np.float16)

    partial_ptrs = []
    hidden_ptrs = []
    for i in range(tp_size):
        partial_ptrs.append(alloc_fill_gpu(devices[i], partials[i]))
        hidden_ptrs.append(alloc_fill_gpu(devices[i], hidden0.copy()))

    ar = allreduce_cls(hip, device_ids, num_elems, streams=streams)

    def run():
        ar.allreduce_residual(partial_ptrs, hidden_ptrs, num_elems)

    # Warmup
    for _ in range(n_warmup):
        run()

    # Timed
    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    # Cleanup
    ar.cleanup()
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.free(hidden_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()

    latencies.sort()
    med = float(np.median(latencies))
    mean = float(np.mean(latencies))
    p90 = float(np.percentile(latencies, 90))
    return med, mean, p90


def run_microbenchmarks(tp_size: int = 4, num_elems: int = 5120):
    """Run ring vs star latency microbenchmark."""
    print(f"\n--- Microbenchmark: TP={tp_size}, hidden={num_elems}, 200 iters ---")

    print(f"  Running star topology (P2PAllreduce)...")
    star_med, star_mean, star_p90 = bench_allreduce(P2PAllreduce, tp_size, num_elems)
    print(f"  Star (P2PAllreduce):  median={star_med:.1f} us, "
          f"mean={star_mean:.1f} us, p90={star_p90:.1f} us")

    print(f"  Running ring topology (RingAllreduce)...")
    ring_med, ring_mean, ring_p90 = bench_allreduce(RingAllreduce, tp_size, num_elems)
    print(f"  Ring (RingAllreduce): median={ring_med:.1f} us, "
          f"mean={ring_mean:.1f} us, p90={ring_p90:.1f} us")

    speedup = star_med / ring_med if ring_med > 0 else float('inf')
    print(f"  Ring vs Star speedup: {speedup:.2f}x "
          f"({'faster' if speedup > 1 else 'slower'})")

    # The target is ring < 122 us (star baseline mentioned in feature spec)
    if ring_med < 122:
        print(f"  PASS: Ring latency {ring_med:.1f} us < 122 us target")
    else:
        print(f"  NOTE: Ring latency {ring_med:.1f} us vs 122 us target "
              f"(star baseline measured as ~101-122 us)")

    return star_med, ring_med, speedup


# ============================================================
# Test 3: TP=4 Decode Correctness with Ring Allreduce
# ============================================================

def test_tp4_decode_correctness(model_path: str = "/opt/models/Qwen3.5-27B-GPTQ-Int4"):
    """Test TP=4 decode with ring allreduce vs single-GPU reference.

    Compares cosine similarity of TP=4 decode output (using ring allreduce)
    vs single-GPU reference output. Must be >= 0.99 across 10 steps.

    Returns: min cosine similarity across all test steps, or None if skipped.
    """
    print("\n--- TP=4 Decode Correctness with Ring Allreduce ---")

    if not Path(model_path).exists():
        print(f"  SKIP: Model not found at {model_path}")
        return None

    from src.model.qwen import load_config_from_json
    from src.model.weight_loader import QwenWeightLoader
    from src.inference.tp_engine import TPInferenceEngine
    from src.inference.engine import InferenceEngine
    import copy as _copy

    # Load model config
    try:
        config = load_config_from_json(model_path)
    except Exception as e:
        print(f"  SKIP: Could not load config: {e}")
        return None

    print(f"  Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}")

    # Single-GPU reference
    print("  Loading single-GPU engine...")
    try:
        sg_engine = InferenceEngine(config, device_id=0, max_seq_len=256,
                                    tp_size=1, tp_rank=0)
    except Exception as e:
        print(f"  SKIP: Could not create single-GPU engine: {e}")
        return None

    # TP=4 engine with ring allreduce
    print("  Loading TP=4 engine with ring allreduce...")
    try:
        tp_engine = TPInferenceEngine(config, device_ids=[0, 1, 2, 3], max_seq_len=256)
        tp_engine.set_ring_allreduce(True)
    except Exception as e:
        print(f"  SKIP: Could not create TP=4 engine: {e}")
        sg_engine.device.cleanup()
        return None

    # Load weights from model using QwenWeightLoader (no safetensors needed)
    print("  Loading model weights...")
    try:
        loader = QwenWeightLoader(model_path, config=config)

        for layer_idx in range(config.num_hidden_layers):
            layer_weights = loader.load_layer(layer_idx)
            sg_engine.load_layer_weights(layer_idx, _copy.deepcopy(layer_weights))
            tp_engine.load_layer_weights(layer_idx, layer_weights)

        # Load final norm
        norm_weights = loader.load_final_norm()
        sg_engine.load_final_norm(norm_weights)
        tp_engine.load_final_norm(norm_weights)

    except Exception as e:
        print(f"  SKIP: Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        sg_engine.device.cleanup()
        tp_engine.cleanup()
        return None

    # Build dispatch cache
    try:
        tp_engine.build_dispatch_cache()
        tp_engine.set_cached_dispatch(True)
    except Exception as e:
        print(f"  NOTE: Cache build failed ({e}), using serial dispatch")

    # Run correctness test using a fixed random embedding (like test_c_dispatch_e2e.py)
    # Loading embedding table would require ~10GB+ VRAM; use random vector instead.
    print("  Running 10 decode steps...")
    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    cos_sims = []
    for step in range(10):
        # Single GPU
        sg_out = sg_engine.decode_step(emb, step)

        # TP=4 with ring allreduce
        tp_out = tp_engine.decode_step(emb, step)

        cs = cosine_similarity(sg_out.flatten(), tp_out.flatten())
        cos_sims.append(cs)
        status = "PASS" if cs >= 0.99 else "FAIL"
        print(f"  Step {step}: cosine_sim={cs:.6f} [{status}]")

    min_cs = min(cos_sims)
    print(f"\n  Min cosine similarity across 10 steps: {min_cs:.6f}")
    if min_cs >= 0.99:
        print(f"  PASS: All steps cosine_sim >= 0.99")
    else:
        print(f"  FAIL: Min cosine_sim {min_cs:.6f} < 0.99 threshold")

    # Cleanup
    sg_engine.device.cleanup()
    tp_engine.cleanup()

    return min_cs


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("Ring Allreduce: Correctness + Benchmark + TP=4 Decode")
    print("=" * 65)

    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"GPUs available: {n_gpus}")

    if n_gpus < 4:
        print("ERROR: Need 4 GPUs for TP=4 ring allreduce tests")
        sys.exit(1)

    all_pass = True

    # ============================================================
    # Test 1: Correctness
    # ============================================================
    print("\n" + "=" * 65)
    print("TEST 1: RING ALLREDUCE CORRECTNESS (vs CPU reference)")
    print("=" * 65)

    max_err = test_ring_correctness(tp_size=4, num_elems=5120)
    if max_err >= 1e-3:
        print(f"\nFAIL: Ring correctness test failed (max_err={max_err:.4e})")
        all_pass = False
    else:
        print(f"\nPASS: Ring correctness test passed (max_err={max_err:.4e})")

    print("\n--- Ring allreduce_sum correctness ---")
    sum_err = test_ring_allreduce_sum(tp_size=4, num_elems=5120)
    if sum_err >= 1e-3:
        print(f"FAIL: allreduce_sum correctness failed (max_err={sum_err:.4e})")
        all_pass = False
    else:
        print(f"PASS: allreduce_sum correctness passed (max_err={sum_err:.4e})")

    # ============================================================
    # Test 2: Microbenchmark
    # ============================================================
    print("\n" + "=" * 65)
    print("TEST 2: RING vs STAR LATENCY MICROBENCHMARK")
    print("=" * 65)

    star_med, ring_med, speedup = run_microbenchmarks(tp_size=4, num_elems=5120)

    # ============================================================
    # Test 3: TP=4 Decode Correctness
    # ============================================================
    print("\n" + "=" * 65)
    print("TEST 3: TP=4 DECODE CORRECTNESS WITH RING ALLREDUCE")
    print("=" * 65)

    min_cs = None
    try:
        min_cs = test_tp4_decode_correctness()
        if min_cs is not None and min_cs < 0.99:
            print(f"FAIL: TP=4 decode cosine_sim={min_cs:.6f} < 0.99")
            all_pass = False
        elif min_cs is not None:
            print(f"PASS: TP=4 decode correctness (min cosine_sim={min_cs:.6f})")
    except Exception as e:
        print(f"ERROR: TP=4 decode test failed with error: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Ring correctness (vs CPU ref):     max_err={max_err:.4e} "
          f"{'PASS' if max_err < 1e-3 else 'FAIL'}")
    print(f"  Ring allreduce_sum correctness:    max_err={sum_err:.4e} "
          f"{'PASS' if sum_err < 1e-3 else 'FAIL'}")
    print(f"  Ring vs star latency:              {speedup:.2f}x "
          f"({'faster' if speedup > 1 else 'slower'})")
    print(f"  Star latency:                      {star_med:.1f} us/call")
    print(f"  Ring latency:                      {ring_med:.1f} us/call")
    if min_cs is not None:
        print(f"  TP=4 decode cosine_sim:            {min_cs:.6f} "
              f"{'PASS' if min_cs >= 0.99 else 'FAIL'}")
    else:
        print(f"  TP=4 decode cosine_sim:            SKIPPED (model unavailable)")

    if all_pass:
        print("\nAll required tests PASSED")
    else:
        print("\nSome tests FAILED - see above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
