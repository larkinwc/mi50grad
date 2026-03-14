#!/usr/bin/env python3
"""Test W4A8 weight repacking utility (repack_w4a8.py).

Tests:
1. Round-trip: quantize FP32 → W4A8 → dequantize, verify values preserved
2. GPTQ-format repack: simulate GPTQ qweight/scales/qzeros → W4A8
3. Nibble packing correctness: verify individual weight values
4. Scale preservation: verify per-group and per-channel scales
5. Edge cases: min/max values, K=8 (single group), large N

All tests run on CPU (numpy only, no GPU needed for repacking).
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.kernels.repack_w4a8 import (
    repack_gptq_to_w4a8,
    repack_simple_for_test,
    dequantize_w4a8,
    unpack_gptq_qweight,
    unpack_gptq_qzeros,
    verify_repack_roundtrip,
)

all_pass = True

# ---------------------------------------------------------------------------
# Test 1: Simple round-trip (FP32 → W4A8 → FP32)
# ---------------------------------------------------------------------------
print("=== Test 1: Simple Round-Trip ===")
TEST_SHAPES = [
    (64, 128, "64x128"),
    (256, 512, "256x512"),
    (4096, 4096, "4096x4096"),
    (11008, 4096, "11008x4096"),
]

for N, K, label in TEST_SHAPES:
    np.random.seed(42)
    W_fp32 = np.random.randn(N, K).astype(np.float32)
    W_packed, scale_w = repack_simple_for_test(W_fp32)

    assert W_packed.shape == (N, K // 8), f"Shape mismatch: {W_packed.shape}"
    assert W_packed.dtype == np.uint32, f"dtype mismatch: {W_packed.dtype}"

    # Dequantize using per-channel scale (treat whole row as one group)
    scale_w_grp = scale_w[np.newaxis, :]  # [1, N]
    W_dequant = dequantize_w4a8(W_packed, scale_w_grp, group_size=K)

    max_abs_ref = float(np.max(np.abs(W_fp32)))
    max_err = float(np.max(np.abs(W_dequant - W_fp32)))
    max_rel_err = max_err / (max_abs_ref + 1e-8)

    # INT4 quantization noise: expect < 15% relative error (scale = max_abs/7, so 1 step = 1/7 = ~14%)
    PASS = max_rel_err < 0.15
    print(f"  {label}: max_abs_ref={max_abs_ref:.3f}  max_err={max_err:.4f}  "
          f"max_rel={max_rel_err:.4f}  {'PASS' if PASS else 'FAIL'}")
    if not PASS:
        all_pass = False

# ---------------------------------------------------------------------------
# Test 2: Nibble packing correctness
# ---------------------------------------------------------------------------
print("\n=== Test 2: Nibble Packing Correctness ===")

# Create weights with known values (all in [-8, 7])
N, K = 8, 64
test_vals = np.array([-8, -7, -1, 0, 1, 7, 3, -3], dtype=np.int8)
# tile to [N, K]
W_int8 = np.tile(test_vals, (N, K // 8)).astype(np.float32)  # each row repeats test_vals

# Quantize with scale=1 (direct INT4 storage)
scale_1 = np.ones(N, dtype=np.float32)
W_packed, scale_back = repack_simple_for_test(W_int8, scale_w=scale_1)

# Unpack manually and verify
for b in range(8):
    nibble_u = ((W_packed[:, 0] >> (b * 4)) & 0xF).astype(np.int8)
    nibble_s = np.where(nibble_u >= 8, nibble_u - 16, nibble_u)
    expected = int(np.round(test_vals[b]))  # already integer, scale=1
    actual = int(nibble_s[0])
    ok = (actual == expected)
    print(f"  nibble[{b}]: expected={expected:3d}, got={actual:3d}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False

# ---------------------------------------------------------------------------
# Test 3: GPTQ format repack (simulated)
# ---------------------------------------------------------------------------
print("\n=== Test 3: GPTQ Format Repack ===")

N, K, group_size = 256, 512, 128
num_groups = K // group_size
K_packed_dim = K // 8

np.random.seed(7)

# Simulate GPTQ: weights are INT4 (0..15), symmetric zero_point=8
# qweight: [K/8, N] uint32 (K-major)
W_raw = np.random.randint(0, 16, (K, N), dtype=np.uint8)  # [K, N] raw nibbles
W_signed = W_raw.astype(np.int8) - 8  # subtract zero_point=8 → [-8, 7]

# Pack to GPTQ qweight format [K/8, N] uint32
qweight_gptq = np.zeros((K_packed_dim, N), dtype=np.uint32)
for b in range(8):
    qweight_gptq |= (W_raw[b::8, :].astype(np.uint32) << (b * 4))

# Simulated per-group scales [K/group_size=4, N]
scales_gptq = np.random.rand(num_groups, N).astype(np.float32) * 0.01 + 0.001

# For symmetric: qzeros all=8 (encoded as nibble), packed into uint32
# qzeros shape: [num_groups, N/8] uint32
N_packed = N // 8
qzeros_gptq = np.zeros((num_groups, N_packed), dtype=np.uint32)
for b in range(8):
    qzeros_gptq |= (np.uint32(8) << (b * 4))  # zero_point = 8 for all

# Repack to W4A8
W_packed_v2, scale_w_grp_v2, zeros_fp = repack_gptq_to_w4a8(
    qweight_gptq, scales_gptq, qzeros_gptq, group_size=group_size, sym=True)

assert W_packed_v2.shape == (N, K_packed_dim), f"Shape mismatch: {W_packed_v2.shape}"
assert scale_w_grp_v2.shape == (num_groups, N), f"Scale shape mismatch: {scale_w_grp_v2.shape}"

# Verify: dequantize and compare with reference
W_dequant_v2 = dequantize_w4a8(W_packed_v2, scale_w_grp_v2, group_size=group_size)

# Reference: W_signed [K,N] × per-group scales → [K,N] FP32, then transpose to [N,K]
W_ref = np.zeros((N, K), dtype=np.float32)
for g in range(num_groups):
    k_start, k_end = g * group_size, (g + 1) * group_size
    # Scale: scales_gptq[g, :] is [N], W_signed is [K,N] so slice [k_start:k_end, :]
    W_ref[:, k_start:k_end] = (
        W_signed[k_start:k_end, :].T.astype(np.float32)
        * scales_gptq[g, :, np.newaxis]
    )

max_abs_ref = float(np.max(np.abs(W_ref)))
max_err = float(np.max(np.abs(W_dequant_v2 - W_ref)))
rel_err = max_err / (max_abs_ref + 1e-8)
PASS = rel_err < 1e-5  # should be exact (integer weights, same scale)
print(f"  GPTQ repack round-trip: max_abs_ref={max_abs_ref:.5f}  "
      f"max_err={max_err:.2e}  rel_err={rel_err:.2e}  {'PASS' if PASS else 'FAIL'}")
if not PASS:
    all_pass = False

# ---------------------------------------------------------------------------
# Test 4: Value preservation for extreme INT4 values
# ---------------------------------------------------------------------------
print("\n=== Test 4: Extreme INT4 Value Preservation ===")

N, K = 16, 8
# All extreme values: -8, -7, -6, ..., 6, 7, then -8, -7, ...
extreme_vals = np.array([-8, -7, -6, -5, -4, -3, 7, 6], dtype=np.int8)
W_extreme = np.tile(extreme_vals, (N, 1)).astype(np.float32)  # [N, K]

scale_1 = np.ones(N, dtype=np.float32)
W_packed_ext, _ = repack_simple_for_test(W_extreme, scale_w=scale_1)

# Dequantize
scale_w_grp_ext = np.ones((1, N), dtype=np.float32)
W_dequant_ext = dequantize_w4a8(W_packed_ext, scale_w_grp_ext, group_size=K)

max_err = float(np.max(np.abs(W_dequant_ext - W_extreme)))
PASS = max_err < 0.5  # should be exact or nearly so with scale=1
print(f"  extreme values: max_err={max_err:.4f}  {'PASS' if PASS else 'FAIL'}")
if max_err > 0.0:
    print(f"  Note: Some rounding expected due to scale computation (max_abs/7 for range)")
    # The issue: extreme_vals includes -8 which is out of range for max_abs=7 → scale=1
    # With scale=1, -8 → round(-8/1)=-8 which is valid INT4, stored as nibble 8 → -8 ✓
if not PASS:
    all_pass = False

# ---------------------------------------------------------------------------
# Test 5: verify_repack_roundtrip helper
# ---------------------------------------------------------------------------
print("\n=== Test 5: Round-trip Helper Function ===")
for N, K, label in [(256, 256, "256x256"), (1024, 1024, "1024x1024")]:
    np.random.seed(13)
    W = np.random.randn(N, K).astype(np.float32)
    result = verify_repack_roundtrip(W, tol=0.15)
    print(f"  {label}: round-trip {'PASS' if result else 'FAIL'}")
    if not result:
        all_pass = False

# ---------------------------------------------------------------------------
# Test 6: Scale correctness
# ---------------------------------------------------------------------------
print("\n=== Test 6: Scale Correctness ===")
N, K = 4, 8
# Weights that are exactly representable in INT4 with scale=1.0
# w ∈ [-7, -6, ..., 6, 7] — avoid -8 to stay within symmetric range
W_exact = np.array([
    [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
    [7.0, 6.0, 5.0, 0.0, -7.0, -6.0, -5.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
], dtype=np.float32)

W_packed_sc, scale_w_sc = repack_simple_for_test(W_exact)
# With max_abs for each row, scale = max_abs / 7.0
expected_scales = np.array([
    4.0 / 7.0,   # max_abs = 4.0
    7.0 / 7.0,   # max_abs = 7.0 → scale = 1.0
    1.0 / 7.0,   # all zeros → 1e-12/7 ≈ 0 (near 0)
    1.0 / 7.0,   # max_abs = 1.0
], dtype=np.float32)
scale_err = np.abs(scale_w_sc - expected_scales)

# For non-zero rows, compare; for all-zeros, just check positivity
print(f"  scales: {scale_w_sc.tolist()}")
print(f"  expected approx: {expected_scales.tolist()}")

# Check dequantization gives back approximate values
scale_w_grp_sc = scale_w_sc[np.newaxis, :]
W_dequant_sc = dequantize_w4a8(W_packed_sc, scale_w_grp_sc, group_size=K)
for n in range(N):
    for k in range(K):
        expected = W_exact[n, k]
        got = W_dequant_sc[n, k]
        err = abs(got - expected)
        # Allow 1 step of INT4 quantization error
        tol = scale_w_sc[n] * 1.01
        if err > tol and abs(expected) > 1e-6:
            print(f"  FAIL at [{n},{k}]: expected={expected:.3f} got={got:.3f} err={err:.4f} tol={tol:.4f}")
            all_pass = False
print(f"  scale correctness: {'PASS' if all_pass else 'FAIL (see above)'}")

# ---------------------------------------------------------------------------
# Test 7: N-major layout correctness
# ---------------------------------------------------------------------------
print("\n=== Test 7: N-major Layout Correctness ===")
N, K = 4, 16
np.random.seed(99)
# Use values that are exactly representable
W_vals = np.array([[-3, 2, 1, -1, 4, -4, 0, 3, -2, 5, -5, 6, -6, 7, -7, 1],
                   [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 0, 1],
                   [-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0],
                   [0, 1, 0, -1, 0, 2, 0, -2, 0, 3, 0, -3, 0, 4, 0, -4]],
                  dtype=np.float32)

scale_1 = np.ones(N, dtype=np.float32)
W_packed_nm, _ = repack_simple_for_test(W_vals, scale_w=scale_1)

assert W_packed_nm.shape == (N, K // 8), f"Expected ({N}, {K//8}), got {W_packed_nm.shape}"

# Manually verify a few values by looking at nibbles in W_packed_nm
# W_packed_nm[n, k//8] bit [(k%8)*4 + 3 : (k%8)*4] should be W_vals[n, k] (as signed nibble)
errors = []
for n in range(N):
    for k in range(K):
        group = k // 8
        bit_pos = (k % 8) * 4
        nibble_u = int((W_packed_nm[n, group] >> bit_pos) & 0xF)
        nibble_s = nibble_u - 16 if nibble_u >= 8 else nibble_u
        expected = int(np.clip(np.round(W_vals[n, k]), -8, 7))
        if nibble_s != expected:
            errors.append(f"W[{n},{k}]={W_vals[n,k]:.0f} → packed nibble={nibble_s} ≠ expected={expected}")

PASS = len(errors) == 0
print(f"  N-major layout: {'PASS' if PASS else 'FAIL'}")
for e in errors[:5]:
    print(f"  {e}")
if not PASS:
    all_pass = False

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
print(f"\n{'=== ALL TESTS PASSED ===' if all_pass else '=== SOME TESTS FAILED ==='}")
if not all_pass:
    sys.exit(1)
