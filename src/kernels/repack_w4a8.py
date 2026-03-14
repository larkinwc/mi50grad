"""
W4A8 weight repacking utility.

Converts GPTQ INT4 weight format to W4A8 packed layout for gemv_w4a8 kernels.

GPTQ format (HuggingFace):
  qweight: [K/8, N] uint32  — K-major packed: bits[4i+3:4i] = weight for k=i, n=col
  scales:  [K/group_size, N] FP16 — per-group per-channel scales
  qzeros:  [K/group_size, N/8] uint32 — per-group zero points (packed INT4)
  (qzeros values: sym=True uses qzeros=8, meaning zero point = 8, stored weights center at 8)

W4A8 format (our gemv_w4a8 kernels):
  W_packed: [N, K/8] uint32  — N-major row-major packed
    bits[4b+3:4b] = (w_raw[n,k+b] - zero_point) ∈ [-8, 7]  (signed INT4)
  scale_w:  [N] FP32  — per-channel scale (for whole-column scale case)
  OR
  scale_w_grp: [K/group_size, N] FP16  — per-group scale (for grouped case)

Two modes:
1. Per-channel (simplified): fold per-group scales into a single per-channel scale.
   Only correct if all groups have the same scale (unusual). Provided for testing.
2. Per-group (correct): preserve per-group scales in [K/group_size, N] FP16 layout.
   This matches the gemv_w4a8_grouped kernel.

The zero-point subtraction is always applied during repacking:
  w_repacked = w_raw - zero_point  (signed INT4, clamped to [-8, 7])
  Where: GPTQ symmetric: zero_point = 8 (so w_raw ∈ {0..15} → w_repacked ∈ {-8..7})
         GPTQ asymmetric: zero_point from qzeros tensor (varies per group/channel)
"""

import numpy as np
from typing import Tuple, Optional


def unpack_gptq_qweight(qweight: np.ndarray) -> np.ndarray:
    """Unpack GPTQ qweight [K/8, N] uint32 → [K, N] uint8 (raw INT4 values 0..15).

    GPTQ packs 8 INT4 values per uint32 in K-major order:
      qweight[k//8, n] bits[4*(k%8)+3 : 4*(k%8)] = weight value for (k, n)
    """
    K_packed, N = qweight.shape
    K = K_packed * 8
    W_unpacked = np.zeros((K, N), dtype=np.uint8)
    for b in range(8):
        W_unpacked[b::8, :] = (qweight >> (b * 4)) & 0xF
    return W_unpacked  # [K, N] uint8, values 0..15


def unpack_gptq_qzeros(qzeros: np.ndarray) -> np.ndarray:
    """Unpack GPTQ qzeros [num_groups, N/8] uint32 → [num_groups, N] uint8.

    Same packing as qweight but for zero points.
    GPTQ symmetric: all qzeros = 8 (stored as packed nibbles of value 8).
    """
    num_groups, N_packed = qzeros.shape
    N = N_packed * 8
    zeros_unpacked = np.zeros((num_groups, N), dtype=np.uint8)
    for b in range(8):
        zeros_unpacked[:, b::8] = (qzeros >> (b * 4)) & 0xF
    return zeros_unpacked  # [num_groups, N] uint8, values 0..15


def repack_gptq_to_w4a8(
    qweight: np.ndarray,           # [K/8, N] uint32 GPTQ packed weights
    scales: np.ndarray,            # [K/group_size, N] FP16 or FP32 per-group scales
    qzeros: Optional[np.ndarray] = None,  # [K/group_size, N/8] uint32 GPTQ zeros
    group_size: int = 128,
    sym: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repack GPTQ INT4 weights to W4A8 format.

    Returns:
        W_packed:    [N, K/8] uint32  — N-major row-major, zero-subtracted signed INT4
        scale_w_grp: [K/group_size, N] FP32  — per-group scales (for gemv_w4a8_grouped)
        zeros_fp:    [K/group_size, N] FP32  — zero points as FP32 (for verification)
    """
    K_packed, N = qweight.shape
    K = K_packed * 8
    num_groups = K // group_size

    # Unpack GPTQ weights: [K, N] uint8 (0..15)
    W_raw = unpack_gptq_qweight(qweight)  # [K, N]

    # Get zero points
    if qzeros is not None:
        zeros_unpacked = unpack_gptq_qzeros(qzeros)  # [num_groups, N] uint8
    else:
        # Symmetric: zero_point = 8 for all groups
        zeros_unpacked = np.full((num_groups, N), 8, dtype=np.uint8)

    # Verify: zero_point should be 8 for symmetric GPTQ (value 8 → encoded as nibble 8)
    zeros_fp = zeros_unpacked.astype(np.float32)  # [num_groups, N]
    scales_fp = np.asarray(scales, dtype=np.float32)  # [num_groups, N]

    # Subtract zero point and sign-extend: w_signed = w_raw - zero ∈ [-8, 7]
    # Expand zeros from [num_groups, N] to [K, N] by repeating per group
    zeros_expanded = np.repeat(zeros_unpacked, group_size, axis=0)  # [K, N]
    W_signed = W_raw.astype(np.int8) - zeros_expanded.astype(np.int8)
    # Clamp to INT4 range [-8, 7]
    W_signed = np.clip(W_signed, -8, 7).astype(np.int8)  # [K, N]

    # Transpose to N-major (row-major for our kernel): [N, K]
    W_Nmajor = W_signed.T  # [N, K]

    # Pack [N, K] int8 → [N, K/8] uint32 (nibble-packed)
    K_groups = K // 8
    W_packed = np.zeros((N, K_groups), dtype=np.uint32)
    for b in range(8):
        # Nibble b at bits[4b+3:4b]: w at column k = group*8 + b
        nibbles = W_Nmajor[:, b::8].astype(np.int32) & 0xF  # [N, K/8]
        W_packed |= (nibbles.astype(np.uint32) << (b * 4))

    # Per-group scales (for gemv_w4a8_grouped): [K/group_size, N] FP32
    scale_w_grp = scales_fp  # already [num_groups, N]

    return W_packed, scale_w_grp, zeros_fp


def repack_gptq_to_w4a8_perchannel(
    qweight: np.ndarray,           # [K/8, N] uint32
    scales: np.ndarray,            # [K/group_size, N] FP16/FP32
    qzeros: Optional[np.ndarray] = None,
    group_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Repack GPTQ INT4 weights to W4A8 with per-channel scales.

    For testing purposes: averages per-group scales into a single per-channel scale.
    Not recommended for production (loses per-group quantization accuracy).

    Returns:
        W_packed:  [N, K/8] uint32
        scale_w:   [N] FP32 per-channel (max of group scales, for rounding purposes)
    """
    W_packed, scale_w_grp, zeros_fp = repack_gptq_to_w4a8(
        qweight, scales, qzeros, group_size)
    # Use mean of per-group scales as the per-channel scale
    scale_w = scale_w_grp.mean(axis=0).astype(np.float32)  # [N]
    return W_packed, scale_w


def dequantize_w4a8(
    W_packed: np.ndarray,       # [N, K/8] uint32 W4A8 packed
    scale_w_grp: np.ndarray,    # [K/group_size, N] FP32
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize W4A8 packed weights back to FP32 for reference computation.

    Returns: W_fp32 [N, K] FP32
    """
    N, K_groups = W_packed.shape
    K = K_groups * 8
    num_groups = K // group_size

    # Unpack nibbles: [N, K] int8
    W_int8 = np.zeros((N, K), dtype=np.int8)
    for b in range(8):
        # Nibble b at bits[4b+3:4b]
        nibble_u = ((W_packed >> (b * 4)) & 0xF).astype(np.int8)  # [N, K/8]
        # Sign-extend: if >= 8, subtract 16
        nibble_s = np.where(nibble_u >= 8, nibble_u - 16, nibble_u)
        W_int8[:, b::8] = nibble_s

    # Apply per-group scales: scale_w_grp [num_groups, N]
    # W_fp32[n, k] = W_int8[n, k] * scale_w_grp[k // group_size, n]
    W_fp32 = np.zeros((N, K), dtype=np.float32)
    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        # scale_w_grp[g, :] is [N] scales for group g
        W_fp32[:, k_start:k_end] = (
            W_int8[:, k_start:k_end].astype(np.float32) * scale_w_grp[g, :, np.newaxis]
        )

    return W_fp32


def repack_simple_for_test(
    W_fp32: np.ndarray,
    scale_w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create W4A8 packed weights from FP32 for testing (quantize then pack).

    Quantizes W_fp32 [N, K] to INT4 (range [-8, 7]) using per-channel scale,
    then packs into W4A8 format.

    Returns:
        W_packed:  [N, K/8] uint32
        scale_w:   [N] FP32 per-channel scales
    """
    N, K = W_fp32.shape
    assert K % 8 == 0, f"K={K} must be divisible by 8"

    # Per-channel quantization: scale = max_abs / 7.0
    if scale_w is None:
        max_abs = np.abs(W_fp32).max(axis=1) + 1e-12  # [N]
        scale_w = (max_abs / 7.0).astype(np.float32)

    # Quantize: round and clamp to [-8, 7]
    W_int8 = np.clip(np.round(W_fp32 / scale_w[:, np.newaxis]), -8, 7).astype(np.int8)

    # Pack [N, K] int8 → [N, K/8] uint32
    K_groups = K // 8
    W_packed = np.zeros((N, K_groups), dtype=np.uint32)
    for b in range(8):
        nibbles = (W_int8[:, b::8].astype(np.int32)) & 0xF  # [N, K/8]
        W_packed |= (nibbles.astype(np.uint32) << (b * 4))

    return W_packed, scale_w.astype(np.float32)


def verify_repack_roundtrip(W_fp32: np.ndarray, tol: float = 1e-2) -> bool:
    """Verify that repack → dequantize preserves values within tolerance.

    Returns True if max relative error < tol.
    """
    N, K = W_fp32.shape
    W_packed, scale_w = repack_simple_for_test(W_fp32)

    # Dequantize: scale_w_grp would be [1, N] for per-channel (single group)
    scale_w_grp = scale_w[np.newaxis, :]  # [1, N]
    W_dequant = dequantize_w4a8(W_packed, scale_w_grp, group_size=K)

    max_abs = np.max(np.abs(W_fp32))
    if max_abs < 1e-8:
        return True
    max_err = np.max(np.abs(W_dequant - W_fp32)) / (max_abs + 1e-8)
    return float(max_err) < tol
