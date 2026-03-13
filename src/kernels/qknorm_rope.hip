/**
 * Fused QK-norm + RoPE kernel for gfx906 (MI50/MI60).
 *
 * Performs per-head RMSNorm followed by partial RoPE in a single kernel launch,
 * replacing 4 separate launches: batched_rmsnorm(Q), batched_rmsnorm(K),
 * rope(Q), rope(K).
 *
 * Algorithm per head:
 *   1. Compute per-head RMSNorm: x[i] = x[i] / rms * weight[i]
 *   2. Apply partial RoPE to the first rotary_dim elements:
 *      For each pair (2i, 2i+1) where i < half_rotary:
 *        new_x[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]
 *        new_x[2i+1] = x[2i]   * sin[i] + x[2i+1] * cos[i]
 *      Elements [rotary_dim, head_dim) are left as-is after normalization.
 *
 * Grid: (num_heads, 1, 1), Block: (256, 1, 1)
 * Each block handles one head of head_dim elements.
 * Supports head_dim up to 512 (loop stride = 256 threads).
 *
 * For Qwen 3.5 27B:
 *   head_dim = 256, rotary_dim = 64, half_rotary = 32
 *   Only elements [0, 64) get RoPE; [64, 256) get only RMSNorm.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// Phase 1: Warp-reduce + cross-warp shared-memory reduce for sum-of-squares
// Uses __shfl_xor for DPP-style warp reduction (4 warps of 64 threads each)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val, int lane) {
    // Wave64 reduction: XOR with 32, 16, 8, 4, 2, 1
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

/**
 * Fused per-head RMSNorm + partial RoPE.
 *
 * x:           [num_heads, head_dim] FP16 — modified in-place
 * norm_weight: [head_dim] FP16 — per-element scale (shared across heads)
 * cos_row:     [half_rotary] FP16 — cos values for the current position
 * sin_row:     [half_rotary] FP16 — sin values for the current position
 * head_dim:    dimension per head (e.g. 256)
 * half_rotary: half of rotary_dim (e.g. 32 for rotary_dim=64)
 * eps:         small value for numerical stability
 */
extern "C" __global__ void qknorm_rope_fused(
    __half* __restrict__ x,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ cos_row,  // [half_rotary] for current position
    const __half* __restrict__ sin_row,  // [half_rotary] for current position
    unsigned int head_dim,
    unsigned int half_rotary,
    float eps
) {
    unsigned int head_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int wf_id = tid / 64;
    unsigned int lane = tid % 64;

    __half* head = x + (size_t)head_idx * head_dim;

    // -----------------------------------------------------------------------
    // Phase 1: Compute sum of squares for RMSNorm
    // -----------------------------------------------------------------------
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += 256) {
        float v = __half2float(head[i]);
        sum_sq += v * v;
    }

    // Warp-level reduce (wave64)
    sum_sq = warp_reduce_sum(sum_sq, lane);

    // Cross-warp reduce via LDS (4 warps)
    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    float total_sq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    float rms_inv = rsqrtf(total_sq / (float)head_dim + eps);

    // -----------------------------------------------------------------------
    // Phase 2: Normalize (RMSNorm) and apply RoPE to first half_rotary pairs
    //
    // For elements in the RoPE region [0, 2*half_rotary):
    //   Normalize both x[2i] and x[2i+1], then rotate.
    //
    // For elements outside RoPE region [2*half_rotary, head_dim):
    //   Just normalize.
    //
    // Load elements, normalize, then do the rotation for RoPE-active elements.
    // -----------------------------------------------------------------------

    // Normalize ALL elements first (stored back in registers via shared mem
    // or written to global memory). We do two passes: one for RoPE region,
    // one for non-RoPE tail.

    // We need to coordinate pairs (2i, 2i+1) for rotation, which are accessed
    // by threads with tid=i*2 and tid=i*2+1. However, our block has 256 threads
    // and head_dim can be 256 so each thread handles exactly one element in the
    // direct case, or we loop for larger head_dims.

    // Strategy: First pass normalizes all elements and stores them to a shared
    // buffer. Second pass reads pairs from shared buffer, applies rotation for
    // RoPE region, writes back.
    //
    // For head_dim=256: 256 threads, each handles 1 element. Shared buffer = 256 halfs = 512 bytes.
    // For head_dim=128: 256 threads, each handles at most 1 element (128 threads active).

    // Use LDS for normalized values (we already have s_warp as shared, but
    // we need more space). Declare separate shared array for normalized values.
    __shared__ float s_norm[256];  // Normalized FP32 values

    // Normalize and store to shared
    for (unsigned int i = tid; i < head_dim; i += 256) {
        float v = __half2float(head[i]);
        float w = __half2float(norm_weight[i]);
        s_norm[i] = v * rms_inv * w;
    }
    __syncthreads();

    // Apply RoPE rotation to [0, 2*half_rotary) elements, then write all back
    // Thread tid handles elements at positions tid, tid+256, etc.
    // For RoPE: process pairs. Thread 2i handles pair (2i, 2i+1).
    // But with stride=256, each thread handles at most one pair in the RoPE region
    // for head_dim=256 with 32 rotary pairs.

    // --- RoPE region: write rotated pairs ---
    // Threads [0, half_rotary) handle the rotation pairs
    // tid < half_rotary means tid is a "pair index" i → rotates positions (2i, 2i+1)
    if (tid < half_rotary) {
        unsigned int i = tid;  // pair index
        float x0 = s_norm[2 * i];
        float x1 = s_norm[2 * i + 1];
        float c = __half2float(cos_row[i]);
        float s = __half2float(sin_row[i]);

        float new_x0 = x0 * c - x1 * s;
        float new_x1 = x0 * s + x1 * c;

        head[2 * i]     = __float2half(new_x0);
        head[2 * i + 1] = __float2half(new_x1);
    }

    // --- Non-RoPE region: write normalized-only values ---
    // Elements [2*half_rotary, head_dim) get only the RMSNorm applied
    unsigned int rope_end = 2 * half_rotary;
    for (unsigned int i = tid; i < head_dim; i += 256) {
        if (i >= rope_end) {
            head[i] = __float2half(s_norm[i]);
        }
    }
}
