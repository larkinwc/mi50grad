/**
 * HIP C++ RoPE (Rotary Position Embedding) kernel for gfx906 (MI50/MI60).
 *
 * Replaces the assembly-only rope.s with a vectorized HIP implementation
 * supporting head_dim=128 and head_dim=256.
 *
 * Algorithm: For each token, head, and dimension pair (2i, 2i+1):
 *   theta = precomputed in cos/sin tables at [token_pos, i]
 *   new_x[2i]   = x[2i] * cos(theta) - x[2i+1] * sin(theta)
 *   new_x[2i+1] = x[2i] * sin(theta) + x[2i+1] * cos(theta)
 *
 * Input interface: same as rope.s (precomputed cos/sin tables).
 *   x:        [num_tokens, num_heads, head_dim] FP16 — modified in-place
 *   cos_tab:  [num_tokens, head_dim/2] FP16 — precomputed cos values
 *   sin_tab:  [num_tokens, head_dim/2] FP16 — precomputed sin values
 *   head_dim: dimension per head (128 or 256)
 *   num_heads: number of heads
 *
 * Grid: (num_tokens, num_heads, 1), Block: (head_dim/2, 1, 1)
 *   For head_dim=128: block has 64 threads
 *   For head_dim=256: block has 128 threads
 *
 * Vectorization: Uses half2 packed loads to load 2 FP16 values at once.
 * Rotation computation is done in FP32 for numerical precision.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/**
 * rope_fp16_v2: Vectorized HIP RoPE kernel.
 *
 * Each thread handles one rotation pair (x[2i], x[2i+1]):
 *   - Loads packed half2 {x[2i+1], x[2i]} as a single 32-bit load
 *   - Loads cos and sin values (FP16, converted to FP32)
 *   - Applies rotation in FP32, stores packed half2 back
 *
 * This matches rope.s semantics but uses vectorized half2 loads.
 */
extern "C" __global__ void rope_fp16_v2(
    __half* __restrict__ x,            // [num_tokens, num_heads, head_dim] in-place
    const __half* __restrict__ cos_tab, // [num_tokens, head_dim/2]
    const __half* __restrict__ sin_tab, // [num_tokens, head_dim/2]
    unsigned int head_dim,             // e.g. 128 or 256
    unsigned int num_heads             // number of heads
) {
    // Grid: (num_tokens, num_heads, 1)
    unsigned int token_idx = blockIdx.x;
    unsigned int head_idx  = blockIdx.y;
    // Thread: one pair per thread; blockDim.x = head_dim / 2
    unsigned int pair_idx  = threadIdx.x;  // i in [0, head_dim/2)

    unsigned int half_dim = head_dim >> 1;  // head_dim / 2

    if (pair_idx >= half_dim) return;

    // -----------------------------------------------------------------------
    // Compute x pointer: token_idx * num_heads * head_dim + head_idx * head_dim
    // -----------------------------------------------------------------------
    size_t head_offset = ((size_t)token_idx * num_heads + head_idx) * head_dim;
    __half* head_ptr = x + head_offset;

    // Load pair (x[2i], x[2i+1]) as packed half2 (vectorized 32-bit load)
    // half2 stores as {hi: x[2i+1], lo: x[2i]}
    __half2 xpair = *reinterpret_cast<const __half2*>(head_ptr + pair_idx * 2);
    float x0 = __half2float(__low2half(xpair));   // x[2i]
    float x1 = __half2float(__high2half(xpair));  // x[2i+1]

    // -----------------------------------------------------------------------
    // Load cos/sin: cos_tab[token_idx, pair_idx] = cos_tab[token_idx*half_dim + pair_idx]
    // -----------------------------------------------------------------------
    size_t cs_offset = (size_t)token_idx * half_dim + pair_idx;
    float c = __half2float(cos_tab[cs_offset]);
    float s = __half2float(sin_tab[cs_offset]);

    // -----------------------------------------------------------------------
    // Apply rotation in FP32
    // -----------------------------------------------------------------------
    float new_x0 = x0 * c - x1 * s;
    float new_x1 = x0 * s + x1 * c;

    // Pack back to half2 and store (vectorized 32-bit store)
    __half2 result = __halves2half2(__float2half(new_x0), __float2half(new_x1));
    *reinterpret_cast<__half2*>(head_ptr + pair_idx * 2) = result;
}
