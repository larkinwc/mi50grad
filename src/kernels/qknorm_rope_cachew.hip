/**
 * Fused QK-norm + RoPE kernel with KV cache write for gfx906 (MI50/MI60).
 *
 * This is a variant of qknorm_rope.hip that additionally writes the post-RoPE
 * output directly to a KV cache position, eliminating the separate
 * hipMemcpyAsync D2D copy from the working buffer to the KV cache.
 *
 * When cache_dst is non-NULL:
 *   1. Performs per-head RMSNorm + partial RoPE (same as qknorm_rope_fused)
 *   2. Also writes the result to cache_dst (the KV cache position for this token)
 *
 * This fuses the KV cache append into the QK-norm+RoPE kernel, eliminating
 * 16 hipMemcpyAsync D2D calls per token (1 per full-attention layer × 16 layers)
 * for K. Combined with direct V writes, eliminates all 32 D2D copies.
 *
 * Grid: (num_heads, 1, 1), Block: (256, 1, 1)
 * Each block handles one head of head_dim elements.
 *
 * For Qwen 3.5 27B:
 *   head_dim = 256, rotary_dim = 64, half_rotary = 32
 *   num_kv_heads = 2 (per GPU with TP=4, local heads)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __forceinline__ float warp_reduce_sum_cw(float val, int lane) {
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

/**
 * Fused per-head RMSNorm + partial RoPE + optional KV cache write.
 *
 * x:           [num_heads, head_dim] FP16 — modified in-place (working buffer)
 * norm_weight: [head_dim] FP16 — per-element scale (shared across heads)
 * cos_row:     [half_rotary] FP16 — cos values for the current position
 * sin_row:     [half_rotary] FP16 — sin values for the current position
 * cache_dst:   [num_heads, head_dim] FP16 — KV cache destination (NULL = skip cache write)
 *              Points to the KV cache position for the current token:
 *              layer_k_ptr + current_pos * num_kv_heads * head_dim * sizeof(fp16)
 *              or NULL to behave identically to qknorm_rope_fused.
 * head_dim:    dimension per head (e.g. 256)
 * half_rotary: half of rotary_dim (e.g. 32 for rotary_dim=64)
 * eps:         small value for numerical stability
 */
extern "C" __global__ void qknorm_rope_cachew_fused(
    __half* __restrict__ x,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ cos_row,   /* [half_rotary] for current position */
    const __half* __restrict__ sin_row,   /* [half_rotary] for current position */
    __half* __restrict__ cache_dst,       /* [num_heads, head_dim] KV cache pos, or NULL */
    unsigned int head_dim,
    unsigned int half_rotary,
    float eps
) {
    unsigned int head_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int wf_id = tid / 64;
    unsigned int lane = tid % 64;

    __half* head = x + (size_t)head_idx * head_dim;
    __half* cache_head = (cache_dst != NULL)
        ? (cache_dst + (size_t)head_idx * head_dim)
        : NULL;

    /* -----------------------------------------------------------------------
     * Phase 1: Compute sum of squares for RMSNorm
     * ----------------------------------------------------------------------- */
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += 256) {
        float v = __half2float(head[i]);
        sum_sq += v * v;
    }

    /* Warp-level reduce (wave64) */
    sum_sq = warp_reduce_sum_cw(sum_sq, lane);

    /* Cross-warp reduce via LDS (4 warps) */
    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    float total_sq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    float rms_inv = rsqrtf(total_sq / (float)head_dim + eps);

    /* -----------------------------------------------------------------------
     * Phase 2: Normalize + RoPE, store normalized values in shared memory
     * ----------------------------------------------------------------------- */
    __shared__ float s_norm[256];

    for (unsigned int i = tid; i < head_dim; i += 256) {
        float v = __half2float(head[i]);
        float w = __half2float(norm_weight[i]);
        s_norm[i] = v * rms_inv * w;
    }
    __syncthreads();

    /* -----------------------------------------------------------------------
     * Phase 3: Apply RoPE rotation to [0, 2*half_rotary), write all back
     * to working buffer AND (if cache_dst != NULL) to the KV cache position.
     * ----------------------------------------------------------------------- */

    /* RoPE region: threads [0, half_rotary) handle pair (2i, 2i+1) */
    if (tid < half_rotary) {
        unsigned int i = tid;
        float x0 = s_norm[2 * i];
        float x1 = s_norm[2 * i + 1];
        float c = __half2float(cos_row[i]);
        float s = __half2float(sin_row[i]);

        float new_x0 = x0 * c - x1 * s;
        float new_x1 = x0 * s + x1 * c;

        /* Write to working buffer */
        head[2 * i]     = __float2half(new_x0);
        head[2 * i + 1] = __float2half(new_x1);

        /* Write to KV cache position (if provided) */
        if (cache_head != NULL) {
            cache_head[2 * i]     = __float2half(new_x0);
            cache_head[2 * i + 1] = __float2half(new_x1);
        }
    }

    /* Non-RoPE region: write normalized-only values */
    unsigned int rope_end = 2 * half_rotary;
    for (unsigned int i = tid; i < head_dim; i += 256) {
        if (i >= rope_end) {
            __half val = __float2half(s_norm[i]);
            head[i] = val;
            if (cache_head != NULL) {
                cache_head[i] = val;
            }
        }
    }
}
