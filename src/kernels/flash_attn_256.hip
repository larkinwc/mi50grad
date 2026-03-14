/**
 * FlashAttention for head_dim=256 on gfx906.
 *
 * out[q] = softmax(Q[q] * K^T / scale) * V with online softmax.
 * Supports causal masking for autoregressive prefill.
 *
 * Each wavefront (64 threads) handles one Q row.
 * Each thread covers 4 head dims (256 / 64 = 4).
 * 4 wavefronts per workgroup = 4 Q rows per WG.
 *
 * Works for both prefill and decode:
 *   Prefill: num_q_rows = kv_seq_len, causal = 1
 *   Decode:  num_q_rows = 1, kv_seq_len = cache_len, causal = 0
 *
 * Grid: (num_heads, ceil(num_q_rows/4), 1), Block: (256, 1, 1)
 *
 * Q/Out layout: [num_q_rows, num_heads, head_dim] FP16
 * K/V layout:   [kv_seq_len, num_kv_heads, head_dim] FP16
 * Supports GQA: num_kv_heads <= num_heads.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HEAD_DIM 256
#define DIMS_PER_THREAD 4  // 256/64 = 4

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down(val, 32);
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val;
}

extern "C" __global__ void flash_attn_256_fp16(
    const __half* __restrict__ Q,    // [num_q_rows, num_heads, 256]
    const __half* __restrict__ K,    // [kv_seq_len, num_kv_heads, 256]
    const __half* __restrict__ V,    // [kv_seq_len, num_kv_heads, 256]
    __half* __restrict__ Out,        // [num_q_rows, num_heads, 256]
    unsigned int kv_seq_len,
    unsigned int num_q_rows,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int causal               // 1 = causal masking
) {
    unsigned int head = blockIdx.x;
    unsigned int q_block = blockIdx.y;
    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane = threadIdx.x % 64;

    unsigned int q_row = q_block * 4 + wf_id;
    if (q_row >= num_q_rows) return;

    // GQA: map query head to KV head
    unsigned int kv_head = head * num_kv_heads / num_heads;

    // Dim indices this thread handles: lane*4, lane*4+1, lane*4+2, lane*4+3
    unsigned int dim_base = lane * DIMS_PER_THREAD;

    float scale = rsqrtf((float)HEAD_DIM);

    // Load Q values (4 dims per thread)
    size_t q_offset = ((size_t)q_row * num_heads + head) * HEAD_DIM + dim_base;
    float q0 = __half2float(Q[q_offset + 0]) * scale;
    float q1 = __half2float(Q[q_offset + 1]) * scale;
    float q2 = __half2float(Q[q_offset + 2]) * scale;
    float q3 = __half2float(Q[q_offset + 3]) * scale;

    // Online softmax state
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // KV loop bound
    unsigned int kv_limit = causal ? min(q_row + 1, kv_seq_len) : kv_seq_len;

    // KV row stride
    size_t kv_stride = (size_t)num_kv_heads * HEAD_DIM;

    for (unsigned int kv_pos = 0; kv_pos < kv_limit; kv_pos++) {
        // Load K[kv_pos]
        size_t k_offset = (size_t)kv_pos * kv_stride + kv_head * HEAD_DIM + dim_base;
        float k0 = __half2float(K[k_offset + 0]);
        float k1 = __half2float(K[k_offset + 1]);
        float k2 = __half2float(K[k_offset + 2]);
        float k3 = __half2float(K[k_offset + 3]);

        // Partial dot product (4 elements per thread)
        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;

        // Warp reduction for full dot product
        float score = warp_reduce_sum(dot);
        // Broadcast score to all lanes
        score = __shfl(score, 0);

        // Online softmax update
        float new_max = fmaxf(running_max, score);
        float correction = expf(running_max - new_max);
        float p = expf(score - new_max);

        running_sum = running_sum * correction + p;
        acc0 = acc0 * correction;
        acc1 = acc1 * correction;
        acc2 = acc2 * correction;
        acc3 = acc3 * correction;

        // Load V[kv_pos] and accumulate
        float v0 = __half2float(V[k_offset + 0]);
        float v1 = __half2float(V[k_offset + 1]);
        float v2 = __half2float(V[k_offset + 2]);
        float v3 = __half2float(V[k_offset + 3]);

        acc0 += p * v0;
        acc1 += p * v1;
        acc2 += p * v2;
        acc3 += p * v3;

        running_max = new_max;
    }

    // Normalize
    float inv_sum = 1.0f / running_sum;
    acc0 *= inv_sum;
    acc1 *= inv_sum;
    acc2 *= inv_sum;
    acc3 *= inv_sum;

    // Store output
    size_t out_offset = ((size_t)q_row * num_heads + head) * HEAD_DIM + dim_base;
    Out[out_offset + 0] = __float2half(acc0);
    Out[out_offset + 1] = __float2half(acc1);
    Out[out_offset + 2] = __float2half(acc2);
    Out[out_offset + 3] = __float2half(acc3);
}
