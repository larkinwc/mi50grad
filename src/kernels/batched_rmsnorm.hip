/**
 * Batched RMSNorm: apply RMSNorm independently to N vectors of dim D.
 * Used for per-head Q/K normalization in attention layers.
 *
 * Grid: (num_vectors, 1, 1), Block: (256, 1, 1)
 * Each workgroup handles one vector of length `dim`.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down(val, 32);
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val;
}

/**
 * Batched RMSNorm for per-head normalization.
 * x: [num_vectors, dim] FP16 — modified in-place
 * weight: [dim] FP16 — shared norm weight across all vectors
 */
extern "C" __global__ void batched_rmsnorm_fp16(
    __half* __restrict__ x,
    const __half* __restrict__ weight,
    unsigned int dim,
    unsigned int eps_bits  // float epsilon, passed as uint32 bits
) {
    unsigned int vec_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    float eps = __uint_as_float(eps_bits);

    __half* vec = x + (size_t)vec_idx * dim;

    // Compute sum of squares
    float sq_sum = 0.0f;
    for (unsigned int i = tid; i < dim; i += 256) {
        float v = __half2float(vec[i]);
        sq_sum += v * v;
    }

    // Warp reduce + cross-warp combine
    float wsum = warp_reduce_sum(sq_sum);
    __shared__ float s_warp[4];
    int warp_id = tid / 64;
    int lane = tid % 64;
    if (lane == 0) s_warp[warp_id] = wsum;
    __syncthreads();

    float total_sq = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    float rms_inv = rsqrtf(total_sq / dim + eps);

    // Normalize
    for (unsigned int i = tid; i < dim; i += 256) {
        float v = __half2float(vec[i]) * rms_inv * __half2float(weight[i]);
        vec[i] = __float2half(v);
    }
}
