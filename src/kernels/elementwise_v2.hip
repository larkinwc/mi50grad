/**
 * Optimized elementwise kernels using packed half2 operations.
 *
 * All kernels process 2 FP16 elements per thread using __half2 packing.
 * Grid: (ceil(n/512), 1, 1), Block: (256, 1, 1) — each thread handles 2 elements.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Residual add: dst[i] += src[i], packed half2
extern "C" __global__ void residual_add_v2(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        // Load as half2 (2 elements at once)
        __half2 d = *reinterpret_cast<const __half2*>(dst + idx);
        __half2 s = *reinterpret_cast<const __half2*>(src + idx);
        *reinterpret_cast<__half2*>(dst + idx) = __hadd2(d, s);
    } else if (idx < n) {
        // Handle odd tail element
        dst[idx] = __hadd(dst[idx], src[idx]);
    }
}

// SiLU fused: gate[i] = silu(gate[i]) * up[i], packed half2
extern "C" __global__ void silu_fused_v2(
    __half* __restrict__ gate,
    const __half* __restrict__ up,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 g = *reinterpret_cast<const __half2*>(gate + idx);
        __half2 u = *reinterpret_cast<const __half2*>(up + idx);

        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // Process each half separately (no native half2 exp)
        float g0 = __half2float(__low2half(g));
        float g1 = __half2float(__high2half(g));
        float u0 = __half2float(__low2half(u));
        float u1 = __half2float(__high2half(u));

        float sig0 = 1.0f / (1.0f + __expf(-g0));
        float sig1 = 1.0f / (1.0f + __expf(-g1));

        float r0 = g0 * sig0 * u0;
        float r1 = g1 * sig1 * u1;

        *reinterpret_cast<__half2*>(gate + idx) = __halves2half2(
            __float2half(r0), __float2half(r1));
    } else if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float sig = 1.0f / (1.0f + __expf(-g));
        gate[idx] = __float2half(g * sig * u);
    }
}

// RMSNorm with DPP warp reduction (no LDS tree)
// Each WG processes one vector of dim elements.
// Grid: (num_vectors, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void rmsnorm_v2(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    const __half* __restrict__ weight,
    unsigned int dim,
    float eps
) {
    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane = threadIdx.x % 64;
    unsigned int vec_offset = blockIdx.x * dim;

    // Phase 1: compute partial sum of squares
    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += 256) {
        float val = __half2float(src[vec_offset + i]);
        sum_sq += val * val;
    }

    // Phase 2: warp reduction via DPP (wave64)
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);
    // Now lane 0 of each warp has the warp-local sum

    // Phase 3: cross-warp reduction via shared memory (4 warps)
    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    // All threads read the 4 partial sums and combine
    if (threadIdx.x < 4) {
        float total = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        s_warp[threadIdx.x] = total;
    }
    __syncthreads();
    float total_sq = s_warp[0];

    // Phase 4: compute RMS inverse and normalize
    // Gemma-style: (1 + weight) * (x / rms)
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);

    for (unsigned int i = threadIdx.x; i < dim; i += 256) {
        float val = __half2float(src[vec_offset + i]);
        float w = __half2float(weight[i]);
        dst[vec_offset + i] = __float2half(val * rms_inv * w);
    }
}

// Fused skip-connection + RMSNorm:
//   hidden[i] = hidden[i] + residual[i]   (in-place update)
//   dst[i] = rmsnorm(hidden[i]) * weight[i]
//
// Eliminates separate residual_add + rmsnorm launches.
// For TP: residual is the allreduced partial sum, hidden accumulates.
// Grid: (1, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void skip_rmsnorm_v2(
    __half* __restrict__ dst,        // normalized output
    __half* __restrict__ hidden,     // hidden state (read + write)
    const __half* __restrict__ residual, // residual to add
    const __half* __restrict__ weight,
    unsigned int dim,
    float eps
) {
    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane = threadIdx.x % 64;

    // Phase 1: add residual to hidden, compute sum of squares
    float sum_sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += 256) {
        float h = __half2float(hidden[i]);
        float r = __half2float(residual[i]);
        float val = h + r;
        hidden[i] = __float2half(val);  // write back updated hidden
        sum_sq += val * val;
    }

    // Phase 2: warp reduction
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);

    // Phase 3: cross-warp reduction
    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 4) {
        float total = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        s_warp[threadIdx.x] = total;
    }
    __syncthreads();
    float total_sq = s_warp[0];

    // Phase 4: normalize
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);

    for (unsigned int i = threadIdx.x; i < dim; i += 256) {
        float val = __half2float(hidden[i]);
        float w = __half2float(weight[i]);
        dst[i] = __float2half(val * rms_inv * w);
    }
}
