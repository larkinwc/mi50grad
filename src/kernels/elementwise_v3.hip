/**
 * Optimized elementwise kernels with wider vectorization.
 * elementwise_v3.hip — upgrade from elementwise_v2.hip
 *
 * Key improvements:
 * 1. float4 loads: Each thread loads 128 bits = 8 FP16 elements per instruction
 *    (global_load_dwordx4 vs dwordx2 in v2). 4x more elements per load.
 * 2. silu_fused_v3: 8-wide float4 load with exact sigmoid (1/(1+exp(-x))).
 * 3. Grid: (ceil(n/2048), 1, 1), Block: (256, 1, 1) per thread = 8 FP16 elements.
 *
 * Vectorized loop design for RMSNorm:
 *   Each thread t processes 8-element groups at offsets {t*8, t*8+2048, t*8+4096, ...}
 *   Condition: offset + 7 < dim
 *   Scalar tail for elements [(dim/8)*8, dim-1] (only if dim%8 != 0).
 *   This partitioning ensures NO double-counting.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ============================================================
// residual_add_v3: dst[i] += src[i]
// Uses float4 (128-bit) loads: 8 FP16 per thread per iteration.
// Grid: (ceil(n/2048), 1, 1), Block: (256, 1, 1)
// ============================================================
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void residual_add_v3(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    unsigned int n
) {
    unsigned int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (base + 7 < n) {
        // Full 8-element vectorized path (128-bit load = global_load_dwordx4)
        float4 d4 = *reinterpret_cast<const float4*>(dst + base);
        float4 s4 = *reinterpret_cast<const float4*>(src + base);

        __half2* d_h2       = reinterpret_cast<__half2*>(&d4);
        const __half2* s_h2 = reinterpret_cast<const __half2*>(&s4);

        d_h2[0] = __hadd2(d_h2[0], s_h2[0]);
        d_h2[1] = __hadd2(d_h2[1], s_h2[1]);
        d_h2[2] = __hadd2(d_h2[2], s_h2[2]);
        d_h2[3] = __hadd2(d_h2[3], s_h2[3]);

        *reinterpret_cast<float4*>(dst + base) = d4;
    } else {
        // Scalar tail: handle remaining < 8 elements
        for (unsigned int i = base; i < n; i++) {
            dst[i] = __hadd(dst[i], src[i]);
        }
    }
}

// ============================================================
// silu_fused_v3: gate[i] = silu(gate[i]) * up[i]
// Vectorized 8-wide float4 loads.
// Uses exact sigmoid (1/(1+exp(-x))) for correctness.
// float4 loads = 8 FP16 per thread per iteration.
// Grid: (ceil(n/2048), 1, 1), Block: (256, 1, 1)
// ============================================================
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void silu_fused_v3(
    __half* __restrict__ gate,
    const __half* __restrict__ up,
    unsigned int n
) {
    unsigned int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (base + 7 < n) {
        // Full 8-element vectorized path
        float4 g4 = *reinterpret_cast<const float4*>(gate + base);
        float4 u4 = *reinterpret_cast<const float4*>(up   + base);

        const __half2* g_h2 = reinterpret_cast<const __half2*>(&g4);
        const __half2* u_h2 = reinterpret_cast<const __half2*>(&u4);

        float4 out4;
        __half2* out_h2 = reinterpret_cast<__half2*>(&out4);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float g0 = __half2float(__low2half (g_h2[k]));
            float g1 = __half2float(__high2half(g_h2[k]));
            float u0 = __half2float(__low2half (u_h2[k]));
            float u1 = __half2float(__high2half(u_h2[k]));

            // SiLU(x) = x * sigmoid(x); exact sigmoid for < 5e-3 correctness
            float sig0 = 1.0f / (1.0f + __expf(-g0));
            float sig1 = 1.0f / (1.0f + __expf(-g1));

            out_h2[k] = __halves2half2(
                __float2half(g0 * sig0 * u0),
                __float2half(g1 * sig1 * u1)
            );
        }
        *reinterpret_cast<float4*>(gate + base) = out4;
    } else {
        // Scalar tail
        for (unsigned int i = base; i < n; i++) {
            float g = __half2float(gate[i]);
            float u = __half2float(up[i]);
            float sig = 1.0f / (1.0f + __expf(-g));
            gate[i] = __float2half(g * sig * u);
        }
    }
}

// ============================================================
// rmsnorm_v3: RMSNorm with float4 (8 FP16) vectorized loads.
//
// Thread layout:
//   Each thread t processes 8-element groups at {t*8, t*8+2048, t*8+4096, ...}
//   while t*8 + k*2048 + 7 < dim.
//   Scalar tail: elements [(dim/8)*8, dim-1] (only exists if dim%8 != 0).
//   This partitioning avoids double-counting.
//
// Grid: (num_vectors, 1, 1), Block: (256, 1, 1)
// ============================================================
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void rmsnorm_v3(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    const __half* __restrict__ weight,
    unsigned int dim,
    float eps
) {
    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane   = threadIdx.x % 64;
    const __half* src_vec = src + blockIdx.x * dim;
    __half*       dst_vec = dst + blockIdx.x * dim;

    // Phase 1: partial sum-of-squares — vectorized 8-wide pass
    float sum_sq = 0.0f;

    // Vectorized path: thread t covers {t*8, t*8+2048, t*8+4096, ...}
    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        float4 v4 = *reinterpret_cast<const float4*>(src_vec + i);
        const __half2* h2 = reinterpret_cast<const __half2*>(&v4);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float a = __half2float(__low2half (h2[k]));
            float b = __half2float(__high2half(h2[k]));
            sum_sq += a * a + b * b;
        }
    }
    // Scalar tail: elements in [(dim/8)*8, dim-1] (only if dim%8 != 0)
    // These elements are NOT covered by the vectorized loop above.
    {
        unsigned int vec_dim = (dim / 8) * 8;  // dim rounded DOWN to multiple of 8
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float val = __half2float(src_vec[i]);
            sum_sq += val * val;
        }
    }

    // Phase 2: 6-step butterfly warp reduction (wave64)
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);

    // Phase 3: cross-warp reduction via LDS (4 warps × 64 lanes)
    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 4) {
        float total = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        s_warp[threadIdx.x] = total;
    }
    __syncthreads();
    float total_sq = s_warp[0];

    float rms_inv = rsqrtf(total_sq / (float)dim + eps);

    // Phase 4: normalize — vectorized 8-wide pass
    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        float4 sv4 = *reinterpret_cast<const float4*>(src_vec + i);
        float4 wv4 = *reinterpret_cast<const float4*>(weight  + i);

        const __half2* sv_h2 = reinterpret_cast<const __half2*>(&sv4);
        const __half2* wv_h2 = reinterpret_cast<const __half2*>(&wv4);

        float4 out4;
        __half2* out_h2 = reinterpret_cast<__half2*>(&out4);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float s0 = __half2float(__low2half (sv_h2[k]));
            float s1 = __half2float(__high2half(sv_h2[k]));
            float w0 = __half2float(__low2half (wv_h2[k]));
            float w1 = __half2float(__high2half(wv_h2[k]));
            out_h2[k] = __halves2half2(
                __float2half(s0 * rms_inv * w0),
                __float2half(s1 * rms_inv * w1)
            );
        }
        *reinterpret_cast<float4*>(dst_vec + i) = out4;
    }
    // Scalar tail: elements in [(dim/8)*8, dim-1]
    {
        unsigned int vec_dim = (dim / 8) * 8;
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float val = __half2float(src_vec[i]);
            float w   = __half2float(weight[i]);
            dst_vec[i] = __float2half(val * rms_inv * w);
        }
    }
}

// ============================================================
// skip_rmsnorm_v3: fused residual-add + RMSNorm
//   hidden[i] += residual[i]   (in-place)
//   dst[i] = rmsnorm(hidden[i]) * weight[i]
//
// Uses same vectorized 8-wide loop as rmsnorm_v3.
// Scalar tail for dim%8 != 0 elements.
//
// Grid: (1, 1, 1), Block: (256, 1, 1)
// ============================================================
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void skip_rmsnorm_v3(
    __half* __restrict__ dst,
    __half* __restrict__ hidden,
    const __half* __restrict__ residual,
    const __half* __restrict__ weight,
    unsigned int dim,
    float eps
) {
    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane   = threadIdx.x % 64;

    // Phase 1: add residual, accumulate sum-of-squares — vectorized 8-wide
    float sum_sq = 0.0f;

    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        float4 h4 = *reinterpret_cast<const float4*>(hidden   + i);
        float4 r4 = *reinterpret_cast<const float4*>(residual + i);

        __half2* h_h2       = reinterpret_cast<__half2*>(&h4);
        const __half2* r_h2 = reinterpret_cast<const __half2*>(&r4);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float h0 = __half2float(__low2half (h_h2[k]));
            float h1 = __half2float(__high2half(h_h2[k]));
            float r0 = __half2float(__low2half (r_h2[k]));
            float r1 = __half2float(__high2half(r_h2[k]));
            float v0 = h0 + r0;
            float v1 = h1 + r1;
            h_h2[k] = __halves2half2(__float2half(v0), __float2half(v1));
            sum_sq += v0 * v0 + v1 * v1;
        }
        *reinterpret_cast<float4*>(hidden + i) = h4;
    }
    // Scalar tail: elements in [(dim/8)*8, dim-1]
    {
        unsigned int vec_dim = (dim / 8) * 8;
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float h = __half2float(hidden[i]);
            float r = __half2float(residual[i]);
            float val = h + r;
            hidden[i] = __float2half(val);
            sum_sq += val * val;
        }
    }

    // Phase 2: 6-step butterfly warp reduction (wave64)
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

    float rms_inv = rsqrtf(total_sq / (float)dim + eps);

    // Phase 4: normalize — vectorized 8-wide
    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        float4 hv4 = *reinterpret_cast<const float4*>(hidden + i);
        float4 wv4 = *reinterpret_cast<const float4*>(weight + i);

        const __half2* hv_h2 = reinterpret_cast<const __half2*>(&hv4);
        const __half2* wv_h2 = reinterpret_cast<const __half2*>(&wv4);

        float4 out4;
        __half2* out_h2 = reinterpret_cast<__half2*>(&out4);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float h0 = __half2float(__low2half (hv_h2[k]));
            float h1 = __half2float(__high2half(hv_h2[k]));
            float w0 = __half2float(__low2half (wv_h2[k]));
            float w1 = __half2float(__high2half(wv_h2[k]));
            out_h2[k] = __halves2half2(
                __float2half(h0 * rms_inv * w0),
                __float2half(h1 * rms_inv * w1)
            );
        }
        *reinterpret_cast<float4*>(dst + i) = out4;
    }
    // Scalar tail: elements in [(dim/8)*8, dim-1]
    {
        unsigned int vec_dim = (dim / 8) * 8;
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float val = __half2float(hidden[i]);
            float w   = __half2float(weight[i]);
            dst[i] = __float2half(val * rms_inv * w);
        }
    }
}
