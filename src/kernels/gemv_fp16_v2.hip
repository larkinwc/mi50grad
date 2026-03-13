/**
 * Optimized FP16 GEMV for decode (M=1) on gfx906.
 *
 * Key optimizations vs v1:
 * - 4 rows per WG (maps to 4 wavefronts): each wavefront computes one output
 * - Wavefront shuffle reduction (DPP) instead of LDS barrier tree
 * - Vectorized float4 loads (8 FP16 per load)
 * - x stays in L2 cache (10KB for K=5120, shared across all WGs)
 *
 * Grid: (ceil(N/4), 1, 1), Block: (256, 1, 1)
 * out[N] = W[N, K] * x[K]  (FP16 in/out, FP32 accumulator)
 *
 * Optional residual epilogue:
 *   If residual != nullptr: out[i] = gemv_result + residual[i]
 *   This fuses the residual-add into the GEMV kernel, eliminating a
 *   separate residual_add launch (saves 1 kernel launch for out_proj).
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void gemv_fp16_v2(
    const __half* __restrict__ x,
    const __half* __restrict__ W,
    __half* __restrict__ out,
    unsigned int K,
    unsigned int N,
    const __half* __restrict__ residual  // optional: if non-null, out[i] += residual[i]
) {
    // 4 rows per WG, 64 threads (1 wavefront) per row
    const unsigned int ROWS_PER_WG = 4;
    unsigned int row = blockIdx.x * ROWS_PER_WG + threadIdx.x / 64;
    unsigned int lane = threadIdx.x % 64;  // lane within wavefront

    if (row >= N) return;

    float acc = 0.0f;

    // Vectorized loads: float4 = 4 floats = 8 FP16
    const float4* w_row = (const float4*)(W + (size_t)row * K);
    const float4* x_v = (const float4*)x;
    unsigned int vec_len = K >> 3;  // K / 8

    // Each of 64 threads processes vec_len/64 float4 elements
    for (unsigned int i = lane; i < vec_len; i += 64) {
        float4 wv = w_row[i];
        float4 xv = x_v[i];

        const __half2* wh = (const __half2*)&wv;
        const __half2* xh = (const __half2*)&xv;

        // 4 × v_dot2_f32_f16: acc += w.x*x.x + w.y*x.y (2 FMAs per instruction)
        acc = __builtin_amdgcn_fdot2(*reinterpret_cast<const _Float16_2*>(&wh[0]),
                                      *reinterpret_cast<const _Float16_2*>(&xh[0]), acc, false);
        acc = __builtin_amdgcn_fdot2(*reinterpret_cast<const _Float16_2*>(&wh[1]),
                                      *reinterpret_cast<const _Float16_2*>(&xh[1]), acc, false);
        acc = __builtin_amdgcn_fdot2(*reinterpret_cast<const _Float16_2*>(&wh[2]),
                                      *reinterpret_cast<const _Float16_2*>(&xh[2]), acc, false);
        acc = __builtin_amdgcn_fdot2(*reinterpret_cast<const _Float16_2*>(&wh[3]),
                                      *reinterpret_cast<const _Float16_2*>(&xh[3]), acc, false);
    }

    // Handle remainder (K not divisible by 8)
    if (lane == 0) {
        unsigned int rem_start = vec_len * 8;
        for (unsigned int k = rem_start; k < K; k++) {
            acc += __half2float(W[(size_t)row * K + k]) * __half2float(x[k]);
        }
    }

    // Wavefront shuffle reduction (64 threads -> 1)
    // On gfx906: wavefront = 64, __shfl_down uses DPP
    acc += __shfl_down(acc, 32);
    acc += __shfl_down(acc, 16);
    acc += __shfl_down(acc, 8);
    acc += __shfl_down(acc, 4);
    acc += __shfl_down(acc, 2);
    acc += __shfl_down(acc, 1);

    if (lane == 0) {
        // Optional residual epilogue: add residual[row] if pointer is non-null
        if (residual != nullptr) {
            out[row] = __float2half(acc + __half2float(residual[row]));
        } else {
            out[row] = __float2half(acc);
        }
    }
}
