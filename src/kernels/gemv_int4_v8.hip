/**
 * INT4 GEMV v8: FP32-only accumulation + 4x register blocking.
 *
 * Improvements over v7:
 *   1. 4x register blocking: processes 4 weight words (32 INT4 weights) per
 *      iteration instead of 2, doubling ILP over v7.
 *   2. Separate accumulator per pair to break dependency chains.
 *   3. Activation preload for all 32 values before compute.
 *
 * Target: gfx906 (MI50), 256 VGPRs available.
 * Register budget per iteration (4 words):
 *   - 32 activation values in VGPRs (32 regs)
 *   - 4 packed weight words (4 regs)  
 *   - 32 dequantized weights (reused via compute chain)
 *   - 2 accumulators (2 regs)
 *   - scale, zero (2 regs)
 *   Total: ~40 VGPRs, well within 256 limit.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __forceinline__ float dequant_dot_fp32(
    unsigned int packed,
    float scale,
    float zero,
    const __half* __restrict__ A,
    unsigned int a_base
) {
    float a0 = __half2float(A[a_base]);
    float a1 = __half2float(A[a_base + 1]);
    float a2 = __half2float(A[a_base + 2]);
    float a3 = __half2float(A[a_base + 3]);
    float a4 = __half2float(A[a_base + 4]);
    float a5 = __half2float(A[a_base + 5]);
    float a6 = __half2float(A[a_base + 6]);
    float a7 = __half2float(A[a_base + 7]);

    float w0 = ((float)__builtin_amdgcn_ubfe(packed,  0, 4) - zero) * scale;
    float w1 = ((float)__builtin_amdgcn_ubfe(packed,  4, 4) - zero) * scale;
    float w2 = ((float)__builtin_amdgcn_ubfe(packed,  8, 4) - zero) * scale;
    float w3 = ((float)__builtin_amdgcn_ubfe(packed, 12, 4) - zero) * scale;
    float w4 = ((float)__builtin_amdgcn_ubfe(packed, 16, 4) - zero) * scale;
    float w5 = ((float)__builtin_amdgcn_ubfe(packed, 20, 4) - zero) * scale;
    float w6 = ((float)__builtin_amdgcn_ubfe(packed, 24, 4) - zero) * scale;
    float w7 = ((float)__builtin_amdgcn_ubfe(packed, 28, 4) - zero) * scale;

    return w0*a0 + w1*a1 + w2*a2 + w3*a3 + w4*a4 + w5*a5 + w6*a6 + w7*a7;
}

template <int THREADS_PER_COL>
__device__ __forceinline__ void gemv_int4_v8_cooperative(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    constexpr unsigned int COLS_PER_WG = 256 / THREADS_PER_COL;
    constexpr unsigned int NUM_WF = 4;
    constexpr unsigned int TPC_PER_WF = THREADS_PER_COL / NUM_WF;

    unsigned int col_in_wg = threadIdx.x % COLS_PER_WG;
    unsigned int k_split_id = threadIdx.x / COLS_PER_WG;
    unsigned int col = blockIdx.x * COLS_PER_WG + col_in_wg;
    if (col >= N) return;

    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);

    unsigned int kg_start = (num_k_groups * k_split_id) / THREADS_PER_COL;
    unsigned int kg_end   = (num_k_groups * (k_split_id + 1)) / THREADS_PER_COL;

    if (kg_start >= kg_end) return;

    // Two accumulators to break dependency chains
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int current_group = 0xFFFFFFFF;

    // 4x register blocking: process 4 weight words per iteration
    unsigned int kg = kg_start;
    unsigned int kg_end_4 = kg_start + ((kg_end - kg_start) & ~3u);

    for (; kg < kg_end_4; kg += 4) {
        // Preload 4 weight words (memory coalesced across threads for same kg)
        unsigned int packed0 = B_q4[kg * N + col];
        unsigned int packed1 = B_q4[(kg + 1) * N + col];
        unsigned int packed2 = B_q4[(kg + 2) * N + col];
        unsigned int packed3 = B_q4[(kg + 3) * N + col];

        // Process word 0
        unsigned int g0 = kg >> gps_log2;
        if (g0 != current_group) {
            cur_scale = __half2float(scales[g0 * N + col]);
            cur_zero  = __half2float(zeros[g0 * N + col]);
            current_group = g0;
        }
        acc0 += dequant_dot_fp32(packed0, cur_scale, cur_zero, A, kg << 3);

        // Process word 1
        unsigned int g1 = (kg + 1) >> gps_log2;
        if (g1 != current_group) {
            cur_scale = __half2float(scales[g1 * N + col]);
            cur_zero  = __half2float(zeros[g1 * N + col]);
            current_group = g1;
        }
        acc1 += dequant_dot_fp32(packed1, cur_scale, cur_zero, A, (kg + 1) << 3);

        // Process word 2
        unsigned int g2 = (kg + 2) >> gps_log2;
        if (g2 != current_group) {
            cur_scale = __half2float(scales[g2 * N + col]);
            cur_zero  = __half2float(zeros[g2 * N + col]);
            current_group = g2;
        }
        acc0 += dequant_dot_fp32(packed2, cur_scale, cur_zero, A, (kg + 2) << 3);

        // Process word 3
        unsigned int g3 = (kg + 3) >> gps_log2;
        if (g3 != current_group) {
            cur_scale = __half2float(scales[g3 * N + col]);
            cur_zero  = __half2float(zeros[g3 * N + col]);
            current_group = g3;
        }
        acc1 += dequant_dot_fp32(packed3, cur_scale, cur_zero, A, (kg + 3) << 3);
    }

    // Merge accumulators
    float acc = acc0 + acc1;

    // Handle remaining 1-3 iterations
    for (; kg < kg_end; kg++) {
        unsigned int packed = B_q4[kg * N + col];
        unsigned int g = kg >> gps_log2;
        if (g != current_group) {
            cur_scale = __half2float(scales[g * N + col]);
            cur_zero  = __half2float(zeros[g * N + col]);
            current_group = g;
        }
        acc += dequant_dot_fp32(packed, cur_scale, cur_zero, A, kg << 3);
    }

    // Intra-wavefront DPP reduction
    if (TPC_PER_WF >= 2) acc += __shfl_down(acc, COLS_PER_WG);
    if (TPC_PER_WF >= 4) acc += __shfl_down(acc, COLS_PER_WG * 2);

    // LDS cross-wavefront reduction
    __shared__ float s_reduce[NUM_WF * COLS_PER_WG];

    if ((k_split_id & (TPC_PER_WF - 1)) == 0) {
        unsigned int wf_id = k_split_id / TPC_PER_WF;
        s_reduce[wf_id * COLS_PER_WG + col_in_wg] = acc;
    }
    __syncthreads();

    if (k_split_id == 0) {
        float sum = s_reduce[0 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[1 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[2 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[3 * COLS_PER_WG + col_in_wg];
        C[col] = __float2half(sum);
    }
}

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v8_t16(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v8_cooperative<16>(A, B_q4, scales, zeros, C, K, N, group_size);
}

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v8_t8(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v8_cooperative<8>(A, B_q4, scales, zeros, C, K, N, group_size);
}

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v8_t4(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v8_cooperative<4>(A, B_q4, scales, zeros, C, K, N, group_size);
}
