/**
 * INT4 GEMV v7: FP32-only accumulation + 2x register blocking.
 *
 * Improvements over v6:
 *   1. FP32-only accumulation: eliminates 12 FP16 conversion ops per weight word
 *      (8x float2half + 4x halves2half2). Uses direct FP32 multiply-add instead
 *      of fdot2, trading 4 fdot2 for 8 v_fma_f32 but saving 12 conversion ops.
 *   2. 2x register blocking: processes 2 weight words (16 INT4 weights) per
 *      iteration, doubling compute-to-load ratio.
 *   3. Activation preload: loads activation values to VGPRs before compute.
 *
 * Thread layout: Same as v6 (preserves coalesced memory access).
 * Reduction: Same hybrid DPP + minimal LDS approach.
 *
 * Weight format: GPTQ format
 *   B_q4[K/8, N] uint32, K-major row-major
 *   scales[K/group_size, N] FP16
 *   zeros[K/group_size, N] FP16
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
__device__ __forceinline__ void gemv_int4_v7_cooperative(
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

    float acc = 0.0f;

    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int current_group = 0xFFFFFFFF;

    // 2x register blocking: process pairs of weight words
    unsigned int kg = kg_start;
    unsigned int kg_end_even = kg_start + ((kg_end - kg_start) & ~1u);

    // Main loop: 2 weight words per iteration
    for (; kg < kg_end_even; kg += 2) {
        // Load 2 weight words
        unsigned int packed0 = B_q4[kg * N + col];
        unsigned int packed1 = B_q4[(kg + 1) * N + col];

        // Update scale/zero for first word
        unsigned int g0 = kg >> gps_log2;
        if (g0 != current_group) {
            cur_scale = __half2float(scales[g0 * N + col]);
            cur_zero  = __half2float(zeros[g0 * N + col]);
            current_group = g0;
        }

        acc += dequant_dot_fp32(packed0, cur_scale, cur_zero, A, kg << 3);

        // Update scale/zero for second word (might cross group boundary)
        unsigned int g1 = (kg + 1) >> gps_log2;
        if (g1 != current_group) {
            cur_scale = __half2float(scales[g1 * N + col]);
            cur_zero  = __half2float(zeros[g1 * N + col]);
            current_group = g1;
        }

        acc += dequant_dot_fp32(packed1, cur_scale, cur_zero, A, (kg + 1) << 3);
    }

    // Handle odd remaining iteration
    if (kg < kg_end) {
        unsigned int packed = B_q4[kg * N + col];
        unsigned int g = kg >> gps_log2;
        if (g != current_group) {
            cur_scale = __half2float(scales[g * N + col]);
            cur_zero  = __half2float(zeros[g * N + col]);
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
void gemv_int4_v7_t16(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v7_cooperative<16>(A, B_q4, scales, zeros, C, K, N, group_size);
}

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v7_t8(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v7_cooperative<8>(A, B_q4, scales, zeros, C, K, N, group_size);
}

extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v7_t4(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v7_cooperative<4>(A, B_q4, scales, zeros, C, K, N, group_size);
}
