/**
 * INT4 GEMV v5 AWQ variant: Hybrid DPP + minimal LDS reduction, no zero-point subtraction.
 *
 * AWQ dequantization: w = q * scale
 * GPTQ dequantization: w = (q - zero) * scale
 *
 * By eliminating the zero-point, we save:
 *   - 8 v_sub_f32 instructions per uint32 (one per nibble)
 *   - The zeros tensor is not loaded at all (no pointer argument, no memory traffic)
 *
 * This is purely a compute/bandwidth savings: ~1 fewer load per scale group
 * and 8 fewer subtractions per inner loop iteration.
 *
 * Everything else is identical to gemv_int4_v5.hip:
 *   - Same thread layout: row-major, col_in_wg = tid % COLS_PER_WG
 *   - Same hybrid DPP + minimal LDS two-phase reduction
 *   - Same public kernel signatures (t4, t8, t16)
 *   - Same weight format: B_q4[K/8, N] uint32, scales[K/group_size, N] FP16
 *
 * The kernel signature drops the `zeros` pointer parameter entirely.
 *
 * Weight format (AWQ-compatible):
 *   B_q4[K/8, N] uint32, K-major row-major
 *   bits[4i+3:4i] = nibble i ∈ [0, 15] (unsigned INT4)
 *   scales[K/group_size, N] FP16 (group scales)
 *   NO zeros tensor
 *   dequantized weight: w = q * scale
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// dequant_fdot2_awq: Process one packed uint32 (8 INT4 weights) — AWQ variant.
//
// Extracts 4 pairs of nibbles, dequantizes to q * scale (no zero-point),
// packs as FP16 pairs, and accumulates via v_dot2_f32_f16.
//
// Saves 8 v_sub_f32 instructions vs the GPTQ version.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_fdot2_awq(
    unsigned int packed,
    float cur_scale,
    const __half* __restrict__ A,
    unsigned int a_base
) {
    // Load 8 FP16 activations as 4 half2 pairs
    __half2 a01 = *reinterpret_cast<const __half2*>(A + a_base);
    __half2 a23 = *reinterpret_cast<const __half2*>(A + a_base + 2);
    __half2 a45 = *reinterpret_cast<const __half2*>(A + a_base + 4);
    __half2 a67 = *reinterpret_cast<const __half2*>(A + a_base + 6);

    float scale = cur_scale;

    // AWQ: w = q * scale (no zero subtraction)
    float w0 = (float)__builtin_amdgcn_ubfe(packed,  0, 4) * scale;
    float w1 = (float)__builtin_amdgcn_ubfe(packed,  4, 4) * scale;
    float w2 = (float)__builtin_amdgcn_ubfe(packed,  8, 4) * scale;
    float w3 = (float)__builtin_amdgcn_ubfe(packed, 12, 4) * scale;
    float w4 = (float)__builtin_amdgcn_ubfe(packed, 16, 4) * scale;
    float w5 = (float)__builtin_amdgcn_ubfe(packed, 20, 4) * scale;
    float w6 = (float)__builtin_amdgcn_ubfe(packed, 24, 4) * scale;
    float w7 = (float)__builtin_amdgcn_ubfe(packed, 28, 4) * scale;

    // Pack dequantized weights as half2 pairs
    __half2 w01 = __halves2half2(__float2half(w0), __float2half(w1));
    __half2 w23 = __halves2half2(__float2half(w2), __float2half(w3));
    __half2 w45 = __halves2half2(__float2half(w4), __float2half(w5));
    __half2 w67 = __halves2half2(__float2half(w6), __float2half(w7));

    // v_dot2_f32_f16 accumulation
    float acc = 0.0f;
    acc = __builtin_amdgcn_fdot2(w01, a01, acc, false);
    acc = __builtin_amdgcn_fdot2(w23, a23, acc, false);
    acc = __builtin_amdgcn_fdot2(w45, a45, acc, false);
    acc = __builtin_amdgcn_fdot2(w67, a67, acc, false);

    return acc;
}

// ---------------------------------------------------------------------------
// AWQ cooperative GEMV: same row-major layout + hybrid DPP/LDS reduction as v5,
// but without the zeros pointer/load.
// ---------------------------------------------------------------------------
template <int THREADS_PER_COL>
__device__ __forceinline__ void gemv_int4_v5_awq_cooperative(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    constexpr unsigned int COLS_PER_WG  = 256 / THREADS_PER_COL;
    constexpr unsigned int NUM_WF       = 4;                 // 256 / 64
    constexpr unsigned int TPC_PER_WF   = THREADS_PER_COL / NUM_WF;

    // Row-major mapping (same as v5)
    unsigned int col_in_wg = threadIdx.x % COLS_PER_WG;
    unsigned int k_split_id = threadIdx.x / COLS_PER_WG;
    unsigned int col = blockIdx.x * COLS_PER_WG + col_in_wg;
    if (col >= N) return;

    unsigned int num_k_groups = K >> 3;  // K/8
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);

    unsigned int kg_start = (num_k_groups * k_split_id) / THREADS_PER_COL;
    unsigned int kg_end   = (num_k_groups * (k_split_id + 1)) / THREADS_PER_COL;

    float acc = 0.0f;
    float cur_scale = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        unsigned int sg = kg >> gps_log2;
        if (sg != last_sg) {
            cur_scale = __half2float(scales[sg * N + col]);
            // AWQ: no zero load here
            last_sg = sg;
        }
        unsigned int packed = B_q4[kg * N + col];
        acc += dequant_fdot2_awq(packed, cur_scale, A, kg << 3);
    }

    // Phase 1: Intra-wavefront DPP reduction (identical to v5)
    if (TPC_PER_WF >= 2) acc += __shfl_down(acc, COLS_PER_WG);
    if (TPC_PER_WF >= 4) acc += __shfl_down(acc, COLS_PER_WG * 2);

    // Phase 2: LDS cross-wavefront reduction (identical to v5)
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

// ---------------------------------------------------------------------------
// Public kernel variants — AWQ (no zeros parameter)
// ---------------------------------------------------------------------------

// 4 threads/col: 64 cols/WG
// Grid: (ceil(N/64), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v5_awq_t4(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v5_awq_cooperative<4>(A, B_q4, scales, C, K, N, group_size);
}

// 8 threads/col: 32 cols/WG
// Grid: (ceil(N/32), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v5_awq_t8(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v5_awq_cooperative<8>(A, B_q4, scales, C, K, N, group_size);
}

// 16 threads/col: 16 cols/WG
// Grid: (ceil(N/16), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v5_awq_t16(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v5_awq_cooperative<16>(A, B_q4, scales, C, K, N, group_size);
}
