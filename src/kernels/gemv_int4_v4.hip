/**
 * INT4 GEMV v4: Uses __builtin_amdgcn_fdot2 (v_dot2_f32_f16) for the decode path.
 *
 * Approach: Extract weight nibbles into FP16 pairs, use v_dot2_f32_f16 for 2 FMAs
 * per instruction. This avoids activation quantization error entirely (FP16 activations
 * used directly) while gaining throughput vs scalar FMA (2x FMAs per v_dot2 instruction).
 *
 * Compute path per packed uint32 (8 INT4 weights):
 *   1. Extract 2 nibbles → apply (q - zero) * scale → pack as __half2 weight pair
 *   2. Load 2 FP16 activations as __half2
 *   3. __builtin_amdgcn_fdot2(weight_pair, act_pair, acc, false) → FP32 accumulate
 *   4. Repeat 4x per uint32 (4 pairs × 2 weights = 8 weights)
 *
 * vs gemv_int4_v3 (scalar approach):
 *   v3: 8 × (v_bfe_u32 + IALÚ + v_cvt + v_fmac) per uint32
 *   v4: 4 × v_dot2_f32_f16 per uint32 (2× arithmetic throughput)
 *
 * Same cooperative reduction pattern as v3 (3 variants: t4, t8, t16).
 * Workgroup: 256 threads, Block: (256, 1, 1).
 * Grid: (ceil(N / COLS_PER_WG), 1, 1) where COLS_PER_WG = 256 / THREADS_PER_COL.
 *
 * Weight format (GPTQ-compatible, same as v3):
 *   B_q4[K/8, N] uint32, K-major row-major
 *   bits[4i+3:4i] = nibble i ∈ [0, 15] (unsigned INT4)
 *   scales[K/group_size, N] FP16 (group scales)
 *   zeros[K/group_size, N] FP16 (group zeros, unsigned: q_w = q_raw - zero)
 *   dequantized weight: w = (q - zero) * scale
 *
 * Note: zeros are FP16 (not necessarily integer); zero is stored as a float offset.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// dequant_fdot2: Process one packed uint32 (8 INT4 weights) using v_dot2_f32_f16.
//
// Extracts 4 pairs of nibbles, dequantizes to (q - zero) * scale as FP16,
// loads matching activation pairs, and accumulates via fdot2.
//
// Returns accumulated FP32 dot product contribution.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_fdot2(
    unsigned int packed,       // 8 INT4 weight nibbles
    float cur_scale,           // group scale (FP32)
    float cur_zero,            // group zero (FP32)
    const __half* __restrict__ A,
    unsigned int a_base        // base index into A (8 elements: a_base..a_base+7)
) {
    // Load 8 FP16 activations as 4 half2 pairs
    __half2 a01 = *reinterpret_cast<const __half2*>(A + a_base);
    __half2 a23 = *reinterpret_cast<const __half2*>(A + a_base + 2);
    __half2 a45 = *reinterpret_cast<const __half2*>(A + a_base + 4);
    __half2 a67 = *reinterpret_cast<const __half2*>(A + a_base + 6);

    // Extract 8 nibbles using __builtin_amdgcn_ubfe (v_bfe_u32)
    // Each nibble is in [0, 15], dequant: (q - zero) * scale
    float scale = cur_scale;
    float zero  = cur_zero;

    // Dequantize 8 nibbles to FP16 and pack into 4 half2 pairs
    // w_i = ((nibble_i - zero) * scale) as FP16
    float w0 = ((float)__builtin_amdgcn_ubfe(packed,  0, 4) - zero) * scale;
    float w1 = ((float)__builtin_amdgcn_ubfe(packed,  4, 4) - zero) * scale;
    float w2 = ((float)__builtin_amdgcn_ubfe(packed,  8, 4) - zero) * scale;
    float w3 = ((float)__builtin_amdgcn_ubfe(packed, 12, 4) - zero) * scale;
    float w4 = ((float)__builtin_amdgcn_ubfe(packed, 16, 4) - zero) * scale;
    float w5 = ((float)__builtin_amdgcn_ubfe(packed, 20, 4) - zero) * scale;
    float w6 = ((float)__builtin_amdgcn_ubfe(packed, 24, 4) - zero) * scale;
    float w7 = ((float)__builtin_amdgcn_ubfe(packed, 28, 4) - zero) * scale;

    // Pack dequantized weights as half2 pairs
    __half2 w01 = __halves2half2(__float2half(w0), __float2half(w1));
    __half2 w23 = __halves2half2(__float2half(w2), __float2half(w3));
    __half2 w45 = __halves2half2(__float2half(w4), __float2half(w5));
    __half2 w67 = __halves2half2(__float2half(w6), __float2half(w7));

    // v_dot2_f32_f16: acc += w[i]*a[i] + w[i+1]*a[i+1] for each pair
    // __builtin_amdgcn_fdot2(half2 a, half2 b, float c, bool sat)
    // → c + a[0]*b[0] + a[1]*b[1]
    float acc = 0.0f;
    acc = __builtin_amdgcn_fdot2(w01, a01, acc, false);
    acc = __builtin_amdgcn_fdot2(w23, a23, acc, false);
    acc = __builtin_amdgcn_fdot2(w45, a45, acc, false);
    acc = __builtin_amdgcn_fdot2(w67, a67, acc, false);

    return acc;
}

// ---------------------------------------------------------------------------
// Cooperative GEMV with fdot2 compute and LDS reduction.
// Same cooperative reduction pattern as v3 (exact same thread layout).
//
// Thread layout:
//   col_in_wg = threadIdx.x % COLS_PER_WG  (which column within WG)
//   k_split_id = threadIdx.x / COLS_PER_WG (which K-slice)
//   col = blockIdx.x * COLS_PER_WG + col_in_wg
//
// Each thread accumulates K_slice / 8 dot products (one uint32 per iteration).
// Then LDS reduction sums across k_split_id threads for the same col.
// ---------------------------------------------------------------------------
template <int THREADS_PER_COL>
__device__ __forceinline__ void gemv_int4_v4_cooperative(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    const unsigned int COLS_PER_WG = 256 / THREADS_PER_COL;

    // Coalesced mapping: adjacent threads handle adjacent columns
    unsigned int col_in_wg = threadIdx.x % COLS_PER_WG;
    unsigned int k_split_id = threadIdx.x / COLS_PER_WG;
    unsigned int col = blockIdx.x * COLS_PER_WG + col_in_wg;
    if (col >= N) return;

    unsigned int num_k_groups = K >> 3;  // K/8
    // Power-of-2 shift for group scale lookup (same optimization as v3)
    unsigned int groups_per_scale = group_size >> 3;  // group_size/8
    unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);

    unsigned int kg_start = (num_k_groups * k_split_id) / THREADS_PER_COL;
    unsigned int kg_end = (num_k_groups * (k_split_id + 1)) / THREADS_PER_COL;

    float acc = 0.0f;
    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        // Power-of-2 shift for group scale lookup
        unsigned int sg = kg >> gps_log2;
        if (sg != last_sg) {
            cur_scale = __half2float(scales[sg * N + col]);
            cur_zero  = __half2float(zeros[sg * N + col]);
            last_sg = sg;
        }
        unsigned int packed = B_q4[kg * N + col];
        acc += dequant_fdot2(packed, cur_scale, cur_zero, A, kg << 3);
    }

    // LDS reduction across k_split_id threads for same column (identical to v3)
    __shared__ float s_reduce[256];
    s_reduce[threadIdx.x] = acc;
    __syncthreads();

    // First K-split thread per column writes the final sum
    if (k_split_id == 0) {
        float sum = s_reduce[col_in_wg];
        #pragma unroll
        for (int s = 1; s < THREADS_PER_COL; s++) {
            sum += s_reduce[col_in_wg + s * COLS_PER_WG];
        }
        C[col] = __float2half(sum);
    }
}

// ---------------------------------------------------------------------------
// Public kernel variants (matching v3's thread-per-column configurations)
// ---------------------------------------------------------------------------

// 4 threads/col: 64 cols/WG. Best for large N (N>=4096).
// Grid: (ceil(N/64), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v4_t4(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v4_cooperative<4>(A, B_q4, scales, zeros, C, K, N, group_size);
}

// 8 threads/col: 32 cols/WG. For medium N with large K.
// Grid: (ceil(N/32), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v4_t8(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v4_cooperative<8>(A, B_q4, scales, zeros, C, K, N, group_size);
}

// 16 threads/col: 16 cols/WG. For small N with large K.
// Grid: (ceil(N/16), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v4_t16(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v4_cooperative<16>(A, B_q4, scales, zeros, C, K, N, group_size);
}
