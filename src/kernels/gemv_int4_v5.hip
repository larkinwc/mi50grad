/**
 * INT4 GEMV v5: Hybrid DPP + minimal LDS reduction with row-major thread layout.
 *
 * Keeps the same row-major thread layout as v4 (preserving memory coalescing),
 * but replaces the full-WG LDS tree reduction with a two-phase approach:
 *
 * Phase 1: Intra-wavefront DPP reduction (no LDS, no barriers).
 *   In the row-major layout with COLS_PER_WG = 256/TPC:
 *   - For t4  (TPC=4,  CPW=64):  1 thread/wf per col → no intra-wf reduction
 *   - For t8  (TPC=8,  CPW=32):  2 threads/wf per col → 1 shfl step
 *   - For t16 (TPC=16, CPW=16):  4 threads/wf per col → 2 shfl steps
 *
 *   For t16 example (wavefront 0, threads 0..63):
 *     col_in_wg = tid % 16  → threads 0..15 cover cols 0..15
 *     k_split   = tid / 16  → within wf0, k_splits 0,1,2,3 per col
 *     shfl_down(16): thread 0 gets sum of k_splits 0 and 1 for col 0
 *     shfl_down(32): thread 0 gets sum of k_splits 0-3 for col 0
 *   → 1/4 of reduction done without LDS!
 *
 * Phase 2: Minimal LDS for cross-wavefront reduction.
 *   After phase 1, only COLS_PER_WG active threads (one per wf) hold partial sums.
 *   For t16: 4 wavefronts × 16 partial sums = 64 LDS writes → sum 4 values per col.
 *   For t8:  4 wavefronts × 32 partial sums = 128 LDS writes → sum 4 values per col.
 *   For t4:  4 wavefronts × 64 partial sums = 256 LDS writes → sum 4 values per col.
 *   vs v4: always 256 LDS writes and sum TPC values per col.
 *
 * Benefits:
 *   - t16: LDS stores 64 (vs 256), reduction reads 4 (vs 16) → 4x less LDS work
 *   - t8:  LDS stores 128 (vs 256), reduction reads 4 (vs 8) → 2x less LDS work
 *   - t4:  same LDS work as v4 (no intra-wf benefit)
 *   - Coalesced memory access preserved (row-major layout unchanged)
 *   - No __syncthreads() saved (one barrier still needed), but reduced LDS traffic
 *
 * Weight format: same GPTQ format as v3/v4.
 *   B_q4[K/8, N] uint32, K-major row-major
 *   scales[K/group_size, N] FP16
 *   zeros[K/group_size, N]  FP16
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// dequant_fdot2: identical to v4 — process one uint32 (8 INT4 weights) via fdot2.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dequant_fdot2(
    unsigned int packed,
    float cur_scale,
    float cur_zero,
    const __half* __restrict__ A,
    unsigned int a_base
) {
    __half2 a01 = *reinterpret_cast<const __half2*>(A + a_base);
    __half2 a23 = *reinterpret_cast<const __half2*>(A + a_base + 2);
    __half2 a45 = *reinterpret_cast<const __half2*>(A + a_base + 4);
    __half2 a67 = *reinterpret_cast<const __half2*>(A + a_base + 6);

    float scale = cur_scale;
    float zero  = cur_zero;

    float w0 = ((float)__builtin_amdgcn_ubfe(packed,  0, 4) - zero) * scale;
    float w1 = ((float)__builtin_amdgcn_ubfe(packed,  4, 4) - zero) * scale;
    float w2 = ((float)__builtin_amdgcn_ubfe(packed,  8, 4) - zero) * scale;
    float w3 = ((float)__builtin_amdgcn_ubfe(packed, 12, 4) - zero) * scale;
    float w4 = ((float)__builtin_amdgcn_ubfe(packed, 16, 4) - zero) * scale;
    float w5 = ((float)__builtin_amdgcn_ubfe(packed, 20, 4) - zero) * scale;
    float w6 = ((float)__builtin_amdgcn_ubfe(packed, 24, 4) - zero) * scale;
    float w7 = ((float)__builtin_amdgcn_ubfe(packed, 28, 4) - zero) * scale;

    __half2 w01 = __halves2half2(__float2half(w0), __float2half(w1));
    __half2 w23 = __halves2half2(__float2half(w2), __float2half(w3));
    __half2 w45 = __halves2half2(__float2half(w4), __float2half(w5));
    __half2 w67 = __halves2half2(__float2half(w6), __float2half(w7));

    float acc = 0.0f;
    acc = __builtin_amdgcn_fdot2(w01, a01, acc, false);
    acc = __builtin_amdgcn_fdot2(w23, a23, acc, false);
    acc = __builtin_amdgcn_fdot2(w45, a45, acc, false);
    acc = __builtin_amdgcn_fdot2(w67, a67, acc, false);

    return acc;
}

// ---------------------------------------------------------------------------
// v5 cooperative GEMV: row-major thread layout + hybrid DPP/LDS reduction.
//
// Thread layout (v5 — same as v4, row-major):
//   col_in_wg = threadIdx.x % COLS_PER_WG  (which column within WG)
//   k_split_id = threadIdx.x / COLS_PER_WG (which K-slice)
//   col = blockIdx.x * COLS_PER_WG + col_in_wg
//
// After accumulation, reduce via two phases:
//   Phase 1: Intra-wavefront shfl reduction (no LDS needed).
//     Compute how many k_splits each wavefront handles:
//     wf_size=64, threads_per_wf_per_col = 64/COLS_PER_WG = TPC/4 (for 4 wfs)
//     Actually: 256 threads / 64 per wf = 4 wfs.
//     Threads in wf i: [64*i .. 64*i+63]. col_in_wg = tid % CPW.
//     For a given col_in_wg, within wf i: k_split values are
//       [64*i/CPW .. (64*i+63)/CPW] = i*(64/CPW) .. i*(64/CPW) + (64/CPW-1)
//     "intra-wf k-splits per col" = 64/CPW = TPC/4.
//     These threads are at positions col_in_wg + CPW*j for j=0..TPC/4-1 within the wf.
//     Shfl_down by CPW, 2*CPW, ... TPC/4*CPW reduces them.
//     For t4:  CPW=64, TPC/4=1 → no intra-wf reduction (1 thread per col per wf)
//     For t8:  CPW=32, TPC/4=2 → shfl_down(32): 2→1 per wf
//     For t16: CPW=16, TPC/4=4 → shfl_down(16) + shfl_down(32): 4→1 per wf
//
//   Phase 2: LDS cross-wavefront reduction.
//     4 wfs each contribute 1 partial sum per col → write to LDS, sum.
//     LDS size: COLS_PER_WG * 4 floats (one row per wavefront).
//     Only 1 syncthreads needed (vs 1 in v4, same).
//     v4 LDS: 256 floats written, TPC sums read per col.
//     v5 LDS: COLS_PER_WG * 4 = CPW*4 floats written, 4 sums read per col.
// ---------------------------------------------------------------------------
template <int THREADS_PER_COL>
__device__ __forceinline__ void gemv_int4_v5_cooperative(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    constexpr unsigned int COLS_PER_WG   = 256 / THREADS_PER_COL;
    constexpr unsigned int NUM_WF        = 4;                 // 256 / 64
    constexpr unsigned int TPC_PER_WF   = THREADS_PER_COL / NUM_WF; // intra-wf k-splits per col

    // Row-major mapping: same as v4 (preserves coalesced memory access)
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
    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        unsigned int sg = kg >> gps_log2;
        if (sg != last_sg) {
            cur_scale = __half2float(scales[sg * N + col]);
            cur_zero  = __half2float(zeros[sg * N + col]);
            last_sg = sg;
        }
        unsigned int packed = B_q4[kg * N + col];
        acc += dequant_fdot2(packed, cur_scale, cur_zero, A, kg << 3);
    }

    // Phase 1: Intra-wavefront DPP reduction.
    // Within each wavefront, TPC_PER_WF threads handle the same col at stride COLS_PER_WG.
    // Reduce them using shfl_down in order of increasing offset (smallest first).
    //
    // For t16 (CPW=16, TPC_PER_WF=4): shfl_down(16) then shfl_down(32)
    //   After shfl_down(16): thread 0 has k_splits 0+1, thread 32 has k_splits 2+3
    //   After shfl_down(32): thread 0 has k_splits 0+1+2+3 ✓
    // For t8  (CPW=32, TPC_PER_WF=2): shfl_down(32) only
    //   Thread 0 has k_splits 0+1 per wf ✓
    // For t4  (CPW=64, TPC_PER_WF=1): no intra-wf reduction
    if (TPC_PER_WF >= 2) acc += __shfl_down(acc, COLS_PER_WG);       // t8+t16: offset CPW (first)
    if (TPC_PER_WF >= 4) acc += __shfl_down(acc, COLS_PER_WG * 2);   // t16: offset 2*CPW (second)

    // After phase 1: only threads with k_split_id % (NUM_WF) == 0 hold valid wavefront partials.
    // I.e., k_split_id ∈ {0, TPC/NUM_WF, 2*TPC/NUM_WF, 3*TPC/NUM_WF} are the "leading" threads.
    // But in v4 terminology: each wavefront's "leading" thread for col is
    // the one with the smallest k_split_id in that wavefront.
    // Wf j covers k_splits [j*TPC/4 .. (j+1)*TPC/4 - 1].
    // After intra-wf reduction, the thread at k_split = j*TPC/4 holds the partial sum.

    // Phase 2: LDS cross-wavefront reduction.
    // Layout: LDS[4 * COLS_PER_WG] floats — row j (wf j) × col_in_wg
    // Only the leading thread (k_split_id % TPC_PER_WF == 0, i.e. k_split_id == wf*TPC_PER_WF)
    // writes its partial to LDS.
    __shared__ float s_reduce[NUM_WF * COLS_PER_WG];  // 4 * CPW floats

    // The leading thread of wf j is: tid = col_in_wg + j*64
    // Its k_split_id = (col_in_wg + j*64) / COLS_PER_WG = j * (64/COLS_PER_WG) = j * TPC_PER_WF
    // So check: k_split_id % TPC_PER_WF == 0 (or equivalently, k_split_id is a multiple of TPC_PER_WF)
    if ((k_split_id & (TPC_PER_WF - 1)) == 0) {
        unsigned int wf_id = k_split_id / TPC_PER_WF;  // which wavefront (0..3)
        s_reduce[wf_id * COLS_PER_WG + col_in_wg] = acc;
    }
    __syncthreads();

    // First k_split per column (k_split_id == 0) sums 4 wavefront partials.
    if (k_split_id == 0) {
        float sum = s_reduce[0 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[1 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[2 * COLS_PER_WG + col_in_wg];
        sum += s_reduce[3 * COLS_PER_WG + col_in_wg];
        C[col] = __float2half(sum);
    }
}

// ---------------------------------------------------------------------------
// Public kernel variants
// ---------------------------------------------------------------------------

// 4 threads/col: 64 cols/WG
// Grid: (ceil(N/64), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v5_t4(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v5_cooperative<4>(A, B_q4, scales, zeros, C, K, N, group_size);
}

// 8 threads/col: 32 cols/WG
// Grid: (ceil(N/32), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v5_t8(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v5_cooperative<8>(A, B_q4, scales, zeros, C, K, N, group_size);
}

// 16 threads/col: 16 cols/WG
// Grid: (ceil(N/16), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v5_t16(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v5_cooperative<16>(A, B_q4, scales, zeros, C, K, N, group_size);
}
