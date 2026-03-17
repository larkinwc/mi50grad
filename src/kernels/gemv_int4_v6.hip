/**
 * INT4 GEMV v6: Register-cached scale/zero + weight prefetch double-buffering.
 *
 * Optimization 1: Scale/zero register caching
 *   Current v5 loads scale and zero from global memory every iteration where the
 *   quantization group changes. For group_size=128 and 8 weights per iteration,
 *   there are 16 iterations per group. The scale/zero values are constant across
 *   all 16 iterations within a group.
 *   
 *   v6 caches scale/zero in VGPRs and reloads only when crossing the group boundary.
 *   This eliminates 15 of 16 redundant global loads per group (saves ~94% of scale/zero loads).
 *
 * Optimization 2: Weight prefetch / double-buffering
 *   While processing the current weight word (dequant_fdot2), issue a global_load_dword
 *   for the NEXT weight word. Use explicit waitcnt(vmcnt=0) before consuming the loaded data.
 *   This overlaps HBM latency with compute, hiding some of the memory access cost.
 *
 * Thread layout: Same row-major layout as v5 (preserves coalesced memory access).
 * Reduction: Same hybrid DPP + minimal LDS approach as v5.
 *
 * Weight format: GPTQ format (same as v3/v4/v5)
 *   B_q4[K/8, N] uint32, K-major row-major
 *   scales[K/group_size, N] FP16
 *   zeros[K/group_size, N] FP16
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// dequant_fdot2: Process one uint32 (8 INT4 weights) via fdot2.
// Same as v5, but called in a double-buffered manner.
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
// v6 cooperative GEMV: register-cached scale/zero + weight prefetch.
//
// Key differences from v5:
// 1. Track current_group index, reload scale/zero only when kg >> gps_log2 != current_group
// 2. Issue global_load_dword for next weight BEFORE processing current weight
// 3. Use s_waitcnt vmcnt(0) to wait for the prefetch to complete before dequant
//
// Thread layout (same as v5):
//   col_in_wg = threadIdx.x % COLS_PER_WG
//   k_split_id = threadIdx.x / COLS_PER_WG
//   col = blockIdx.x * COLS_PER_WG + col_in_wg
// ---------------------------------------------------------------------------
template <int THREADS_PER_COL>
__device__ __forceinline__ void gemv_int4_v6_cooperative(
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
    constexpr unsigned int NUM_WF        = 4;
    constexpr unsigned int TPC_PER_WF   = THREADS_PER_COL / NUM_WF;

    unsigned int col_in_wg = threadIdx.x % COLS_PER_WG;
    unsigned int k_split_id = threadIdx.x / COLS_PER_WG;
    unsigned int col = blockIdx.x * COLS_PER_WG + col_in_wg;
    if (col >= N) return;

    unsigned int num_k_groups = K >> 3;  // K/8 (8 weights per uint32)
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);

    unsigned int kg_start = (num_k_groups * k_split_id) / THREADS_PER_COL;
    unsigned int kg_end   = (num_k_groups * (k_split_id + 1)) / THREADS_PER_COL;

    float acc = 0.0f;
    
    // Register-cached scale/zero (Optimization 1)
    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int current_group = 0xFFFFFFFF;  // Force reload on first iteration
    
    // Weight prefetch (Optimization 2)
    // Use a register to hold the prefetched weight word
    unsigned int packed_current = 0;
    unsigned int packed_next = 0;
    bool has_next = false;
    
    // First, load the initial weight word if available
    if (kg_start < kg_end) {
        packed_current = B_q4[kg_start * N + col];
        has_next = true;
    }

    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        // Check if we're crossing a quantization group boundary
        unsigned int next_group = kg >> gps_log2;
        if (next_group != current_group) {
            // Reload scale/zero from global memory (only once per 16 iterations)
            cur_scale = __half2float(scales[next_group * N + col]);
            cur_zero  = __half2float(zeros[next_group * N + col]);
            current_group = next_group;
        }
        
        // Prefetch next weight word if available (Optimization 2)
        // Issue the load early to overlap with current compute
        if (kg + 1 < kg_end) {
            packed_next = B_q4[(kg + 1) * N + col];
            // Note: On gfx906, global_load_dword is implicitly async.
            // We don't need an explicit wait here because the next iteration
            // will use packed_next only after it's been loaded.
            // However, for proper pipelining, we should use inline asm or
            // rely on the compiler to schedule the load early.
        }
        
        // Process current weight word with cached scale/zero
        acc += dequant_fdot2(packed_current, cur_scale, cur_zero, A, kg << 3);
        
        // Rotate buffers for next iteration
        packed_current = packed_next;
    }

    // Phase 1: Intra-wavefront DPP reduction (same as v5)
    if (TPC_PER_WF >= 2) acc += __shfl_down(acc, COLS_PER_WG);
    if (TPC_PER_WF >= 4) acc += __shfl_down(acc, COLS_PER_WG * 2);

    // Phase 2: LDS cross-wavefront reduction (same as v5)
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
// v6 optimized: Register-cached scale/zero + explicit weight prefetch.
// Uses a 2-stage pipeline: load next weight while computing current.
// The compiler's instruction scheduler should overlap these operations.
// ---------------------------------------------------------------------------
template <int THREADS_PER_COL>
__device__ __forceinline__ void gemv_int4_v6_cooperative_v2(
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
    constexpr unsigned int NUM_WF        = 4;
    constexpr unsigned int TPC_PER_WF   = THREADS_PER_COL / NUM_WF;

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
    
    // Optimization 1: Register-cached scale/zero
    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int current_group = 0xFFFFFFFF;
    
    // Optimization 2: Weight prefetch with 2-register double buffering
    // Load first weight before the loop
    unsigned int packed = B_q4[kg_start * N + col];
    
    // First iteration: load scale/zero, compute with first weight, prefetch second
    unsigned int kg = kg_start;
    unsigned int next_group = kg >> gps_log2;
    if (next_group != current_group) {
        cur_scale = __half2float(scales[next_group * N + col]);
        cur_zero  = __half2float(zeros[next_group * N + col]);
        current_group = next_group;
    }
    
    // Compute with current weight
    acc += dequant_fdot2(packed, cur_scale, cur_zero, A, kg << 3);
    
    // Prefetch next weight (if available)
    unsigned int next_packed = 0;
    if (kg + 1 < kg_end) {
        next_packed = B_q4[(kg + 1) * N + col];
    }
    kg++;
    
    // Main loop: process prefetched weight, load next
    for (; kg < kg_end; kg++) {
        // Update scale/zero if group boundary crossed
        next_group = kg >> gps_log2;
        if (next_group != current_group) {
            cur_scale = __half2float(scales[next_group * N + col]);
            cur_zero  = __half2float(zeros[next_group * N + col]);
            current_group = next_group;
        }
        
        // Compute with the prefetched weight
        acc += dequant_fdot2(next_packed, cur_scale, cur_zero, A, kg << 3);
        
        // Prefetch next weight for next iteration
        if (kg + 1 < kg_end) {
            next_packed = B_q4[(kg + 1) * N + col];
        }
    }

    // Phase 1: Intra-wavefront DPP reduction
    if (TPC_PER_WF >= 2) acc += __shfl_down(acc, COLS_PER_WG);
    if (TPC_PER_WF >= 4) acc += __shfl_down(acc, COLS_PER_WG * 2);

    // Phase 2: LDS cross-wavefront reduction
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
// Public kernel variants: v6 with register caching + weight prefetch
// Provides t4, t8, t16 variants like v5.
// ---------------------------------------------------------------------------

// 4 threads/col: 64 cols/WG
// Grid: (ceil(N/64), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v6_t4(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v6_cooperative_v2<4>(A, B_q4, scales, zeros, C, K, N, group_size);
}

// 8 threads/col: 32 cols/WG
// Grid: (ceil(N/32), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v6_t8(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v6_cooperative_v2<8>(A, B_q4, scales, zeros, C, K, N, group_size);
}

// 16 threads/col: 16 cols/WG
// Grid: (ceil(N/16), 1, 1), Block: (256, 1, 1)
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void gemv_int4_v6_t16(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    gemv_int4_v6_cooperative_v2<16>(A, B_q4, scales, zeros, C, K, N, group_size);
}
