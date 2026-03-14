/**
 * FP16 GEMM for prefill projections on gfx906 — v2 (dot2 + XOR-swizzled LDS).
 *
 * C[M, N] = A[M, K] @ B^T, where B is [N, K] (row-major weight matrix).
 *
 * Key optimizations vs v1 (original HIP scalar FMA):
 * 1. v_dot2_f32_f16 via __builtin_amdgcn_fdot2: 2 FMAs per instruction.
 * 2. XOR-swizzled LDS tile layout: phys_addr = (row * LDS_STRIDE) + (col ^ (row % 32))
 *    Applied at float4-group (8-half) granularity (NGRP=2, VEC=8):
 *      phys_group = logical_group ^ (row & 1)
 *    LDS stride-18 (36 bytes/row) provides bank separation.
 * 3. Register preloading: A rows and B cols preloaded before fdot2 outer product.
 * 4. Precomputed physical group addresses minimize address overhead.
 * 5. Separate A/B loading wavefronts (threads 0-127 → A, 128-255 → B).
 * 6. Vectorized float4 global loads.
 * 7. amdgpu_waves_per_eu hint to encourage higher occupancy.
 *
 * Tile: 64×64 output per workgroup, K-tiles of 16.
 * Thread block: 256 threads (16×16), each computes 4×4 output elements.
 * LDS: A [64×18 FP16] + B [64×18 FP16] = 4608 bytes.
 * Accumulation in FP32, output as FP16.
 *
 * Grid: (ceil(N/64), ceil(M/64), 1), Block: (256, 1, 1)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREADS 256
#define THREAD_M 4
#define THREAD_N 4

// LDS row stride: 16 data + 2 padding = 18 halfs = 36 bytes
#define LDS_STRIDE 18
// Float4 group (8 halfs = 16 bytes)
#define VEC 8
#define NGRP (TILE_K / VEC)  // = 2

// XOR swizzle: phys_group = log_group ^ (row & (NGRP-1)) = log_group ^ (row & 1)
#define SWIZZLE_GRP(g, r) ((g) ^ ((r) & (NGRP - 1u)))

extern "C"
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void gemm_fp16_prefill(
    const __half* __restrict__ A,    // [M, K] row-major
    const __half* __restrict__ B,    // [N, K] row-major
    __half* __restrict__ C,          // [M, N] row-major
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tx = tid % 16;
    const unsigned int ty = tid / 16;

    const unsigned int block_n = blockIdx.x * TILE_N;
    const unsigned int block_m = blockIdx.y * TILE_M;

    __shared__ __half smem_A[TILE_M * LDS_STRIDE];
    __shared__ __half smem_B[TILE_N * LDS_STRIDE];

    float acc[THREAD_M][THREAD_N] = {};

    // Precompute row indices (constant across K iterations)
    const unsigned int a_rows[THREAD_M] = {
        ty * THREAD_M + 0, ty * THREAD_M + 1,
        ty * THREAD_M + 2, ty * THREAD_M + 3
    };
    const unsigned int b_rows[THREAD_N] = {
        tx * THREAD_N + 0, tx * THREAD_N + 1,
        tx * THREAD_N + 2, tx * THREAD_N + 3
    };

    // Precompute physical group base offsets for each row
    unsigned int a_grp_base[THREAD_M][NGRP];
    unsigned int b_grp_base[THREAD_N][NGRP];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int g = 0; g < NGRP; g++)
            a_grp_base[i][g] = SWIZZLE_GRP(g, a_rows[i]) * VEC;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        #pragma unroll
        for (int g = 0; g < NGRP; g++)
            b_grp_base[j][g] = SWIZZLE_GRP(g, b_rows[j]) * VEC;

    // ---- Main K-loop ----
    for (unsigned int k_tile = 0; k_tile < K; k_tile += TILE_K) {

        // Load A (tid < 128) or B (tid >= 128)
        if (tid < 128) {
            const unsigned int a_row = tid >> 1;
            const unsigned int log_g = tid & 1;
            const unsigned int phys_col = SWIZZLE_GRP(log_g, a_row) * VEC;
            const unsigned int g_row = block_m + a_row;
            const unsigned int g_col = k_tile + log_g * VEC;
            __half vals[VEC];
            if (g_row < M && g_col + VEC - 1 < K) {
                *reinterpret_cast<float4*>(vals) =
                    *reinterpret_cast<const float4*>(A + g_row * K + g_col);
            } else {
                #pragma unroll
                for (int c = 0; c < VEC; c++)
                    vals[c] = (g_row < M && g_col + c < K)
                              ? A[g_row * K + g_col + c] : __float2half(0.0f);
            }
            *reinterpret_cast<float4*>(&smem_A[a_row * LDS_STRIDE + phys_col]) =
                *reinterpret_cast<const float4*>(vals);
        } else {
            const unsigned int local = tid - 128;
            const unsigned int b_row = local >> 1;
            const unsigned int log_g = local & 1;
            const unsigned int phys_col = SWIZZLE_GRP(log_g, b_row) * VEC;
            const unsigned int g_row = block_n + b_row;
            const unsigned int g_col = k_tile + log_g * VEC;
            __half vals[VEC];
            if (g_row < N && g_col + VEC - 1 < K) {
                *reinterpret_cast<float4*>(vals) =
                    *reinterpret_cast<const float4*>(B + g_row * K + g_col);
            } else {
                #pragma unroll
                for (int c = 0; c < VEC; c++)
                    vals[c] = (g_row < N && g_col + c < K)
                              ? B[g_row * K + g_col + c] : __float2half(0.0f);
            }
            *reinterpret_cast<float4*>(&smem_B[b_row * LDS_STRIDE + phys_col]) =
                *reinterpret_cast<const float4*>(vals);
        }

        __syncthreads();

        // ============================================================
        // COMPUTE: register-preloaded outer product with fdot2
        // For each kk pair, preload A[4] and B[4], then compute 4×4.
        // ============================================================
        #pragma unroll
        for (unsigned int kk = 0; kk < TILE_K; kk += 2) {
            const unsigned int g    = kk / VEC;   // compile-time: 0 or 1
            const unsigned int in_g = kk % VEC;   // compile-time: 0,2,4,6

            __half2 av[THREAD_M];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                const unsigned int phys = a_grp_base[i][g] + in_g;
                av[i] = *reinterpret_cast<const __half2*>(&smem_A[a_rows[i] * LDS_STRIDE + phys]);
            }

            __half2 bv[THREAD_N];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                const unsigned int phys = b_grp_base[j][g] + in_g;
                bv[j] = *reinterpret_cast<const __half2*>(&smem_B[b_rows[j] * LDS_STRIDE + phys]);
            }

            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    acc[i][j] = __builtin_amdgcn_fdot2(
                        *reinterpret_cast<const _Float16_2*>(&av[i]),
                        *reinterpret_cast<const _Float16_2*>(&bv[j]),
                        acc[i][j], false);
                }
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        const unsigned int out_row = block_m + a_rows[i];
        if (out_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            const unsigned int out_col = block_n + b_rows[j];
            if (out_col >= N) continue;
            C[out_row * N + out_col] = __float2half(acc[i][j]);
        }
    }
}

/**
 * FP16 GEMM for prefill projections — double-buffered K-loop (db variant).
 *
 * Same computation as gemm_fp16_prefill, but uses two LDS buffer sets
 * (ping: smem_A0/smem_B0, pong: smem_A1/smem_B1) to overlap the global
 * load of tile t+1 with the compute of tile t.
 *
 * Iteration flow:
 *   1. Load tile 0 into ping buffer
 *   2. __syncthreads()
 *   3. For t = 0 to num_k_tiles-1:
 *      a. Start loading tile t+1 into pong buffer (if t+1 < num_tiles)
 *      b. Compute on ping buffer (v_dot2 outer product)
 *      c. __syncthreads()
 *      d. Swap ping/pong pointers
 *
 * LDS budget: 2 * (64*18 + 64*18) * 2 = 9216 bytes (< 32KB limit).
 * All other parameters (tile sizes, thread layout, swizzle) unchanged.
 *
 * Grid: (ceil(N/64), ceil(M/64), 1), Block: (256, 1, 1)
 */
extern "C"
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void gemm_fp16_prefill_db(
    const __half* __restrict__ A,    // [M, K] row-major
    const __half* __restrict__ B,    // [N, K] row-major
    __half* __restrict__ C,          // [M, N] row-major
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tx = tid % 16;
    const unsigned int ty = tid / 16;

    const unsigned int block_n = blockIdx.x * TILE_N;
    const unsigned int block_m = blockIdx.y * TILE_M;

    // Double-buffered LDS: ping (0) and pong (1)
    __shared__ __half smem_A0[TILE_M * LDS_STRIDE];
    __shared__ __half smem_B0[TILE_N * LDS_STRIDE];
    __shared__ __half smem_A1[TILE_M * LDS_STRIDE];
    __shared__ __half smem_B1[TILE_N * LDS_STRIDE];

    float acc[THREAD_M][THREAD_N] = {};

    // Precompute row indices (constant across K iterations)
    const unsigned int a_rows[THREAD_M] = {
        ty * THREAD_M + 0, ty * THREAD_M + 1,
        ty * THREAD_M + 2, ty * THREAD_M + 3
    };
    const unsigned int b_rows[THREAD_N] = {
        tx * THREAD_N + 0, tx * THREAD_N + 1,
        tx * THREAD_N + 2, tx * THREAD_N + 3
    };

    // Precompute physical group base offsets for each row
    unsigned int a_grp_base[THREAD_M][NGRP];
    unsigned int b_grp_base[THREAD_N][NGRP];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int g = 0; g < NGRP; g++)
            a_grp_base[i][g] = SWIZZLE_GRP(g, a_rows[i]) * VEC;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        #pragma unroll
        for (int g = 0; g < NGRP; g++)
            b_grp_base[j][g] = SWIZZLE_GRP(g, b_rows[j]) * VEC;

    const unsigned int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Helper lambda-style inline load into a given LDS buffer pair
    // We use a macro to keep the load logic DRY for ping/pong variants.
    #define LOAD_TILE(smA, smB, k_tile_idx)                                        \
    do {                                                                            \
        const unsigned int _kt = (k_tile_idx) * TILE_K;                            \
        if (tid < 128) {                                                            \
            const unsigned int a_row  = tid >> 1;                                  \
            const unsigned int log_g  = tid & 1;                                   \
            const unsigned int phys_c = SWIZZLE_GRP(log_g, a_row) * VEC;          \
            const unsigned int g_row  = block_m + a_row;                           \
            const unsigned int g_col  = _kt + log_g * VEC;                        \
            __half vals[VEC];                                                       \
            if (g_row < M && g_col + VEC - 1 < K) {                               \
                *reinterpret_cast<float4*>(vals) =                                  \
                    *reinterpret_cast<const float4*>(A + g_row * K + g_col);       \
            } else {                                                                \
                _Pragma("unroll")                                                   \
                for (int _c = 0; _c < VEC; _c++)                                   \
                    vals[_c] = (g_row < M && g_col + _c < K)                       \
                               ? A[g_row * K + g_col + _c] : __float2half(0.0f);  \
            }                                                                       \
            *reinterpret_cast<float4*>(&(smA)[a_row * LDS_STRIDE + phys_c]) =     \
                *reinterpret_cast<const float4*>(vals);                             \
        } else {                                                                    \
            const unsigned int local  = tid - 128;                                  \
            const unsigned int b_row  = local >> 1;                                \
            const unsigned int log_g  = local & 1;                                 \
            const unsigned int phys_c = SWIZZLE_GRP(log_g, b_row) * VEC;          \
            const unsigned int g_row  = block_n + b_row;                           \
            const unsigned int g_col  = _kt + log_g * VEC;                        \
            __half vals[VEC];                                                       \
            if (g_row < N && g_col + VEC - 1 < K) {                               \
                *reinterpret_cast<float4*>(vals) =                                  \
                    *reinterpret_cast<const float4*>(B + g_row * K + g_col);       \
            } else {                                                                \
                _Pragma("unroll")                                                   \
                for (int _c = 0; _c < VEC; _c++)                                   \
                    vals[_c] = (g_row < N && g_col + _c < K)                       \
                               ? B[g_row * K + g_col + _c] : __float2half(0.0f);  \
            }                                                                       \
            *reinterpret_cast<float4*>(&(smB)[b_row * LDS_STRIDE + phys_c]) =     \
                *reinterpret_cast<const float4*>(vals);                             \
        }                                                                           \
    } while (0)

    #define COMPUTE_TILE(smA, smB)                                                 \
    do {                                                                            \
        _Pragma("unroll")                                                           \
        for (unsigned int kk = 0; kk < TILE_K; kk += 2) {                         \
            const unsigned int g    = kk / VEC;                                    \
            const unsigned int in_g = kk % VEC;                                    \
            __half2 av[THREAD_M];                                                   \
            _Pragma("unroll")                                                       \
            for (int i = 0; i < THREAD_M; i++) {                                   \
                const unsigned int phys = a_grp_base[i][g] + in_g;                \
                av[i] = *reinterpret_cast<const __half2*>(                          \
                    &(smA)[a_rows[i] * LDS_STRIDE + phys]);                        \
            }                                                                       \
            __half2 bv[THREAD_N];                                                   \
            _Pragma("unroll")                                                       \
            for (int j = 0; j < THREAD_N; j++) {                                   \
                const unsigned int phys = b_grp_base[j][g] + in_g;                \
                bv[j] = *reinterpret_cast<const __half2*>(                          \
                    &(smB)[b_rows[j] * LDS_STRIDE + phys]);                        \
            }                                                                       \
            _Pragma("unroll")                                                       \
            for (int i = 0; i < THREAD_M; i++) {                                   \
                _Pragma("unroll")                                                   \
                for (int j = 0; j < THREAD_N; j++) {                               \
                    acc[i][j] = __builtin_amdgcn_fdot2(                             \
                        *reinterpret_cast<const _Float16_2*>(&av[i]),               \
                        *reinterpret_cast<const _Float16_2*>(&bv[j]),               \
                        acc[i][j], false);                                          \
                }                                                                   \
            }                                                                       \
        }                                                                           \
    } while (0)

    // ---- Prologue: load tile 0 into ping (smem_A0/B0) ----
    LOAD_TILE(smem_A0, smem_B0, 0);
    __syncthreads();

    // ---- Main double-buffered K-loop ----
    // ping = buf 0, pong = buf 1; we alternate each iteration.
    for (unsigned int t = 0; t < num_k_tiles; t++) {

        // Determine which buffer is currently "ping" (current) and "pong" (next load)
        const bool use_buf0 = ((t & 1) == 0);

        // Start loading next tile into the pong buffer (if it exists)
        if (t + 1 < num_k_tiles) {
            if (use_buf0) {
                LOAD_TILE(smem_A1, smem_B1, t + 1);
            } else {
                LOAD_TILE(smem_A0, smem_B0, t + 1);
            }
        }

        // Compute on the ping buffer (current tile, already loaded and synced)
        if (use_buf0) {
            COMPUTE_TILE(smem_A0, smem_B0);
        } else {
            COMPUTE_TILE(smem_A1, smem_B1);
        }

        // Sync after compute (and after the async pong load above)
        // This ensures pong buffer is fully written before next iteration uses it.
        __syncthreads();
    }

    #undef LOAD_TILE
    #undef COMPUTE_TILE

    // Write output
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        const unsigned int out_row = block_m + a_rows[i];
        if (out_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            const unsigned int out_col = block_n + b_rows[j];
            if (out_col >= N) continue;
            C[out_row * N + out_col] = __float2half(acc[i][j]);
        }
    }
}
