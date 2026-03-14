/**
 * FP16 GEMM dual-output (gate+up) for prefill FFN with fused SiLU — gfx906.
 *
 * Strategy A: Single kernel computes BOTH gate and up projections from the
 * same input A, fusing the SiLU activation in the epilogue.
 *
 * **Fused flow (1 launch):**
 *   gemm_fp16_prefill_silu_dual(A, W_gate, W_up, Out, M, N, K):
 *     gate[tile] = A[tile] @ W_gate^T[tile]   (computed in registers)
 *     up[tile]   = A[tile] @ W_up^T[tile]     (computed in registers)
 *     Out[tile]  = silu(gate[tile]) * up[tile] (applied before write)
 *
 * **Why this beats Strategy B (epilogue-only):**
 * Strategy B (separate gate GEMM + up GEMM-with-SiLU) still requires two
 * full GEMM launches. Each GEMM loads A[M,K] from HBM. For M=128, K=5120
 * that's 1.3MB loaded twice unnecessarily. This kernel loads A once.
 *
 * **Traffic analysis (M=128, N=11008, K=5120):**
 *   Baseline (3 launches): 2*(A+W_gate+W_up reads) + gate+up+out writes + gate+up reads
 *     = 2*(1.3MB + 112.6MB + 112.6MB) + 3*2.8MB + 2*2.8MB = 458.6MB total
 *   Dual (1 launch): A + W_gate + W_up reads + out write
 *     = 1.3MB + 112.6MB + 112.6MB + 2.8MB = 229.3MB total
 *   Saving: ~229MB (~50% reduction in HBM traffic)
 *
 * **Tile design (TILE_M=64, TILE_N=32 per WG):**
 *   - A tile: 64×16 in LDS (same as gemm_fp16_prefill)
 *   - B_gate tile: 32×16 in LDS (half-width)
 *   - B_up tile:   32×16 in LDS (half-width)
 *   - LDS: A[64×18] + B_gate[32×18] + B_up[32×18] = 2304 + 1152 + 1152 = 4608 bytes
 *   - Accumulators: gate[4×2] + up[4×2] = 16 FP32 (same as original 4×4=16)
 *   - THREAD_M=4, THREAD_N=2 (halved in N due to 32-wide tile)
 *   - Grid: (ceil(N/32), ceil(M/64), 1) — 2x more WGs in N dimension vs original
 *
 * Kernel parameters:
 *   A      [M, K] FP16 row-major
 *   B_gate [N, K] FP16 row-major W_gate weights
 *   B_up   [N, K] FP16 row-major W_up weights
 *   Out    [M, N] FP16 row-major output = silu(gate) * up
 *   M, N, K  dimensions
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define DUAL_TILE_M  64
#define DUAL_TILE_N  32    // Half of original TILE_N=64
#define DUAL_TILE_K  16
#define DUAL_THREADS 256
#define DUAL_THREAD_M 4
#define DUAL_THREAD_N 2   // Half of original THREAD_N=4

// LDS row stride: 16 data + 2 padding = 18 halfs
#define DUAL_LDS_STRIDE 18
#define DUAL_VEC 8
#define DUAL_NGRP (DUAL_TILE_K / DUAL_VEC)  // = 2

// XOR swizzle: phys_group = log_group ^ (row & 1)
#define DUAL_SWIZZLE_GRP(g, r) ((g) ^ ((r) & (DUAL_NGRP - 1u)))

extern "C"
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void gemm_fp16_prefill_silu_dual(
    const __half* __restrict__ A,       // [M, K] row-major
    const __half* __restrict__ B_gate,  // [N, K] row-major W_gate
    const __half* __restrict__ B_up,    // [N, K] row-major W_up
    __half* __restrict__ Out,           // [M, N] row-major output = silu(gate)*up
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int tid = threadIdx.x;
    // Thread layout: 256 threads = 128 rows × 2 cols
    // Each thread computes DUAL_THREAD_M × DUAL_THREAD_N = 4×2 outputs
    // tx: column index within DUAL_TILE_N/DUAL_THREAD_N = 32/2 = 16 cols
    // ty: row index within DUAL_TILE_M/DUAL_THREAD_M = 64/4 = 16 rows
    const unsigned int tx = tid % 16;   // 0..15
    const unsigned int ty = tid / 16;   // 0..15

    const unsigned int block_n = blockIdx.x * DUAL_TILE_N;
    const unsigned int block_m = blockIdx.y * DUAL_TILE_M;

    // LDS: A tile + B_gate tile + B_up tile
    // A: 64 rows × 18 stride = 1152 halfs
    // B_gate: 32 rows × 18 stride = 576 halfs
    // B_up:   32 rows × 18 stride = 576 halfs
    __shared__ __half smem_A[DUAL_TILE_M * DUAL_LDS_STRIDE];
    __shared__ __half smem_Bg[DUAL_TILE_N * DUAL_LDS_STRIDE];
    __shared__ __half smem_Bu[DUAL_TILE_N * DUAL_LDS_STRIDE];

    // Dual FP32 accumulators: gate and up outputs
    float acc_gate[DUAL_THREAD_M][DUAL_THREAD_N] = {};
    float acc_up[DUAL_THREAD_M][DUAL_THREAD_N]   = {};

    // Precompute row indices
    const unsigned int a_rows[DUAL_THREAD_M] = {
        ty * DUAL_THREAD_M + 0, ty * DUAL_THREAD_M + 1,
        ty * DUAL_THREAD_M + 2, ty * DUAL_THREAD_M + 3
    };
    const unsigned int b_rows[DUAL_THREAD_N] = {
        tx * DUAL_THREAD_N + 0, tx * DUAL_THREAD_N + 1
    };

    // Precompute physical group base offsets
    unsigned int a_grp_base[DUAL_THREAD_M][DUAL_NGRP];
    unsigned int b_grp_base[DUAL_THREAD_N][DUAL_NGRP];
    #pragma unroll
    for (int i = 0; i < DUAL_THREAD_M; i++)
        #pragma unroll
        for (int g = 0; g < DUAL_NGRP; g++)
            a_grp_base[i][g] = DUAL_SWIZZLE_GRP(g, a_rows[i]) * DUAL_VEC;
    #pragma unroll
    for (int j = 0; j < DUAL_THREAD_N; j++)
        #pragma unroll
        for (int g = 0; g < DUAL_NGRP; g++)
            b_grp_base[j][g] = DUAL_SWIZZLE_GRP(g, b_rows[j]) * DUAL_VEC;

    // ---- Main K-loop ----
    for (unsigned int k_tile = 0; k_tile < K; k_tile += DUAL_TILE_K) {

        // Load A tile (tid < 128), B_gate tile (128 <= tid < 192), B_up tile (192 <= tid < 256)
        if (tid < 128) {
            // Load A: 64 rows × 16 cols (128 threads load 1 float4 each)
            const unsigned int a_row = tid >> 1;
            const unsigned int log_g = tid & 1;
            const unsigned int phys_col = DUAL_SWIZZLE_GRP(log_g, a_row) * DUAL_VEC;
            const unsigned int g_row = block_m + a_row;
            const unsigned int g_col = k_tile + log_g * DUAL_VEC;
            __half vals[DUAL_VEC];
            if (g_row < M && g_col + DUAL_VEC - 1 < K) {
                *reinterpret_cast<float4*>(vals) =
                    *reinterpret_cast<const float4*>(A + g_row * K + g_col);
            } else {
                #pragma unroll
                for (int c = 0; c < DUAL_VEC; c++)
                    vals[c] = (g_row < M && g_col + c < K)
                              ? A[g_row * K + g_col + c] : __float2half(0.0f);
            }
            *reinterpret_cast<float4*>(&smem_A[a_row * DUAL_LDS_STRIDE + phys_col]) =
                *reinterpret_cast<const float4*>(vals);
        } else if (tid < 192) {
            // Load B_gate: 32 rows × 16 cols (64 threads load 1 float4 each)
            const unsigned int local = tid - 128;
            const unsigned int b_row = local >> 1;
            const unsigned int log_g = local & 1;
            const unsigned int phys_col = DUAL_SWIZZLE_GRP(log_g, b_row) * DUAL_VEC;
            const unsigned int g_row = block_n + b_row;
            const unsigned int g_col = k_tile + log_g * DUAL_VEC;
            __half vals[DUAL_VEC];
            if (g_row < N && g_col + DUAL_VEC - 1 < K) {
                *reinterpret_cast<float4*>(vals) =
                    *reinterpret_cast<const float4*>(B_gate + g_row * K + g_col);
            } else {
                #pragma unroll
                for (int c = 0; c < DUAL_VEC; c++)
                    vals[c] = (g_row < N && g_col + c < K)
                              ? B_gate[g_row * K + g_col + c] : __float2half(0.0f);
            }
            *reinterpret_cast<float4*>(&smem_Bg[b_row * DUAL_LDS_STRIDE + phys_col]) =
                *reinterpret_cast<const float4*>(vals);
        } else {
            // Load B_up: 32 rows × 16 cols (64 threads load 1 float4 each)
            const unsigned int local = tid - 192;
            const unsigned int b_row = local >> 1;
            const unsigned int log_g = local & 1;
            const unsigned int phys_col = DUAL_SWIZZLE_GRP(log_g, b_row) * DUAL_VEC;
            const unsigned int g_row = block_n + b_row;
            const unsigned int g_col = k_tile + log_g * DUAL_VEC;
            __half vals[DUAL_VEC];
            if (g_row < N && g_col + DUAL_VEC - 1 < K) {
                *reinterpret_cast<float4*>(vals) =
                    *reinterpret_cast<const float4*>(B_up + g_row * K + g_col);
            } else {
                #pragma unroll
                for (int c = 0; c < DUAL_VEC; c++)
                    vals[c] = (g_row < N && g_col + c < K)
                              ? B_up[g_row * K + g_col + c] : __float2half(0.0f);
            }
            *reinterpret_cast<float4*>(&smem_Bu[b_row * DUAL_LDS_STRIDE + phys_col]) =
                *reinterpret_cast<const float4*>(vals);
        }

        __syncthreads();

        // ============================================================
        // COMPUTE: outer product for gate and up simultaneously
        // ============================================================
        #pragma unroll
        for (unsigned int kk = 0; kk < DUAL_TILE_K; kk += 2) {
            const unsigned int g    = kk / DUAL_VEC;    // 0 or 1
            const unsigned int in_g = kk % DUAL_VEC;

            // Load A values for 4 output rows
            __half2 av[DUAL_THREAD_M];
            #pragma unroll
            for (int i = 0; i < DUAL_THREAD_M; i++) {
                const unsigned int phys = a_grp_base[i][g] + in_g;
                av[i] = *reinterpret_cast<const __half2*>(
                    &smem_A[a_rows[i] * DUAL_LDS_STRIDE + phys]);
            }

            // Load B_gate values for 2 output cols (gate)
            __half2 bgv[DUAL_THREAD_N];
            #pragma unroll
            for (int j = 0; j < DUAL_THREAD_N; j++) {
                const unsigned int phys = b_grp_base[j][g] + in_g;
                bgv[j] = *reinterpret_cast<const __half2*>(
                    &smem_Bg[b_rows[j] * DUAL_LDS_STRIDE + phys]);
            }

            // Load B_up values for 2 output cols (up)
            __half2 buv[DUAL_THREAD_N];
            #pragma unroll
            for (int j = 0; j < DUAL_THREAD_N; j++) {
                const unsigned int phys = b_grp_base[j][g] + in_g;
                buv[j] = *reinterpret_cast<const __half2*>(
                    &smem_Bu[b_rows[j] * DUAL_LDS_STRIDE + phys]);
            }

            // Compute outer products for both gate and up
            #pragma unroll
            for (int i = 0; i < DUAL_THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < DUAL_THREAD_N; j++) {
                    acc_gate[i][j] = __builtin_amdgcn_fdot2(
                        *reinterpret_cast<const _Float16_2*>(&av[i]),
                        *reinterpret_cast<const _Float16_2*>(&bgv[j]),
                        acc_gate[i][j], false);
                    acc_up[i][j] = __builtin_amdgcn_fdot2(
                        *reinterpret_cast<const _Float16_2*>(&av[i]),
                        *reinterpret_cast<const _Float16_2*>(&buv[j]),
                        acc_up[i][j], false);
                }
            }
        }

        __syncthreads();
    }

    // ============================================================
    // EPILOGUE: silu(gate) * up -> Out (no intermediate HBM writes!)
    // ============================================================
    #pragma unroll
    for (int i = 0; i < DUAL_THREAD_M; i++) {
        const unsigned int out_row = block_m + a_rows[i];
        if (out_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < DUAL_THREAD_N; j++) {
            const unsigned int out_col = block_n + b_rows[j];
            if (out_col >= N) continue;

            float gate_val = acc_gate[i][j];
            float up_val   = acc_up[i][j];

            // SiLU(gate) = gate * sigmoid(gate)
            float silu_gate = gate_val * (1.0f / (1.0f + __expf(-gate_val)));

            Out[out_row * N + out_col] = __float2half(silu_gate * up_val);
        }
    }
}
