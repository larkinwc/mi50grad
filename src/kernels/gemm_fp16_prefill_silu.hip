/**
 * FP16 GEMM for prefill projections with fused SiLU epilogue — gfx906.
 *
 * Strategy B: epilogue fusion only.
 *
 * This kernel computes:
 *   up = A @ B^T   (GEMM, same as gemm_fp16_prefill)
 *
 * In the epilogue (instead of writing up to HBM), it reads the pre-computed
 * gate projection from gate_buf, applies SiLU activation and element-wise
 * multiply, then writes the result to Out:
 *
 *   Out[i,j] = silu(gate_buf[i,j]) * up[i,j]
 *   where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * **Current prefill FFN flow (3 launches):**
 *   1. gemm_fp16_prefill: gate[M,N] = A @ W_gate^T   -> HBM
 *   2. gemm_fp16_prefill: up[M,N]   = A @ W_up^T     -> HBM
 *   3. silu_fused_v2:     out[M,N]  = silu(gate)*up  -> HBM  (reads gate+up)
 *
 * **Fused flow (2 launches):**
 *   1. gemm_fp16_prefill:           gate[M,N] = A @ W_gate^T    -> HBM
 *   2. gemm_fp16_prefill_silu_epilogue:
 *      out[M,N] = silu(gate[M,N]) * (A @ W_up^T)
 *      (reads gate from HBM but never writes up intermediate)
 *
 * Memory traffic saved per FFN step (M=128, N=11008):
 *   Old:  write gate(2MB) + write up(2MB) + write out(2MB) + read gate+up(4MB) = 10MB
 *   New:  write gate(2MB) + write out(2MB) + read gate(2MB) = 6MB
 *   Saving: ~4MB = 2*M*N*2 bytes per FFN call.
 *
 * Kernel parameters:
 *   A        [M, K] FP16 row-major
 *   B        [N, K] FP16 row-major (W_up^T)
 *   gate_buf [M, N] FP16 row-major (pre-computed gate projection)
 *   Out      [M, N] FP16 row-major (output = silu(gate)*up)
 *   M, N, K  dimensions
 *
 * Tile: 64x64 output per workgroup, K-tiles of 16.
 * Thread block: 256 threads (16x16), each computes 4x4 output elements.
 * LDS: A[64x18 FP16] + B[64x18 FP16] = 4608 bytes.
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

/**
 * gemm_fp16_prefill_silu_epilogue:
 *
 * Computes up = A @ B^T (FP16 GEMM), then applies fused SiLU epilogue:
 *   Out[i,j] = silu(gate_buf[i,j]) * up[i,j]
 *
 * This replaces the separate up GEMM + silu_fused_v2 launches when gate
 * has already been computed and stored in gate_buf.
 */
extern "C"
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void gemm_fp16_prefill_silu_epilogue(
    const __half* __restrict__ A,        // [M, K] row-major activations
    const __half* __restrict__ B,        // [N, K] row-major W_up weights
    const __half* __restrict__ gate_buf, // [M, N] row-major gate projection (pre-computed)
    __half* __restrict__ Out,            // [M, N] row-major output = silu(gate) * up
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

        // Load A (tid < 128) or B (tid >= 128) into shared memory
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

    // ============================================================
    // EPILOGUE: silu(gate_buf[row,col]) * up[row,col] -> Out[row,col]
    // ============================================================
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        const unsigned int out_row = block_m + a_rows[i];
        if (out_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            const unsigned int out_col = block_n + b_rows[j];
            if (out_col >= N) continue;

            // up value (computed by GEMM)
            float up_val = acc[i][j];

            // Read gate value from HBM gate_buf
            float gate_val = __half2float(gate_buf[out_row * N + out_col]);

            // SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
            float silu_gate = gate_val * (1.0f / (1.0f + __expf(-gate_val)));

            // Write fused result
            Out[out_row * N + out_col] = __float2half(silu_gate * up_val);
        }
    }
}
