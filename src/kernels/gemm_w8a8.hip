/**
 * W8A8 tiled GEMM for prefill on gfx906.
 *
 * C[M, N] = scale_a * diag(scale_w) * (A_int8[M, K] @ B_int8^T[N, K])
 *
 * Where:
 *   A_int8  [M, K] - INT8 activations (per-tensor quantized, from activation_quant)
 *   B_int8  [N, K] - INT8 weights (per-channel quantized, row-major)
 *   scale_w [N]    - FP32 per-channel weight scales
 *   scale_a        - FP32 per-tensor activation scale (scalar)
 *   C       [M, N] - FP16 output
 *
 * Tiling strategy (following gemm_fp16_prefill.hip):
 *   Tile: 64x64 output per workgroup, K-tiles of 16 (each thread handles 16 INT8s = 16 bytes).
 *   Thread block: 256 threads (16x16), each computes 4x4 output elements.
 *   LDS: A [64x16 INT8] + B [64x16 INT8] = 2048 bytes total.
 *
 * Inner loop uses __builtin_amdgcn_sdot4 (v_dot4_i32_i8):
 *   4 INT8*INT8 products per instruction → INT32 accumulator.
 *   Each thread accumulates 4x4 = 16 INT32 partial sums.
 *
 * Epilogue: INT32 → FP32, multiply by (scale_w[col] * scale_a), → FP16 output.
 *
 * Grid:  (ceil(N/64), ceil(M/64), 1)
 * Block: (256, 1, 1)
 *
 * Representative shapes: 4096x4096x4096, 128x4096x4096
 * Target: correctness vs numpy reference (VAL-W8A8-002)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16   // 16 INT8s per row section loaded into LDS
#define THREADS 256
#define THREAD_M 4  // each thread computes 4 rows of A
#define THREAD_N 4  // each thread computes 4 cols of B (output rows)

// LDS layout: [64 rows x 16 cols INT8] per tile
// Row stride = TILE_K = 16 bytes (no padding needed for INT8 — 16 is divisible by 16)
// To avoid LDS bank conflicts with 4-byte access: stride 16 bytes = 4 int32s per row.
// With 32 banks of 4 bytes, 64 rows × 16 bytes:
//   row i, col j accesses bank (i*4 + j/4) % 32 — same 4-byte word occupies consecutive banks
//   This is conflict-free for the inner loop accessing 4-byte words.
#define LDS_A_ROWS TILE_M    // 64
#define LDS_A_COLS TILE_K    // 16
#define LDS_B_ROWS TILE_N    // 64
#define LDS_B_COLS TILE_K    // 16

extern "C"
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void gemm_w8a8(
    const signed char* __restrict__ A,    // [M, K] row-major INT8 activations
    const signed char* __restrict__ B,    // [N, K] row-major INT8 weights
    const float*       __restrict__ scale_w, // [N] per-channel weight scales
    float              scale_a,            // per-tensor activation scale
    __half*            __restrict__ C,     // [M, N] row-major FP16 output
    unsigned int       M,
    unsigned int       N,
    unsigned int       K
) {
    // Thread layout: 256 threads = 16x16 (tx = col thread, ty = row thread)
    const unsigned int tid = threadIdx.x;
    const unsigned int tx = tid % 16;  // column thread index (0..15)
    const unsigned int ty = tid / 16;  // row thread index (0..15)

    // Block-level tile origin
    const unsigned int block_n = blockIdx.x * TILE_N;  // output column tile
    const unsigned int block_m = blockIdx.y * TILE_M;  // output row tile

    // LDS storage: INT8 tiles for A (activation rows) and B (weight rows)
    __shared__ signed char smem_A[LDS_A_ROWS * LDS_A_COLS];  // 64 × 16 = 1024 bytes
    __shared__ signed char smem_B[LDS_B_ROWS * LDS_B_COLS];  // 64 × 16 = 1024 bytes

    // Each thread accumulates 4×4 INT32 output elements
    int acc[THREAD_M][THREAD_N] = {};

    // Thread's row/col assignments in the output tile
    // ty handles rows: [ty*4, ty*4+1, ty*4+2, ty*4+3] within the tile
    // tx handles cols: [tx*4, tx*4+1, tx*4+2, tx*4+3] within the tile
    const unsigned int a_rows[THREAD_M] = {
        ty * THREAD_M + 0, ty * THREAD_M + 1,
        ty * THREAD_M + 2, ty * THREAD_M + 3
    };
    const unsigned int b_rows[THREAD_N] = {
        tx * THREAD_N + 0, tx * THREAD_N + 1,
        tx * THREAD_N + 2, tx * THREAD_N + 3
    };

    // ---- Main K-loop: process TILE_K=16 INT8 elements per iteration ----
    for (unsigned int k_tile = 0; k_tile < K; k_tile += TILE_K) {

        // Load A tile: 64 rows × 16 cols = 1024 bytes
        // 256 threads load 4 bytes (int32) each: 256*4 = 1024 bytes ✓
        // Thread tid loads: row = tid/4, col_word = tid%4 (each word = 4 INT8s)
        {
            const unsigned int a_load_row = tid >> 2;   // tid / 4 → row in [0..63]
            const unsigned int a_load_col = (tid & 3) << 2;  // (tid % 4) * 4 → col in {0,4,8,12}
            const unsigned int g_row = block_m + a_load_row;
            const unsigned int g_col = k_tile + a_load_col;

            signed char vals[4];
            if (g_row < M && g_col + 3 < K) {
                // Vectorized 4-byte load
                *reinterpret_cast<int*>(vals) =
                    *reinterpret_cast<const int*>(A + (size_t)g_row * K + g_col);
            } else {
                for (int c = 0; c < 4; c++)
                    vals[c] = (g_row < M && g_col + c < K) ? A[(size_t)g_row * K + g_col + c] : 0;
            }
            *reinterpret_cast<int*>(&smem_A[a_load_row * LDS_A_COLS + a_load_col]) =
                *reinterpret_cast<const int*>(vals);
        }

        // Load B tile: 64 rows × 16 cols = 1024 bytes
        // Thread tid loads: row = tid/4, col_word = tid%4
        {
            const unsigned int b_load_row = tid >> 2;   // row in [0..63]
            const unsigned int b_load_col = (tid & 3) << 2;  // col in {0,4,8,12}
            const unsigned int g_row = block_n + b_load_row;
            const unsigned int g_col = k_tile + b_load_col;

            signed char vals[4];
            if (g_row < N && g_col + 3 < K) {
                *reinterpret_cast<int*>(vals) =
                    *reinterpret_cast<const int*>(B + (size_t)g_row * K + g_col);
            } else {
                for (int c = 0; c < 4; c++)
                    vals[c] = (g_row < N && g_col + c < K) ? B[(size_t)g_row * K + g_col + c] : 0;
            }
            *reinterpret_cast<int*>(&smem_B[b_load_row * LDS_B_COLS + b_load_col]) =
                *reinterpret_cast<const int*>(vals);
        }

        __syncthreads();

        // ============================================================
        // COMPUTE: 4×4 tile using v_dot4_i32_i8 (sdot4)
        //
        // Each sdot4 computes: acc += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
        // where a[i], b[i] are signed INT8 values packed into int32.
        //
        // We process TILE_K=16 elements per K-tile:
        //   4 groups of 4 INT8s = 4 sdot4 calls per (row, col) pair.
        //
        // For each (i, j) output pair, we load 4 packed int32 words from smem_A[row_i]
        // and 4 packed int32 words from smem_B[row_j], and call sdot4 4 times.
        // ============================================================
        #pragma unroll
        for (unsigned int kk = 0; kk < TILE_K; kk += 4) {
            // Load 4 INT8s from A for each of THREAD_M rows (preload for reuse)
            int av[THREAD_M];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                av[i] = *reinterpret_cast<const int*>(&smem_A[a_rows[i] * LDS_A_COLS + kk]);
            }

            // Load 4 INT8s from B for each of THREAD_N rows
            int bv[THREAD_N];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                bv[j] = *reinterpret_cast<const int*>(&smem_B[b_rows[j] * LDS_B_COLS + kk]);
            }

            // Outer product: THREAD_M × THREAD_N sdot4 calls
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    // __builtin_amdgcn_sdot4 → v_dot4_i32_i8
                    // acc += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]  (INT8 signed)
                    acc[i][j] = __builtin_amdgcn_sdot4(av[i], bv[j], acc[i][j], false);
                }
            }
        }

        __syncthreads();
    }

    // ============================================================
    // EPILOGUE: INT32 → FP32 scale → FP16 output
    //
    // For output element C[m, n]:
    //   C[m, n] = FP16(acc_int32[m-block_m][n-block_n] * scale_w[n] * scale_a)
    //
    // scale_w[n] is the per-channel weight scale for output column n.
    // scale_a is the per-tensor activation scale (shared across all rows/cols).
    // ============================================================
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        const unsigned int out_row = block_m + a_rows[i];
        if (out_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            const unsigned int out_col = block_n + b_rows[j];
            if (out_col >= N) continue;
            // Load per-channel scale for this output column
            float sw = scale_w[out_col];
            // Compute: INT32 → FP32 → scale → FP16
            float result = (float)acc[i][j] * sw * scale_a;
            C[(size_t)out_row * N + out_col] = __float2half(result);
        }
    }
}
