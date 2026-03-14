/**
 * INT4 GEMM v2 for prefill FFN on gfx906 — on-the-fly dequantization.
 *
 * C[M, N] = A[M, K] @ dequant(B_q4[K/8, N]), with per-group FP16 scales/zeros.
 *
 * Key improvement over gemm_int4_prefill_hip.hip:
 * - B tile stored PACKED in LDS (uint32, 8 INT4 per word): 1024 bytes vs 4096 bytes (4x)
 * - Dequantization happens on-the-fly in the compute inner loop
 * - LDS bandwidth reduced 4x for B tile loads
 * - Per-group scales/zeros loaded once per group boundary crossing
 *
 * Strategy: Nibble extract + FP16 FMA
 * - For each k position in compute loop, read packed uint32 from LDS (shared among 8 k-positions)
 * - Use __builtin_amdgcn_ubfe to extract INT4 nibble
 * - Apply dequant: w = (q - zero) * scale
 * - Compute FMA with FP16 activation from A tile
 *
 * LDS layout:
 * - smem_A: [TILE_M, TILE_K] FP16 = 64*32*2 = 4096 bytes
 * - smem_Bp: [TILE_N, TILE_K/8] uint32 = 64*4*4 = 1024 bytes
 * Total LDS: 5120 bytes (vs 8192 bytes in original, 36% reduction)
 *
 * Grid: (ceil(N/64), ceil(M/64), 1), Block: (256, 1, 1)
 *
 * Requires: TILE_K divisible by 8 (group nibbles per uint32)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define THREADS 256
#define THREAD_M 4
#define THREAD_N 4
#define TILE_K_PACKED (TILE_K / 8)  // = 4 uint32 per column

extern "C"
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void gemm_int4_prefill_v2(
    const __half* __restrict__ A,           // [M, K] FP16 row-major
    const unsigned int* __restrict__ B_q4,  // [K/8, N] packed INT4 (uint32)
    const __half* __restrict__ scales,      // [K/group_size, N] FP16
    const __half* __restrict__ zeros,       // [K/group_size, N] FP16
    __half* __restrict__ C,                 // [M, N] FP16
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int group_size
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tx = tid % 16;   // [0..15]: output column group
    const unsigned int ty = tid / 16;   // [0..15]: output row group

    const unsigned int block_n = blockIdx.x * TILE_N;  // N-tile start
    const unsigned int block_m = blockIdx.y * TILE_M;  // M-tile start

    // LDS: A tile (FP16) + B tile packed (uint32)
    __shared__ __half smem_A[TILE_M * TILE_K];                   // [64, 32]
    __shared__ unsigned int smem_Bp[TILE_N * TILE_K_PACKED];     // [64, 4] packed

    // FP32 accumulators: 4×4 per thread
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    // Precompute output indices for this thread
    const unsigned int out_rows[THREAD_M] = {
        ty * THREAD_M + 0, ty * THREAD_M + 1,
        ty * THREAD_M + 2, ty * THREAD_M + 3
    };
    const unsigned int out_cols[THREAD_N] = {
        tx * THREAD_N + 0, tx * THREAD_N + 1,
        tx * THREAD_N + 2, tx * THREAD_N + 3
    };

    // Global N indices for this thread's output columns (for scale/zero lookup)
    const unsigned int n_idx[THREAD_N] = {
        block_n + tx * THREAD_N + 0,
        block_n + tx * THREAD_N + 1,
        block_n + tx * THREAD_N + 2,
        block_n + tx * THREAD_N + 3
    };

    // ---- Main K-tile loop ----
    for (unsigned int k_tile = 0; k_tile < K; k_tile += TILE_K) {

        // ---- Load A tile: [TILE_M, TILE_K] FP16 ----
        // 256 threads × 8 elements = 2048 = TILE_M × TILE_K
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned int elem = tid + i * THREADS;
            unsigned int row = elem / TILE_K;
            unsigned int col = elem % TILE_K;
            unsigned int g_row = block_m + row;
            unsigned int g_col = k_tile + col;
            smem_A[row * TILE_K + col] = (g_row < M && g_col < K)
                ? A[g_row * K + g_col]
                : __float2half(0.0f);
        }

        // ---- Load B tile PACKED: [TILE_N, TILE_K_PACKED] uint32 ----
        // We need K/8 packed words per column (N column).
        // B_q4 layout: [K/8, N] — row k8, column n: B_q4[k8 * N + n]
        // We want: for each (b_col, b_k8) in [TILE_N, TILE_K_PACKED]:
        //   smem_Bp[b_col * TILE_K_PACKED + b_k8] = B_q4[(k_tile/8 + b_k8) * N + (block_n + b_col)]
        //
        // Total elements: 64 * 4 = 256, one per thread (load exactly THREADS = 256 elements).
        {
            // Each thread loads 1 uint32: elem = tid (0..255)
            unsigned int b_col = tid / TILE_K_PACKED;    // [0..63]: N offset in tile
            unsigned int b_k8  = tid % TILE_K_PACKED;    // [0..3]: which packed word in K tile

            unsigned int n_global = block_n + b_col;
            unsigned int k8_global = k_tile / 8 + b_k8;
            unsigned int val = (n_global < N && k8_global < (K / 8))
                ? B_q4[k8_global * N + n_global]
                : 0u;
            smem_Bp[b_col * TILE_K_PACKED + b_k8] = val;
        }

        __syncthreads();

        // ---- Compute: inner K loop with on-the-fly dequantization ----
        // For each k in [0, TILE_K):
        //   - k8 = k/8: which packed word (into smem_Bp)
        //   - bit = (k%8)*4: nibble position within the word
        //   - scale/zero: determined by global k index = k_tile + k, group = (k_tile+k)/group_size

        // Preload scale and zero for each output column (j = 0..3).
        // We cache them for consecutive k positions that share the same group.
        float cur_scale[THREAD_N], cur_zero[THREAD_N];
        unsigned int cur_group[THREAD_N];
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            cur_group[j] = 0xFFFFFFFF;  // force load on first iteration
            cur_scale[j] = 0.0f;
            cur_zero[j]  = 0.0f;
        }

        // Preload packed B words for each output column's group
        // Each output column (j) maps to smem_Bp[(tx*4+j), :] = 4 packed words
        // Each output row (i) maps to smem_A[ty*4+i, k]

        // Inner loop: iterate over k positions within the tile
        #pragma unroll 4
        for (unsigned int k = 0; k < TILE_K; k++) {
            unsigned int k_global = k_tile + k;
            unsigned int g = k_global / group_size;

            // Load A values for all output rows
            float a_val[THREAD_M];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                a_val[i] = __half2float(smem_A[out_rows[i] * TILE_K + k]);
            }

            // For each output column, dequantize and accumulate
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                // Update scale/zero if group changed
                if (g != cur_group[j]) {
                    cur_group[j] = g;
                    if (n_idx[j] < N) {
                        cur_scale[j] = __half2float(scales[g * N + n_idx[j]]);
                        cur_zero[j]  = __half2float(zeros[g * N + n_idx[j]]);
                    } else {
                        cur_scale[j] = 0.0f;
                        cur_zero[j]  = 0.0f;
                    }
                }

                // Read packed INT4 from LDS and extract nibble
                // smem_Bp[(tx*4+j) * TILE_K_PACKED + k/8]
                unsigned int col_idx = out_cols[j];  // column in tile (0..63)
                unsigned int k8 = k / 8;
                unsigned int packed = smem_Bp[col_idx * TILE_K_PACKED + k8];
                unsigned int nibble = __builtin_amdgcn_ubfe(packed, (k % 8) * 4, 4);

                // Dequantize: w = (q - zero) * scale
                float w = ((float)nibble - cur_zero[j]) * cur_scale[j];

                // Accumulate: acc[i][j] += a[i] * w
                #pragma unroll
                for (int i = 0; i < THREAD_M; i++) {
                    acc[i][j] += a_val[i] * w;
                }
            }
        }

        __syncthreads();
    }

    // ---- Write output ----
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        unsigned int out_row = block_m + out_rows[i];
        if (out_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            unsigned int out_col = n_idx[j];
            if (out_col >= N) continue;
            C[out_row * N + out_col] = __float2half(acc[i][j]);
        }
    }
}
