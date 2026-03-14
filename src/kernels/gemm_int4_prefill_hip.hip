/**
 * INT4 GEMM for prefill FFN on gfx906.
 *
 * C[M, N] = A[M, K] @ dequant(B_q4[K/8, N]), with per-group FP16 scales/zeros.
 *
 * Used for: FFN gate/up/down projections during prefill.
 * A = [seq_len, K] FP16 (input activations)
 * B_q4 = [K/8, N] uint32 (packed INT4 weights, 8 per uint32)
 * scales = [K/group_size, N] FP16
 * zeros = [K/group_size, N] FP16 (unpacked zero points)
 * C = [seq_len, N] FP16 (output)
 *
 * Tile: 64x64 output, K-tiles of 32 (aligned to group_size boundaries).
 * Grid: (ceil(N/64), ceil(M/64), 1), Block: (256, 1, 1)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define THREADS 256
#define THREAD_M 4
#define THREAD_N 4

extern "C" __global__ void gemm_int4_prefill_hip(
    const __half* __restrict__ A,       // [M, K]
    const unsigned int* __restrict__ B,  // [K/8, N] packed INT4
    const __half* __restrict__ scales,   // [K/group_size, N]
    const __half* __restrict__ zeros,      // [K/group_size, N] FP16
    __half* __restrict__ C,              // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int group_size
) {
    unsigned int tx = threadIdx.x % 16;
    unsigned int ty = threadIdx.x / 16;
    unsigned int block_n = blockIdx.x * TILE_N;
    unsigned int block_m = blockIdx.y * TILE_M;

    __shared__ __half smem_A[TILE_M * TILE_K];  // [64, 32]
    __shared__ __half smem_B[TILE_N * TILE_K];  // [64, 32] dequantized

    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    unsigned int k_packed = K / 8;  // Number of packed uint32 columns

    for (unsigned int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile: [64, 32]
        // 256 threads, 2048 elements, 8 per thread
        {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned int elem = threadIdx.x + i * THREADS;
                unsigned int row = elem / TILE_K;
                unsigned int col = elem % TILE_K;
                unsigned int g_row = block_m + row;
                unsigned int g_col = k_tile + col;
                __half val = (g_row < M && g_col < K)
                    ? A[g_row * K + g_col]
                    : __float2half(0.0f);
                smem_A[row * TILE_K + col] = val;
            }
        }

        // Load and dequantize B tile: [64, 32]
        // B is packed [K/8, N], we need rows k_tile..k_tile+31, cols block_n..block_n+63
        // Each uint32 in B has 8 INT4 values for 8 consecutive K positions
        {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned int elem = threadIdx.x + i * THREADS;
                unsigned int b_row = elem / TILE_K;  // Output column within tile (0-63)
                unsigned int b_col = elem % TILE_K;  // K offset within tile (0-31)
                unsigned int n_idx = block_n + b_row;  // Global N index
                unsigned int k_idx = k_tile + b_col;   // Global K index

                float val = 0.0f;
                if (n_idx < N && k_idx < K) {
                    // Which packed uint32 contains this K position?
                    unsigned int pack_row = k_idx / 8;
                    unsigned int pack_bit = (k_idx % 8) * 4;
                    unsigned int packed = B[pack_row * N + n_idx];
                    int q = (packed >> pack_bit) & 0xF;

                    // Get scale and zero for this group
                    unsigned int group = k_idx / group_size;
                    float scale = __half2float(scales[group * N + n_idx]);

                    float zero = __half2float(zeros[group * N + n_idx]);

                    val = ((float)q - zero) * scale;
                }
                smem_B[b_row * TILE_K + b_col] = __float2half(val);
            }
        }

        __syncthreads();

        // Compute: A_tile[64, 32] @ B_tile[64, 32]^T
        #pragma unroll
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            float a_val[THREAD_M];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                a_val[i] = __half2float(smem_A[(ty * THREAD_M + i) * TILE_K + kk]);

            float b_val[THREAD_N];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++)
                b_val[j] = __half2float(smem_B[(tx * THREAD_N + j) * TILE_K + kk]);

            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += a_val[i] * b_val[j];
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        unsigned int out_row = block_m + ty * THREAD_M + i;
        if (out_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            unsigned int out_col = block_n + tx * THREAD_N + j;
            if (out_col >= N) continue;
            C[out_row * N + out_col] = __float2half(acc[i][j]);
        }
    }
}
