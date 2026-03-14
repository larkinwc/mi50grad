/**
 * W8A8 GEMV for decode (M=1) on gfx906.
 *
 * Computes: y = diag(scale_w) * (W_int8 @ x_int8) * scale_a
 * where:
 *   W_int8  is [N, K] signed INT8 weights (per-channel scale in scale_w[N])
 *   x_int8  is [K] signed INT8 activations (per-tensor scale in scale_a)
 *   y       is [N] FP16 output
 *
 * Inner loop uses __builtin_amdgcn_sdot4 (v_dot4_i32_i8) to accumulate
 * 4 INT8×INT8 products into INT32 per instruction.
 *
 * Layout: 4 rows per workgroup, 64 threads (1 wavefront) per row.
 * Each thread in a wavefront computes dot product for K/64 groups of 4 elements.
 * Wavefront butterfly reduction gives final INT32 partial sums → FP32 epilogue.
 *
 * Epilogue: acc_int32 → acc_fp32 * (scale_w[row] * scale_a) → FP16
 *
 * Grid:  (ceil(N / 4), 1, 1)
 * Block: (256, 1, 1) — 4 wavefronts of 64 threads each
 *
 * Target dimensions: N=4096,K=4096 and N=11008,K=4096
 * Tolerance: relative error < 5% vs FP16 reference
 *
 * v_dot4_i32_i8 confirmed working on gfx906 (validated by CK library).
 * Intrinsic: __builtin_amdgcn_sdot4(int a_i8x4, int b_i8x4, int c, bool sat)
 *   → computes: c + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
 *   where a[i], b[i] are signed 8-bit extracted from the 32-bit packed inputs.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// Warp-level INT32 sum reduction via DPP (64 lanes on gfx906)
// ---------------------------------------------------------------------------
__device__ __forceinline__ int warp_reduce_sum_i32(int val) {
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// Main kernel: gemv_w8a8
//
// Parameters:
//   x        [K]    INT8 activations (from activation_quant)
//   W        [N, K] INT8 weights (per-channel quantized, row-major)
//   scale_w  [N]    FP32 per-channel weight scales
//   scale_a         FP32 per-tensor activation scale (scalar)
//   out      [N]    FP16 output
//   K, N            dimensions
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_w8a8(
    const signed char* __restrict__ x,       // [K] INT8 input activation
    const signed char* __restrict__ W,       // [N, K] INT8 weights
    const float*       __restrict__ scale_w, // [N] FP32 per-channel weight scales
    float              scale_a,              // FP32 per-tensor activation scale
    __half*            __restrict__ out,     // [N] FP16 output
    unsigned int       K,
    unsigned int       N
) {
    // 4 rows per WG, 64 threads (1 wavefront) per row
    const unsigned int ROWS_PER_WG = 4;
    const unsigned int WARP_SIZE   = 64;

    unsigned int row  = blockIdx.x * ROWS_PER_WG + threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;

    if (row >= N) return;

    // K must be divisible by 4 for v_dot4_i32_i8; handle remainder separately.
    // For typical Qwen shapes (K=4096), K is divisible by 16, so no remainder.
    unsigned int vec_len = K >> 2;  // K / 4 (number of int32 groups)

    const int* W_i32 = reinterpret_cast<const int*>(W + (size_t)row * K);
    const int* x_i32 = reinterpret_cast<const int*>(x);

    int acc = 0;

    // Each lane processes vec_len/64 elements (strided across the K dimension)
    for (unsigned int i = lane; i < vec_len; i += WARP_SIZE) {
        int w_packed = W_i32[i];
        int x_packed = x_i32[i];
        // __builtin_amdgcn_sdot4 maps to v_dot4_i32_i8:
        //   acc += w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]  (all INT8 signed)
        acc = __builtin_amdgcn_sdot4(w_packed, x_packed, acc, false);
    }

    // Handle remainder (K not divisible by 4)
    if (lane == 0) {
        unsigned int rem_start = vec_len * 4;
        const signed char* W_sc = W + (size_t)row * K;
        for (unsigned int k = rem_start; k < K; k++) {
            acc += (int)W_sc[k] * (int)x[k];
        }
    }

    // Wavefront butterfly reduction: sum across 64 lanes
    acc = warp_reduce_sum_i32(acc);

    if (lane == 0) {
        // Epilogue: INT32 → FP32, scale by (scale_w[row] * scale_a), → FP16
        float result = (float)acc * scale_w[row] * scale_a;
        out[row] = __float2half(result);
    }
}

// ---------------------------------------------------------------------------
// Batched variant that processes multiple scale_a values (for future extension)
// Simplified version: same as above but with scale_a passed per batch element.
// (Not used in current tests, kept for API completeness)
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_w8a8_batched(
    const signed char* __restrict__ x,       // [B, K] INT8 input activations
    const signed char* __restrict__ W,       // [N, K] INT8 weights
    const float*       __restrict__ scale_w, // [N] FP32 per-channel weight scales
    const float*       __restrict__ scale_a, // [B] FP32 per-tensor activation scales
    __half*            __restrict__ out,     // [B, N] FP16 output
    unsigned int       K,
    unsigned int       N,
    unsigned int       B
) {
    // For batch: gridDim.x = ceil(N/4), gridDim.y = B
    const unsigned int ROWS_PER_WG = 4;
    const unsigned int WARP_SIZE   = 64;

    unsigned int batch = blockIdx.y;
    unsigned int row   = blockIdx.x * ROWS_PER_WG + threadIdx.x / WARP_SIZE;
    unsigned int lane  = threadIdx.x % WARP_SIZE;

    if (row >= N || batch >= B) return;

    unsigned int vec_len = K >> 2;

    const int* W_i32 = reinterpret_cast<const int*>(W + (size_t)row * K);
    const int* x_i32 = reinterpret_cast<const int*>(x + (size_t)batch * K);

    int acc = 0;
    for (unsigned int i = lane; i < vec_len; i += WARP_SIZE) {
        acc = __builtin_amdgcn_sdot4(W_i32[i], x_i32[i], acc, false);
    }

    // Remainder
    if (lane == 0) {
        unsigned int rem_start = vec_len * 4;
        const signed char* W_sc = W + (size_t)row * K;
        const signed char* x_sc = x + (size_t)batch * K;
        for (unsigned int k = rem_start; k < K; k++) {
            acc += (int)W_sc[k] * (int)x_sc[k];
        }
    }

    acc = warp_reduce_sum_i32(acc);

    if (lane == 0) {
        float result = (float)acc * scale_w[row] * scale_a[batch];
        out[(size_t)batch * N + row] = __float2half(result);
    }
}
