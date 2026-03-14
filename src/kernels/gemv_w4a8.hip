/**
 * W4A8 GEMV for decode (M=1) on gfx906.
 *
 * W4A8 = INT4 weights stored on-device + INT8 activations (from activation_quant).
 * Memory advantage: reads K/2 bytes of weights vs K bytes for W8A8 → closer to W4A16 bandwidth.
 * Compute: INT32 accumulator via INT8 dot products (v_dot4_i32_i8) after on-the-fly
 * INT4→INT8 weight unpacking.
 *
 * Two kernels:
 *   1. gemv_w4a8_dot4: Uses v_dot4_i32_i8 (4 INT8×INT8 per instr). Unpack INT4→INT8 inline.
 *   2. gemv_w4a8_dot8: Uses v_dot8_i32_i4 (8 INT4×INT4 per instr). Requires activation
 *      splitting into low/high nibbles since activations are INT8 not INT4.
 *
 * Weight format (W4A8 packed layout, compatible with GPTQ repacker):
 *   W_packed[N * K/8] uint32 array, row-major: W_packed[row * (K/8) + k_group]
 *   Each uint32 holds 8 signed INT4 nibbles (zero-subtracted during repacking):
 *     bits[3:0]  = w[k_group*8+0], signed INT4 ∈ [-8, 7]
 *     bits[7:4]  = w[k_group*8+1]
 *     ...
 *     bits[31:28] = w[k_group*8+7]
 *
 * Scale layout: scale_w[N] FP32 per-channel (includes group scale from GPTQ).
 *   Epilogue: y[n] = FP16(sum_k(W[n,k] * x[k]) * scale_w[n] * scale_a)
 *
 * For W4A8 repacked from GPTQ:
 *   GPTQ format: qweight[K/8, N] uint32 (K-major), scales[K/group_size, N] FP16
 *   Repacked format: W_packed[N, K/8] uint32 (N-major, row-major), scale_w[N] FP32 per-channel
 *     - Weights transposed to row-major for coalesced lane access
 *     - Per-group scales rolled into per-channel scale (if group_size=K for simple case)
 *       OR kept separate in a separate scale buffer (for group_size < K)
 *   For group_size=128: we keep per-group scales separate (see gemv_w4a8_grouped below).
 *
 * Grid:  (ceil(N / 4), 1, 1)
 * Block: (256, 1, 1) — 4 wavefronts (rows) × 64 threads each
 *
 * v_dot4_i32_i8 confirmed working on gfx906 (validated by W8A8 kernels).
 * v_dot8_i32_i4 confirmed in ISA for gfx906 (llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html).
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// Warp-level INT32 sum reduction (64 lanes on gfx906)
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
// Kernel 1: gemv_w4a8_dot4
//
// Uses v_dot4_i32_i8 after unpacking INT4 weights to INT8.
// This is the primary "W4A8" kernel: weights stored as INT4 (K/2 bytes),
// activations are INT8, inner loop uses v_dot4_i32_i8 for INT32 accumulation.
//
// Each lane handles K/64 groups of 8 weights (8 activations per group).
// For each group: unpack 8 INT4 → 8 INT8, then 2× v_dot4 (4+4).
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_w4a8_dot4(
    const signed char*   __restrict__ x,         // [K] INT8 input activation
    const unsigned int*  __restrict__ W_packed,  // [N, K/8] packed INT4 (W4A8 layout)
    const float*         __restrict__ scale_w,   // [N] FP32 per-channel weight scales
    float                scale_a,                // FP32 per-tensor activation scale
    __half*              __restrict__ out,        // [N] FP16 output
    unsigned int         K,
    unsigned int         N
) {
    const unsigned int ROWS_PER_WG = 4;
    const unsigned int WARP_SIZE   = 64;

    unsigned int row  = blockIdx.x * ROWS_PER_WG + threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;

    if (row >= N) return;

    // K/8 groups per row (each uint32 holds 8 INT4 nibbles)
    unsigned int vec_len = K >> 3;

    const unsigned int* W_row = W_packed + (size_t)row * vec_len;

    int acc = 0;

    // Each lane strides across K dimension
    for (unsigned int i = lane; i < vec_len; i += WARP_SIZE) {
        unsigned int w_packed = W_row[i];
        unsigned int k_base   = i << 3;  // i * 8

        // Load 8 consecutive INT8 activations as 2 int32 words
        int x_lo = *reinterpret_cast<const int*>(x + k_base);
        int x_hi = *reinterpret_cast<const int*>(x + k_base + 4);

        // Unpack INT4 → INT8 (sign-extend each 4-bit nibble to 8 bits).
        // Nibble layout: bits[4b+3:4b] = nibble b, values ∈ [-8, 7].
        // Sign-extend via arithmetic right shift (guaranteed on gfx906/HIP):
        //   nibble_s8 = ((int)(w << (28 - 4*b))) >> 28  → ∈ [-8, 7] as int
        // Pack 4 sign-extended bytes into one int32 for v_dot4_i32_i8:

        // Lower 4 nibbles → w_i8_lo (bytes 0..3)
        int w_i8_lo =
              (((int)(w_packed <<  28) >> 28) & 0xFF)
            | ((((int)(w_packed << 24) >> 28) & 0xFF) <<  8)
            | ((((int)(w_packed << 20) >> 28) & 0xFF) << 16)
            | ((((int)(w_packed << 16) >> 28) & 0xFF) << 24);

        // Upper 4 nibbles → w_i8_hi (bytes 4..7)
        int w_i8_hi =
              (((int)(w_packed <<  12) >> 28) & 0xFF)
            | ((((int)(w_packed <<   8) >> 28) & 0xFF) <<  8)
            | ((((int)(w_packed <<   4) >> 28) & 0xFF) << 16)
            | ((((int)(w_packed        ) >> 28) & 0xFF) << 24);

        // v_dot4_i32_i8: acc += dot(w_i8_lo, x_lo) + dot(w_i8_hi, x_hi)
        acc = __builtin_amdgcn_sdot4(w_i8_lo, x_lo, acc, false);
        acc = __builtin_amdgcn_sdot4(w_i8_hi, x_hi, acc, false);
    }

    // Scalar remainder (K not divisible by 8; rare for Qwen K=4096)
    if (lane == 0) {
        unsigned int k_rem = vec_len * 8;
        for (unsigned int k = k_rem; k < K; k++) {
            unsigned int b = k & 7;
            unsigned int gi = k >> 3;
            unsigned int w_p = W_row[gi];
            int nibble_s = ((int)(w_p << (28 - b*4))) >> 28;
            acc += nibble_s * (int)x[k];
        }
    }

    // Wavefront butterfly reduction (64 lanes → lane 0)
    acc = warp_reduce_sum_i32(acc);

    if (lane == 0) {
        float result = (float)acc * scale_w[row] * scale_a;
        out[row] = __float2half(result);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: gemv_w4a8_dot8
//
// Uses v_dot8_i32_i4 (8 INT4×INT4 products per instruction → INT32 accumulator).
// Since activations are INT8 (not INT4), we split each INT8 activation x into:
//   x = x_lo_u + 16 * x_hi    where x_lo_u = x & 0xF ∈ [0,15] (unsigned)
//                                    x_hi   = x >> 4 ∈ [-8,7]  (signed)
// Then: dot(W,x) = dot(W, x_lo_u) + 16*dot(W, x_hi)
// where dot(W, x_lo_u) uses v_dot8 with x_lo_u mapped to signed INT4:
//   x_lo_s = x_lo_u - 8 ∈ [-8,7]   (center to get signed representation)
//   dot(W, x_lo_u) = dot(W, x_lo_s) + 8*sum(W)
// Final: dot(W,x) = dot(W, x_lo_s) + 8*sum(W) + 16*dot(W, x_hi)
//                 = v_dot8(W, x_lo_s) + 8*w_sum_lane + 16*v_dot8(W, x_hi)
//
// w_sum_lane is accumulated inline per lane (sum of all sign-extended nibbles
// for the K-range this lane processes), then reduced at the end.
//
// This kernel uses v_dot8_i32_i4 as the compute primitive.
// Intrinsic: __builtin_amdgcn_sdot8(int a, int b, int c, bool sat)
//   → c + sum_{i=0}^{7} a[i]*b[i], where a[i]=bits[4i+3:4i] (signed INT4)
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_w4a8_dot8(
    const signed char*   __restrict__ x,         // [K] INT8 input activation
    const unsigned int*  __restrict__ W_packed,  // [N, K/8] packed INT4 (W4A8 layout)
    const float*         __restrict__ scale_w,   // [N] FP32 per-channel weight scales
    float                scale_a,                // FP32 per-tensor activation scale
    __half*              __restrict__ out,        // [N] FP16 output
    unsigned int         K,
    unsigned int         N
) {
    const unsigned int ROWS_PER_WG = 4;
    const unsigned int WARP_SIZE   = 64;

    unsigned int row  = blockIdx.x * ROWS_PER_WG + threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;

    if (row >= N) return;

    unsigned int vec_len = K >> 3;
    const unsigned int* W_row = W_packed + (size_t)row * vec_len;

    int acc_lo = 0;   // dot(W_i4, x_lo_s) where x_lo_s = (x & 0xF) - 8 ∈ [-8,7]
    int acc_hi = 0;   // dot(W_i4, x_hi)   where x_hi = x >> 4 ∈ [-8,7]
    int w_sum  = 0;   // sum of all signed INT4 weight values (for correction)

    for (unsigned int i = lane; i < vec_len; i += WARP_SIZE) {
        unsigned int w_packed = W_row[i];
        unsigned int k_base   = i << 3;  // i * 8

        // Load 8 INT8 activations
        // Signed byte values x[k_base..k_base+7]
        // Split each into: x_lo_s = (x & 0xF) - 8 ∈ [-8,7] and x_hi = x>>4 ∈ [-8,7]
        // Pack 8 nibbles into 2 int32 words for v_dot8
        int x_lo_packed = 0;
        int x_hi_packed = 0;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            int xb = (int)x[k_base + b];
            int xb_lo_u = xb & 0xF;           // unsigned low nibble: 0..15
            int xb_lo_s = xb_lo_u - 8;        // signed low nibble: -8..7
            int xb_hi   = xb >> 4;            // arithmetic right shift: -8..7
            // Pack signed INT4 nibbles (4 bits each) into int32
            x_lo_packed |= ((xb_lo_s & 0xF) << (b * 4));
            x_hi_packed |= ((xb_hi   & 0xF) << (b * 4));
        }

        // v_dot8_i32_i4: 8 signed INT4 × INT4 → INT32
        acc_lo = __builtin_amdgcn_sdot8((int)w_packed, x_lo_packed, acc_lo, false);
        acc_hi = __builtin_amdgcn_sdot8((int)w_packed, x_hi_packed, acc_hi, false);

        // Accumulate weight sum for the correction term (8 * sum(W) per group)
        // sum of 8 signed INT4 nibbles in w_packed:
        int wsum_group = 0;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            // Sign-extend nibble b via arithmetic right shift
            wsum_group += ((int)(w_packed << (28 - b*4))) >> 28;
        }
        w_sum += wsum_group;
    }

    // Wavefront butterfly reduction
    acc_lo = warp_reduce_sum_i32(acc_lo);
    acc_hi = warp_reduce_sum_i32(acc_hi);
    w_sum  = warp_reduce_sum_i32(w_sum);

    if (lane == 0) {
        // Reconstruct: dot(W, x) = acc_lo + 8*w_sum + 16*acc_hi
        // (from: dot(W, x_lo_s + 8) + 16*dot(W, x_hi) = acc_lo + 8*w_sum + 16*acc_hi)
        int final_acc = acc_lo + 8 * w_sum + 16 * acc_hi;
        float result  = (float)final_acc * scale_w[row] * scale_a;
        out[row] = __float2half(result);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: gemv_w4a8_grouped
//
// W4A8 GEMV with per-group scales (like standard GPTQ group_size=128).
// Uses gemv_w4a8_dot4 approach but with per-group weight scales.
//
// Scale layout: scale_w[K/group_size, N] FP16 (same as v3 INT4 GEMV layout)
// The activation scale is separate (per-tensor FP32).
//
// This is needed to match GPTQ quantization accuracy.
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_w4a8_grouped(
    const signed char*   __restrict__ x,           // [K] INT8 input activation
    const unsigned int*  __restrict__ W_packed,    // [N, K/8] packed INT4 (W4A8 layout)
    const __half*        __restrict__ scale_w_grp, // [K/group_size, N] FP16 per-group scales
    float                scale_a,                  // FP32 per-tensor activation scale
    __half*              __restrict__ out,          // [N] FP16 output
    unsigned int         K,
    unsigned int         N,
    unsigned int         group_size                // must be divisible by 8
) {
    const unsigned int ROWS_PER_WG = 4;
    const unsigned int WARP_SIZE   = 64;

    unsigned int row  = blockIdx.x * ROWS_PER_WG + threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;

    if (row >= N) return;

    unsigned int vec_len = K >> 3;
    // groups_per_scale: how many 8-element K-groups per scale group
    unsigned int groups_per_scale = group_size >> 3;  // group_size/8
    unsigned int gps_log2 = 31u - __builtin_clz(groups_per_scale);  // log2 for power-of-2

    const unsigned int* W_row = W_packed + (size_t)row * vec_len;

    float acc = 0.0f;
    float cur_scale = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int i = lane; i < vec_len; i += WARP_SIZE) {
        unsigned int w_packed = W_row[i];
        unsigned int k_base   = i << 3;

        // Per-group scale lookup
        unsigned int sg = i >> gps_log2;  // scale group index
        if (sg != last_sg) {
            cur_scale = __half2float(scale_w_grp[sg * N + row]);
            last_sg = sg;
        }

        // Load 8 INT8 activations
        int x_lo = *reinterpret_cast<const int*>(x + k_base);
        int x_hi = *reinterpret_cast<const int*>(x + k_base + 4);

        // Unpack INT4 → INT8 (sign-extend nibbles)
        int w_i8_lo =
              (((int)(w_packed <<  28) >> 28) & 0xFF)
            | ((((int)(w_packed << 24) >> 28) & 0xFF) <<  8)
            | ((((int)(w_packed << 20) >> 28) & 0xFF) << 16)
            | ((((int)(w_packed << 16) >> 28) & 0xFF) << 24);

        int w_i8_hi =
              (((int)(w_packed <<  12) >> 28) & 0xFF)
            | ((((int)(w_packed <<   8) >> 28) & 0xFF) <<  8)
            | ((((int)(w_packed <<   4) >> 28) & 0xFF) << 16)
            | ((((int)(w_packed        ) >> 28) & 0xFF) << 24);

        // INT32 dot products via v_dot4_i32_i8
        int dot_val = __builtin_amdgcn_sdot4(w_i8_lo, x_lo, 0, false);
        dot_val     = __builtin_amdgcn_sdot4(w_i8_hi, x_hi, dot_val, false);

        // Scale immediately (FP32 accumulation per group)
        acc += (float)dot_val * cur_scale;
    }

    // Warp reduction (FP32 butterfly)
    acc += __shfl_xor(acc, 32);
    acc += __shfl_xor(acc, 16);
    acc += __shfl_xor(acc, 8);
    acc += __shfl_xor(acc, 4);
    acc += __shfl_xor(acc, 2);
    acc += __shfl_xor(acc, 1);

    if (lane == 0) {
        float result = acc * scale_a;
        out[row] = __float2half(result);
    }
}
