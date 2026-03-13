/**
 * Optimized INT4 GEMV for decode (M=1) on gfx906.
 *
 * Three kernels:
 * 1. gemv_int4_v2_direct: No split-K, writes FP16 directly. Fastest for small K.
 * 2. gemv_int4_v2_splitk: Split-K with atomicAdd to FP32 buffer + convert.
 *    Requires: memset(C_fp32) before + fp32_to_fp16 after = 3 launches total.
 * 3. gemv_int4_v2_fused: Fully fused split-K — initializes C_fp32 internally
 *    on split_id=0, writes FP16 output directly in the last-completing split
 *    (tracked via atomic counter). Reduces to 1 launch, no separate memset
 *    or fp32_to_fp16 needed.
 *
 * All kernels use coalesced B_q4 reads (consecutive threads read consecutive columns).
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Inline dequant+dot for 8 INT4 values against 8 FP16 activations
// Vectorized A loads (half2) + factored zero subtraction
__device__ __forceinline__ float dequant_dot8(
    unsigned int packed, float cur_scale, float cur_zero,
    const __half* __restrict__ A, unsigned int a_base
) {
    // Vectorized: 4 × half2 loads instead of 8 × half
    __half2 a01 = *reinterpret_cast<const __half2*>(A + a_base);
    __half2 a23 = *reinterpret_cast<const __half2*>(A + a_base + 2);
    __half2 a45 = *reinterpret_cast<const __half2*>(A + a_base + 4);
    __half2 a67 = *reinterpret_cast<const __half2*>(A + a_base + 6);

    float a0 = __half2float(__low2half(a01));
    float a1 = __half2float(__high2half(a01));
    float a2 = __half2float(__low2half(a23));
    float a3 = __half2float(__high2half(a23));
    float a4 = __half2float(__low2half(a45));
    float a5 = __half2float(__high2half(a45));
    float a6 = __half2float(__low2half(a67));
    float a7 = __half2float(__high2half(a67));

    // Factor out zero: (q-z)*s*a = q*s*a - z*s*sum(a)
    float zs = cur_zero * cur_scale;
    float a_sum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

    float acc = 0.0f;
    acc += (float)(packed        & 0xF) * a0;
    acc += (float)((packed >> 4)  & 0xF) * a1;
    acc += (float)((packed >> 8)  & 0xF) * a2;
    acc += (float)((packed >> 12) & 0xF) * a3;
    acc += (float)((packed >> 16) & 0xF) * a4;
    acc += (float)((packed >> 20) & 0xF) * a5;
    acc += (float)((packed >> 24) & 0xF) * a6;
    acc += (float)(packed >> 28)         * a7;

    return acc * cur_scale - zs * a_sum;
}

// Direct FP16 output (no split-K). Grid: (ceil(N/256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void gemv_int4_v2_direct(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __half* __restrict__ C,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;

    float acc = 0.0f;
    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int kg = 0; kg < num_k_groups; kg++) {
        unsigned int sg = kg / groups_per_scale;
        if (sg != last_sg) {
            cur_scale = __half2float(scales[sg * N + col]);
            cur_zero  = __half2float(zeros[sg * N + col]);
            last_sg = sg;
        }
        unsigned int packed = B_q4[kg * N + col];
        acc += dequant_dot8(packed, cur_scale, cur_zero, A, kg << 3);
    }

    C[col] = __float2half(acc);
}

// Split-K with atomicAdd. Grid: (ceil(N/256), K_SPLITS, 1), Block: (256, 1, 1)
// C_fp32 must be zeroed before launch.
extern "C" __global__ void gemv_int4_v2_splitk(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    float* __restrict__ C_fp32,
    unsigned int K,
    unsigned int N,
    unsigned int group_size
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int k_splits = gridDim.y;
    unsigned int k_split_id = blockIdx.y;
    unsigned int kg_start = (num_k_groups * k_split_id) / k_splits;
    unsigned int kg_end = (num_k_groups * (k_split_id + 1)) / k_splits;

    float acc = 0.0f;
    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        unsigned int sg = kg / groups_per_scale;
        if (sg != last_sg) {
            cur_scale = __half2float(scales[sg * N + col]);
            cur_zero  = __half2float(zeros[sg * N + col]);
            last_sg = sg;
        }
        unsigned int packed = B_q4[kg * N + col];
        acc += dequant_dot8(packed, cur_scale, cur_zero, A, kg << 3);
    }

    atomicAdd(&C_fp32[col], acc);
}

// Convert FP32 to FP16
extern "C" __global__ void fp32_to_fp16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = __float2half(src[idx]);
    }
}

/**
 * Fused split-K INT4 GEMV: single kernel launch, no separate memset or convert.
 *
 * Protocol (race-free using atomic counter per column):
 *   - All k_splits tiles atomicAdd their partial FP32 sums into C_fp32[col].
 *     C_fp32 is a PERSISTENT buffer initialized to 0 once at engine startup.
 *   - Each tile atomicAdd(&done[col], 1). The tile that sees old == k_splits-1
 *     is the LAST tile for that column — it:
 *       (a) reads C_fp32[col] (which now holds the complete sum)
 *       (b) writes __float2half(C_fp32[col]) to C_fp16[col]
 *          (optionally adding residual[col] if residual != nullptr)
 *       (c) resets C_fp32[col] = 0.0f and done[col] = 0 for the next call
 *
 * This means C_fp32 and done[] need to be zero-initialized ONLY ONCE at
 * buffer allocation time (e.g., via device.memset at engine init, not per-call).
 * Each call leaves C_fp32 and done[] zeroed for the next call.
 *
 * Parameters:
 *   A        - input activations [K] FP16
 *   B_q4     - packed INT4 weight matrix [K/8, N] uint32
 *   scales   - quantization scales [num_groups, N] FP16
 *   zeros    - quantization zeros  [num_groups, N] FP16
 *   C_fp32   - persistent FP32 accumulator buffer [N] float
 *              Must be zeroed at init time. Reset internally after each call.
 *   done     - persistent atomic counter buffer [N] uint32
 *              Must be zeroed at init time. Reset internally after each call.
 *   C_fp16   - FP16 output buffer [N] half
 *   K, N, group_size - GEMV dimensions
 *   k_splits - number of K-dimension splits (== gridDim.y)
 *   residual - optional: if non-null, C_fp16[col] += residual[col]
 *              This fuses the residual-add into the GEMV, eliminating a
 *              separate residual_add launch (saves 1 launch for down_proj).
 *
 * Grid:  (ceil(N/256), k_splits, 1)
 * Block: (256, 1, 1)
 */
extern "C" __global__ void gemv_int4_v2_fused(
    const __half* __restrict__ A,
    const unsigned int* __restrict__ B_q4,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    float* __restrict__ C_fp32,
    unsigned int* __restrict__ done,
    __half* __restrict__ C_fp16,
    unsigned int K,
    unsigned int N,
    unsigned int group_size,
    unsigned int k_splits,
    const __half* __restrict__ residual  // optional: if non-null, add residual[col]
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    unsigned int num_k_groups = K >> 3;
    unsigned int groups_per_scale = group_size >> 3;
    unsigned int k_split_id = blockIdx.y;
    unsigned int kg_start = (num_k_groups * k_split_id) / k_splits;
    unsigned int kg_end = (num_k_groups * (k_split_id + 1)) / k_splits;

    // Compute partial FP32 sum for this tile
    float acc = 0.0f;
    float cur_scale = 0.0f, cur_zero = 0.0f;
    unsigned int last_sg = 0xFFFFFFFF;

    for (unsigned int kg = kg_start; kg < kg_end; kg++) {
        unsigned int sg = kg / groups_per_scale;
        if (sg != last_sg) {
            cur_scale = __half2float(scales[sg * N + col]);
            cur_zero  = __half2float(zeros[sg * N + col]);
            last_sg = sg;
        }
        unsigned int packed = B_q4[kg * N + col];
        acc += dequant_dot8(packed, cur_scale, cur_zero, A, kg << 3);
    }

    // Accumulate into persistent FP32 buffer
    atomicAdd(&C_fp32[col], acc);

    // Fence to ensure our atomicAdd is globally visible before incrementing done
    __threadfence();

    // Increment completion counter. The last tile writes FP16 output and resets.
    unsigned int old_done = atomicAdd(&done[col], 1U);
    if (old_done == k_splits - 1U) {
        // We are the last tile to complete for this column.
        // C_fp32[col] now holds the complete accumulated sum.
        float total = C_fp32[col];

        // Optional residual epilogue: add residual[col] if pointer is non-null
        if (residual != nullptr) {
            C_fp16[col] = __float2half(total + __half2float(residual[col]));
        } else {
            C_fp16[col] = __float2half(total);
        }

        // Reset buffers for the next kernel call (so no per-call memset is needed)
        C_fp32[col] = 0.0f;
        done[col] = 0U;
    }
}
