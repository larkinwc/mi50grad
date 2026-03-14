/**
 * Dynamic per-tensor INT8 quantization of FP16 activations.
 *
 * Two-kernel approach:
 *   1. activation_quant_reduce: Finds max absolute value across the input tensor
 *      using two-level reduction (DPP warp reduction + atomicMax via float-as-uint trick).
 *   2. activation_quant_quant: Computes scale = max_abs / 127.0 and quantizes
 *      each element to INT8: clamp(round(x / scale), -128, 127).
 *
 * Output: INT8 tensor + FP32 scale.
 * This is the foundation for W8A8 and W4A8 kernels.
 *
 * gfx906 notes:
 *   - Wavefront size = 64, so __shfl_xor / __shfl_down cover full warp
 *   - DPP reductions via __shfl_xor (6 steps for 64 lanes)
 *   - atomicMax on int32 reinterpretation of float (works for non-negative floats
 *     since IEEE 754 bit pattern is monotonic for positive values)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// Warp-level max reduction via DPP (64 lanes on gfx906)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor(val, 32));
    val = fmaxf(val, __shfl_xor(val, 16));
    val = fmaxf(val, __shfl_xor(val, 8));
    val = fmaxf(val, __shfl_xor(val, 4));
    val = fmaxf(val, __shfl_xor(val, 2));
    val = fmaxf(val, __shfl_xor(val, 1));
    return val;
}

// ---------------------------------------------------------------------------
// Kernel 1: activation_quant_reduce
//
// Finds max|x[i]| across the entire tensor.
// Each workgroup of 256 threads (4 warps) reduces a chunk of 4*256=1024 elements
// (threads stride across the full tensor when chunk_size > 1024).
// Uses DPP warp reduction + shared-memory cross-warp reduction + atomicMax
// (encoded as uint32 of non-negative float, safe because max_abs >= 0).
//
// Grid: (ceil(n/1024), 1, 1)  Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void activation_quant_reduce(
    const __half* __restrict__ x,   // [n] FP16 input
    float*        __restrict__ max_abs_out, // scalar output (must be pre-zeroed)
    unsigned int  n
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 64;
    const unsigned int lane    = tid % 64;

    // Each WG owns a stripe of elements: elements where (elem_idx / 1024) == blockIdx.x
    // i.e., global stride = gridDim.x * 1024 = gridDim.x * (blockDim.x * 4)
    // Simpler: each WG processes elements at positions:
    //   start = blockIdx.x * blockDim.x + tid,  stride = gridDim.x * blockDim.x
    // This ensures load-balance across all WGs.
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int idx    = blockIdx.x * blockDim.x + tid;

    float local_max = 0.0f;
    while (idx < n) {
        float v = __half2float(x[idx]);
        local_max = fmaxf(local_max, fabsf(v));
        idx += stride;
    }

    // Warp-level DPP max reduction
    local_max = warp_reduce_max(local_max);

    // Cross-warp reduction via LDS (4 warps per WG)
    __shared__ float s_warp_max[4];
    if (lane == 0) s_warp_max[warp_id] = local_max;
    __syncthreads();

    float wg_max = 0.0f;
    if (tid < 4) {
        wg_max = s_warp_max[tid];
    }
    // Reduce the 4 warp-max values (within the first warp, lanes 0-3)
    // Use shfl_xor to broadcast across lane 0-3
    wg_max = fmaxf(wg_max, __shfl_xor(wg_max, 2));
    wg_max = fmaxf(wg_max, __shfl_xor(wg_max, 1));

    // Thread 0 has the WG max; use atomicMax (via uint trick for non-negative floats)
    if (tid == 0 && wg_max > 0.0f) {
        // IEEE 754: for non-negative floats, bit-patterns are monotonically ordered
        atomicMax((unsigned int*)max_abs_out,
                  __float_as_uint(wg_max));
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: activation_quant_quant
//
// Given max_abs (computed by kernel 1), computes:
//   scale = max_abs / 127.0
//   q[i] = clamp(round(x[i] / scale), -128, 127)
// Writes INT8 output and FP32 scale.
//
// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void activation_quant_quant(
    const __half*  __restrict__ x,           // [n] FP16 input
    signed char*   __restrict__ q_out,       // [n] INT8 output
    const float*   __restrict__ max_abs_in,  // scalar: max absolute value
    float*         __restrict__ scale_out,   // scalar output: scale = max_abs / 127.0
    unsigned int   n
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;

    // Thread 0 of block 0 writes the scale
    float scale;
    if (blockIdx.x == 0 && tid == 0) {
        float max_abs = *max_abs_in;
        // Guard against all-zero input
        scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        *scale_out = scale;
    }

    // All threads need the scale — broadcast via shared memory
    __shared__ float s_scale;
    if (blockIdx.x == 0 && tid == 0) s_scale = scale;
    __syncthreads();

    // For blocks other than block 0, we need to read scale_out after block 0 writes it.
    // However, there is no cross-block synchronization in HIP without atomics.
    // Strategy: read max_abs_in directly in every block, compute scale locally.
    // This is safe: max_abs_in is written before this kernel is launched.
    float s;
    if (blockIdx.x == 0) {
        s = s_scale;
    } else {
        float max_abs = *max_abs_in;
        s = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    }

    if (gid < n) {
        float v = __half2float(x[gid]);
        // Quantize: round(v / scale), clamp to [-128, 127]
        float q_f = __builtin_amdgcn_fmed3f(
            roundf(v / s), -128.0f, 127.0f);
        q_out[gid] = (signed char)(int)q_f;
    }
}
