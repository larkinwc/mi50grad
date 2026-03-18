/**
 * Fused P2P allreduce + RMSNorm kernel for TP inference.
 *
 * Milestone: fused-kernel
 *
 * This kernel eliminates the intermediate buffer write and separate RMSNorm kernel
 * launch by fusing:
 *   1. P2P allreduce: Each GPU reads partials from all peers via BAR1-mapped P2P
 *   2. FP32 accumulation: Reduces all partials with FP32 precision
 *   3. RMSNorm: Applies RMSNorm with learned weights
 *   4. Write result: Single write to output buffer
 *
 * Compared to separate kernels (kernel_p2p_allreduce + rmsnorm_v3):
 *   - Eliminates 1 intermediate buffer write per token
 *   - Eliminates 1 kernel launch per layer per token
 *   - Reduces kernel launch count from 128 to 64 per token (for 64-layer model)
 *
 * Each GPU runs the fused kernel simultaneously:
 *   - partial_local: this GPU's partial GEMV result
 *   - partial_peer0, partial_peer1, partial_peer2: peer GPU partials (P2P pointers)
 *   - weight: RMSNorm weight buffer (local)
 *   - output: output buffer (local, writes RMSNorm result)
 *
 * Target: gfx906 (MI50/MI60), compiled as SHARED LIBRARY with:
 *   hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC
 *
 * Exposes host-callable C functions for ctypes-based dispatch.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/* ===================================================================
 * GPU kernel: fused P2P allreduce + RMSNorm
 * =================================================================== */

/**
 * kernel_p2p_allreduce_rmsnorm_tp4_kernel:
 *
 *   fused_op[i] = rmsnorm(
 *     partial_local[i] + partial_peer0[i] + partial_peer1[i] + partial_peer2[i]
 *   ) * weight[i]
 *
 * Runs on EACH GPU simultaneously. Each GPU:
 *   - partial_local:   its own partial GEMV result (local memory)
 *   - partial_peer0/1/2: the other 3 GPUs' partial buffers (P2P pointers via BAR1)
 *   - weight:          RMSNorm weight buffer (local memory)
 *   - output:          output buffer (local memory, writes RMSNorm result)
 *
 * Grid: (batch_size, 1, 1), Block: (256, 1, 1)
 * Each thread processes 2 FP16 elements (half2 packing).
 * FP32 accumulation for correctness (mandatory on gfx906).
 *
 * Thread layout for RMSNorm (same as rmsnorm_v3):
 *   Each thread t processes 8-element groups at {t*8, t*8+2048, ...}
 *   Scalar tail for elements [(dim/8)*8, dim-1] when dim%8 != 0.
 *
 * @param output        Output buffer (RMSNorm result), one per batch item
 * @param partial_local This GPU's partial buffer
 * @param partial_peer0 First peer GPU's partial buffer (P2P pointer)
 * @param partial_peer1 Second peer GPU's partial buffer (P2P pointer)
 * @param partial_peer2 Third peer GPU's partial buffer (P2P pointer)
 * @param weight        RMSNorm weight buffer
 * @param dim           Hidden dimension (number of FP16 elements)
 * @param eps           RMSNorm epsilon (default: 1e-6)
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_p2p_allreduce_rmsnorm_tp4_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ partial_peer1,
    const __half* __restrict__ partial_peer2,
    const __half* __restrict__ weight,
    unsigned int dim,
    float eps
) {
    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane   = threadIdx.x % 64;

    // Pointer to this batch item's data
    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    const __half* p_peer1 = partial_peer1 + blockIdx.x * dim;
    const __half* p_peer2 = partial_peer2 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;

    // =================================================================
    // Phase 1: Reduce all partials + compute sum-of-squares
    // Uses vectorized 8-wide float4 loads for bandwidth efficiency
    // =================================================================
    float sum_sq = 0.0f;

    // Vectorized path: thread t covers {t*8, t*8+2048, t*8+4096, ...}
    // while t*8 + k*2048 + 7 < dim
    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        // Load 8 elements from each partial buffer (float4 = 128-bit = 8 FP16)
        float4 pl4 = *reinterpret_cast<const float4*>(p_local + i);
        float4 p04 = *reinterpret_cast<const float4*>(p_peer0 + i);
        float4 p14 = *reinterpret_cast<const float4*>(p_peer1 + i);
        float4 p24 = *reinterpret_cast<const float4*>(p_peer2 + i);

        const __half2* pl_h2 = reinterpret_cast<const __half2*>(&pl4);
        const __half2* p0_h2 = reinterpret_cast<const __half2*>(&p04);
        const __half2* p1_h2 = reinterpret_cast<const __half2*>(&p14);
        const __half2* p2_h2 = reinterpret_cast<const __half2*>(&p24);

        float4 sum4;
        __half2* sum_h2 = reinterpret_cast<__half2*>(&sum4);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            // FP32 accumulation for each of the 8 elements
            float v0 = __half2float(__low2half (pl_h2[k]))
                     + __half2float(__low2half (p0_h2[k]))
                     + __half2float(__low2half (p1_h2[k]))
                     + __half2float(__low2half (p2_h2[k]));
            float v1 = __half2float(__high2half(pl_h2[k]))
                     + __half2float(__high2half(p0_h2[k]))
                     + __half2float(__high2half(p1_h2[k]))
                     + __half2float(__high2half(p2_h2[k]));

            sum_h2[k] = __halves2half2(__float2half(v0), __float2half(v1));

            // Accumulate sum-of-squares for RMSNorm
            sum_sq += v0 * v0 + v1 * v1;
        }

        // Store reduced sum temporarily (will be normalized in Phase 4)
        *reinterpret_cast<float4*>(out + i) = sum4;
    }

    // Scalar tail: elements in [(dim/8)*8, dim-1] (only if dim%8 != 0)
    {
        unsigned int vec_dim = (dim / 8) * 8;
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float v = __half2float(p_local[i])
                    + __half2float(p_peer0[i])
                    + __half2float(p_peer1[i])
                    + __half2float(p_peer2[i]);
            out[i] = __float2half(v);
            sum_sq += v * v;
        }
    }

    // =================================================================
    // Phase 2: 6-step butterfly warp reduction (wave64)
    // Reduces sum_sq across all 64 lanes in the warp
    // =================================================================
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);

    // =================================================================
    // Phase 3: Cross-warp reduction via LDS (4 warps × 64 lanes)
    // Each warp leader writes its partial sum to LDS
    // =================================================================
    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 4) {
        float total = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        s_warp[threadIdx.x] = total;
    }
    __syncthreads();
    float total_sq = s_warp[0];

    // Compute inverse RMS
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);

    // =================================================================
    // Phase 4: Apply RMSNorm + weight
    // Reads reduced sum from output buffer, applies normalization, writes final result
    // =================================================================
    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        // Load reduced sum (stored in Phase 1)
        float4 sum4 = *reinterpret_cast<const float4*>(out + i);
        // Load weight
        float4 w4   = *reinterpret_cast<const float4*>(weight + i);

        const __half2* sum_h2 = reinterpret_cast<const __half2*>(&sum4);
        const __half2* w_h2   = reinterpret_cast<const __half2*>(&w4);

        float4 out4;
        __half2* out_h2 = reinterpret_cast<__half2*>(&out4);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float s0 = __half2float(__low2half (sum_h2[k]));
            float s1 = __half2float(__high2half(sum_h2[k]));
            float w0 = __half2float(__low2half (w_h2[k]));
            float w1 = __half2float(__high2half(w_h2[k]));

            // Apply RMSNorm and weight
            out_h2[k] = __halves2half2(
                __float2half(s0 * rms_inv * w0),
                __float2half(s1 * rms_inv * w1)
            );
        }
        *reinterpret_cast<float4*>(out + i) = out4;
    }

    // Scalar tail: elements in [(dim/8)*8, dim-1]
    {
        unsigned int vec_dim = (dim / 8) * 8;
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float val = __half2float(out[i]);
            float w   = __half2float(weight[i]);
            out[i] = __float2half(val * rms_inv * w);
        }
    }
}

/**
 * kernel_p2p_allreduce_rmsnorm_tp2_kernel:
 *
 *   fused_op[i] = rmsnorm(partial_local[i] + partial_peer0[i]) * weight[i]
 *
 * Runs on BOTH GPUs simultaneously (TP=2).
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_p2p_allreduce_rmsnorm_tp2_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ weight,
    unsigned int dim,
    float eps
) {
    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane   = threadIdx.x % 64;

    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;

    // Phase 1: Reduce + sum-of-squares
    float sum_sq = 0.0f;

    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        float4 pl4 = *reinterpret_cast<const float4*>(p_local + i);
        float4 p04 = *reinterpret_cast<const float4*>(p_peer0 + i);

        const __half2* pl_h2 = reinterpret_cast<const __half2*>(&pl4);
        const __half2* p0_h2 = reinterpret_cast<const __half2*>(&p04);

        float4 sum4;
        __half2* sum_h2 = reinterpret_cast<__half2*>(&sum4);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float v0 = __half2float(__low2half (pl_h2[k]))
                     + __half2float(__low2half (p0_h2[k]));
            float v1 = __half2float(__high2half(pl_h2[k]))
                     + __half2float(__high2half(p0_h2[k]));

            sum_h2[k] = __halves2half2(__float2half(v0), __float2half(v1));
            sum_sq += v0 * v0 + v1 * v1;
        }
        *reinterpret_cast<float4*>(out + i) = sum4;
    }

    // Scalar tail
    {
        unsigned int vec_dim = (dim / 8) * 8;
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float v = __half2float(p_local[i]) + __half2float(p_peer0[i]);
            out[i] = __float2half(v);
            sum_sq += v * v;
        }
    }

    // Phase 2: Warp reduction
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);

    // Phase 3: Cross-warp reduction
    __shared__ float s_warp[4];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x < 4) {
        float total = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
        s_warp[threadIdx.x] = total;
    }
    __syncthreads();
    float total_sq = s_warp[0];

    float rms_inv = rsqrtf(total_sq / (float)dim + eps);

    // Phase 4: Apply RMSNorm + weight
    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        float4 sum4 = *reinterpret_cast<const float4*>(out + i);
        float4 w4   = *reinterpret_cast<const float4*>(weight + i);

        const __half2* sum_h2 = reinterpret_cast<const __half2*>(&sum4);
        const __half2* w_h2   = reinterpret_cast<const __half2*>(&w4);

        float4 out4;
        __half2* out_h2 = reinterpret_cast<__half2*>(&out4);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float s0 = __half2float(__low2half (sum_h2[k]));
            float s1 = __half2float(__high2half(sum_h2[k]));
            float w0 = __half2float(__low2half (w_h2[k]));
            float w1 = __half2float(__high2half(w_h2[k]));

            out_h2[k] = __halves2half2(
                __float2half(s0 * rms_inv * w0),
                __float2half(s1 * rms_inv * w1)
            );
        }
        *reinterpret_cast<float4*>(out + i) = out4;
    }

    // Scalar tail
    {
        unsigned int vec_dim = (dim / 8) * 8;
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float val = __half2float(out[i]);
            float w   = __half2float(weight[i]);
            out[i] = __float2half(val * rms_inv * w);
        }
    }
}

/* ===================================================================
 * Host-callable C wrappers for Python ctypes dispatch.
 * Uses hipLaunchKernelGGL for reliable ctypes-based launch.
 * Returns hipError_t (0 = success).
 * =================================================================== */

/**
 * kernel_p2p_allreduce_rmsnorm_tp4:
 *
 * Launch the TP=4 fused P2P allreduce + RMSNorm kernel.
 *
 * Args:
 *   output:        Output buffer (RMSNorm result), one per batch item
 *   partial_local: This GPU's partial buffer
 *   partial_peer0: First peer GPU's partial buffer (P2P pointer)
 *   partial_peer1: Second peer GPU's partial buffer (P2P pointer)
 *   partial_peer2: Third peer GPU's partial buffer (P2P pointer)
 *   weight:        RMSNorm weight buffer
 *   dim:           Hidden dimension (number of FP16 elements)
 *   batch_size:    Number of sequences in batch (typically 1 for decode)
 *   eps:           RMSNorm epsilon (default: 1e-6)
 *   stream:        HIP stream to launch on
 */
extern "C" int kernel_p2p_allreduce_rmsnorm_tp4(
    void* output,
    const void* partial_local,
    const void* partial_peer0,
    const void* partial_peer1,
    const void* partial_peer2,
    const void* weight,
    unsigned int dim,
    unsigned int batch_size,
    float eps,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    hipLaunchKernelGGL(kernel_p2p_allreduce_rmsnorm_tp4_kernel,
                       dim3(batch_size), dim3(256), 0, stream,
                       (__half*)output,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)partial_peer1,
                       (const __half*)partial_peer2,
                       (const __half*)weight,
                       dim, eps);
    return (int)hipGetLastError();
}

/**
 * kernel_p2p_allreduce_rmsnorm_tp2:
 *
 * Launch the TP=2 fused P2P allreduce + RMSNorm kernel.
 */
extern "C" int kernel_p2p_allreduce_rmsnorm_tp2(
    void* output,
    const void* partial_local,
    const void* partial_peer0,
    const void* weight,
    unsigned int dim,
    unsigned int batch_size,
    float eps,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    hipLaunchKernelGGL(kernel_p2p_allreduce_rmsnorm_tp2_kernel,
                       dim3(batch_size), dim3(256), 0, stream,
                       (__half*)output,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)weight,
                       dim, eps);
    return (int)hipGetLastError();
}
