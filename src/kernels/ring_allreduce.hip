/**
 * Ring allreduce kernels for TP=4 inference.
 *
 * Implements ring allreduce to distribute PCIe bandwidth across all GPUs,
 * replacing the star topology (GPU0 gathers/broadcasts everything)
 * with a ring where each GPU only communicates with its neighbors.
 *
 * Ring topology (TP=4):
 *   GPU0 → GPU1 → GPU2 → GPU3 → GPU0 (right/send direction)
 *   GPU0 ← GPU1 ← GPU2 ← GPU3 ← GPU0 (left/recv direction)
 *
 * Reduce-scatter phase (TP-1 = 3 rounds):
 *   Each GPU sends one chunk to right neighbor, accumulates received chunk
 *   After TP-1 rounds: each GPU[i] has fully-reduced chunk[(i+1)%TP]
 *
 * All-gather phase (TP-1 = 3 rounds):
 *   Each GPU sends its fully-reduced chunk to right neighbor.
 *   After TP-1 rounds: all GPUs have the complete result.
 *
 * FP32 scratch buffers are used for intermediate accumulation to avoid
 * FP16 rounding errors from multi-round accumulation. The send buffers
 * (used for P2P copies) are FP16 (converted from FP32 just before sending),
 * which halves the P2P bandwidth requirement vs sending FP32.
 *
 * For hidden_size=5120 (10KB): 4 chunks of 1280 FP16 elements (2560 bytes each).
 *
 * Target: gfx906 (MI50/MI60), compiled as SHARED LIBRARY with:
 *   hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC
 *
 * Exposes host-callable C functions for use from Python ctypes.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/**
 * ring_fp16_to_fp32_chunk:
 * Convert FP16 chunk to FP32 scratch buffer.
 * fp32_out[i] = (float)fp16_in[i]
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void ring_fp16_to_fp32_chunk_kernel(
    float* __restrict__ fp32_out,
    const __half* __restrict__ fp16_in,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp32_out[idx] = __half2float(fp16_in[idx]);
    }
}

/**
 * ring_fp32_accumulate_fp16:
 * Accumulate FP16 received chunk into FP32 scratch buffer.
 * fp32_acc[i] += (float)fp16_recv[i]
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void ring_fp32_accumulate_fp16_kernel(
    float* __restrict__ fp32_acc,
    const __half* __restrict__ fp16_recv,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp32_acc[idx] += __half2float(fp16_recv[idx]);
    }
}

/**
 * ring_fp32_to_fp16_chunk:
 * Convert FP32 scratch back to FP16 chunk.
 * fp16_out[i] = (__half)fp32_in[i]
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void ring_fp32_to_fp16_chunk_kernel(
    __half* __restrict__ fp16_out,
    const float* __restrict__ fp32_in,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp16_out[idx] = __float2half(fp32_in[idx]);
    }
}

/**
 * ring_fp32_copy:
 * Copy FP32 chunk from src to dst.
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void ring_fp32_copy_kernel(
    float* __restrict__ fp32_dst,
    const float* __restrict__ fp32_src,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp32_dst[idx] = fp32_src[idx];
    }
}

/**
 * ring_residual_add_fp32_to_fp16:
 * Add FP32 result to FP16 hidden buffer (residual connection).
 * hidden[i] = __float2half(__half2float(hidden[i]) + fp32_result[i])
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void ring_residual_add_fp32_to_fp16_kernel(
    __half* __restrict__ hidden,
    const float* __restrict__ fp32_result,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float h = __half2float(hidden[idx]);
        hidden[idx] = __float2half(h + fp32_result[idx]);
    }
}

/**
 * ring_fp32_to_fp16_result:
 * Convert full FP32 result buffer to FP16 (for allreduce_sum output).
 * Same as ring_fp32_to_fp16_chunk_kernel but operates on full hidden_size.
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void ring_fp32_to_fp16_result_kernel(
    __half* __restrict__ fp16_out,
    const float* __restrict__ fp32_in,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp16_out[idx] = __float2half(fp32_in[idx]);
    }
}

/* ===================================================================
 * Host-callable C wrapper functions.
 * =================================================================== */

extern "C" int ring_fp16_to_fp32_chunk(
    void* fp32_out, const void* fp16_in, unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = (n + 255) / 256;
    hipLaunchKernelGGL(ring_fp16_to_fp32_chunk_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (float*)fp32_out, (const __half*)fp16_in, n);
    return (int)hipGetLastError();
}

extern "C" int ring_fp32_accumulate_fp16(
    void* fp32_acc, const void* fp16_recv, unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = (n + 255) / 256;
    hipLaunchKernelGGL(ring_fp32_accumulate_fp16_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (float*)fp32_acc, (const __half*)fp16_recv, n);
    return (int)hipGetLastError();
}

extern "C" int ring_fp32_to_fp16_chunk(
    void* fp16_out, const void* fp32_in, unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = (n + 255) / 256;
    hipLaunchKernelGGL(ring_fp32_to_fp16_chunk_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)fp16_out, (const float*)fp32_in, n);
    return (int)hipGetLastError();
}

extern "C" int ring_fp32_copy(
    void* dst, const void* src, unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = (n + 255) / 256;
    hipLaunchKernelGGL(ring_fp32_copy_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (float*)dst, (const float*)src, n);
    return (int)hipGetLastError();
}

extern "C" int ring_residual_add_fp32_to_fp16(
    void* hidden, const void* fp32_result, unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = (n + 255) / 256;
    hipLaunchKernelGGL(ring_residual_add_fp32_to_fp16_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)hidden, (const float*)fp32_result, n);
    return (int)hipGetLastError();
}

extern "C" int ring_fp32_to_fp16_result(
    void* fp16_out, const void* fp32_in, unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = (n + 255) / 256;
    hipLaunchKernelGGL(ring_fp32_to_fp16_result_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)fp16_out, (const float*)fp32_in, n);
    return (int)hipGetLastError();
}

/**
 * ring_fp32_accumulate_fp32:
 * Accumulate FP32 received chunk into FP32 accumulation buffer.
 * fp32_acc[i] += fp32_recv[i]
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void ring_fp32_accumulate_fp32_kernel(
    float* __restrict__ fp32_acc,
    const float* __restrict__ fp32_recv,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp32_acc[idx] += fp32_recv[idx];
    }
}

extern "C" int ring_fp32_accumulate_fp32(
    void* fp32_acc, const void* fp32_recv, unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = (n + 255) / 256;
    hipLaunchKernelGGL(ring_fp32_accumulate_fp32_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (float*)fp32_acc, (const float*)fp32_recv, n);
    return (int)hipGetLastError();
}
