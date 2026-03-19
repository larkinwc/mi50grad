/**
 * P2P Bandwidth Benchmark Kernels for PCIe 4.0 Analysis
 * 
 * Tests different access patterns and vector widths for P2P memory reads
 * on gfx906 (MI50) architecture.
 * 
 * Hardware: PCIe 4.0 x16, 2-hop via switch
 * Theoretical bandwidth: ~25.6 GB/s
 * Measured P2P (2-hop): ~12 GB/s effective
 */

#include <hip/hip_runtime.h>
#include <cstdint>

// ===================================================================
// Sequential Coalesced Access Kernels
// ===================================================================

/**
 * Sequential access with float (4 bytes) loads
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_read_sequential_float(
    float* __restrict__ output,
    const float* __restrict__ peer_data,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    for (unsigned int i = idx; i < num_elements; i += stride) {
        float val = peer_data[i];
        if (output) output[i] = val;
    }
}

/**
 * Sequential access with float2 (8 bytes) loads
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_read_sequential_float2(
    float2* __restrict__ output,
    const float2* __restrict__ peer_data,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    for (unsigned int i = idx; i < num_elements; i += stride) {
        float2 val = peer_data[i];
        if (output) output[i] = val;
    }
}

/**
 * Sequential access with float4 (16 bytes) loads
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_read_sequential_float4(
    float4* __restrict__ output,
    const float4* __restrict__ peer_data,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    for (unsigned int i = idx; i < num_elements; i += stride) {
        float4 val = peer_data[i];
        if (output) output[i] = val;
    }
}

/**
 * Sequential access with uint4/dwordx4 (16 bytes) loads
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_read_sequential_dwordx4(
    uint4* __restrict__ output,
    const uint4* __restrict__ peer_data,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    for (unsigned int i = idx; i < num_elements; i += stride) {
        uint4 val = peer_data[i];
        if (output) output[i] = val;
    }
}

// ===================================================================
// Strided Access Kernels
// ===================================================================

/**
 * Strided access (stride=64 elements) with float loads
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_read_strided_float(
    float* __restrict__ output,
    const float* __restrict__ peer_data,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int stride_access = 64;
    
    for (unsigned int i = idx * stride_access; i < num_elements; i += stride * stride_access) {
        float val = peer_data[i];
        if (output) output[i] = val;
    }
}

/**
 * Strided access with float4 loads
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_read_strided_float4(
    float4* __restrict__ output,
    const float4* __restrict__ peer_data,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int stride_access = 16;  // 16 float4 = 64 floats
    
    for (unsigned int i = idx * stride_access; i < num_elements; i += stride * stride_access) {
        float4 val = peer_data[i];
        if (output) output[i] = val;
    }
}

// ===================================================================
// Random Access Kernels (latency-bound)
// ===================================================================

/**
 * Random access with float loads (LCG-based pseudo-random)
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_read_random_float(
    float* __restrict__ output,
    const float* __restrict__ peer_data,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    unsigned int state = idx + blockIdx.x * blockDim.x + 12345u;
    unsigned int count = 0;
    
    for (unsigned int i = 0; i < num_elements / stride; i++) {
        state = state * 1664525u + 1013904223u;
        unsigned int rand_idx = state % num_elements;
        float val = peer_data[rand_idx];
        if (output) output[count++] = val;
    }
}

/**
 * Random access with float4 loads
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_read_random_float4(
    float4* __restrict__ output,
    const float4* __restrict__ peer_data,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    unsigned int state = idx + blockIdx.x * blockDim.x + 12345u;
    unsigned int count = 0;
    
    for (unsigned int i = 0; i < num_elements / stride; i++) {
        state = state * 1664525u + 1013904223u;
        unsigned int rand_idx = (state % num_elements) & ~3u;  // Align to 4
        float4 val = peer_data[rand_idx];
        if (output) output[count++] = val;
    }
}

// ===================================================================
// Host-callable wrappers
// ===================================================================

extern "C" int p2p_read_sequential_float(
    void* output,
    const void* peer_data,
    unsigned int num_elements,
    hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = 256;
    hipLaunchKernelGGL(p2p_read_sequential_float, dim3(blocks), dim3(256), 0, stream,
                       (float*)output, (const float*)peer_data, num_elements);
    return (int)hipGetLastError();
}

extern "C" int p2p_read_sequential_float2(
    void* output,
    const void* peer_data,
    unsigned int num_elements,
    hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = 256;
    hipLaunchKernelGGL(p2p_read_sequential_float2, dim3(blocks), dim3(256), 0, stream,
                       (float2*)output, (const float2*)peer_data, num_elements);
    return (int)hipGetLastError();
}

extern "C" int p2p_read_sequential_float4(
    void* output,
    const void* peer_data,
    unsigned int num_elements,
    hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = 256;
    hipLaunchKernelGGL(p2p_read_sequential_float4, dim3(blocks), dim3(256), 0, stream,
                       (float4*)output, (const float4*)peer_data, num_elements);
    return (int)hipGetLastError();
}

extern "C" int p2p_read_sequential_dwordx4(
    void* output,
    const void* peer_data,
    unsigned int num_elements,
    hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = 256;
    hipLaunchKernelGGL(p2p_read_sequential_dwordx4, dim3(blocks), dim3(256), 0, stream,
                       (uint4*)output, (const uint4*)peer_data, num_elements);
    return (int)hipGetLastError();
}

extern "C" int p2p_read_strided_float(
    void* output,
    const void* peer_data,
    unsigned int num_elements,
    hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = 256;
    hipLaunchKernelGGL(p2p_read_strided_float, dim3(blocks), dim3(256), 0, stream,
                       (float*)output, (const float*)peer_data, num_elements);
    return (int)hipGetLastError();
}

extern "C" int p2p_read_strided_float4(
    void* output,
    const void* peer_data,
    unsigned int num_elements,
    hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = 256;
    hipLaunchKernelGGL(p2p_read_strided_float4, dim3(blocks), dim3(256), 0, stream,
                       (float4*)output, (const float4*)peer_data, num_elements);
    return (int)hipGetLastError();
}

extern "C" int p2p_read_random_float(
    void* output,
    const void* peer_data,
    unsigned int num_elements,
    hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = 256;
    hipLaunchKernelGGL(p2p_read_random_float, dim3(blocks), dim3(256), 0, stream,
                       (float*)output, (const float*)peer_data, num_elements);
    return (int)hipGetLastError();
}

extern "C" int p2p_read_random_float4(
    void* output,
    const void* peer_data,
    unsigned int num_elements,
    hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = 256;
    hipLaunchKernelGGL(p2p_read_random_float4, dim3(blocks), dim3(256), 0, stream,
                       (float4*)output, (const float4*)peer_data, num_elements);
    return (int)hipGetLastError();
}
