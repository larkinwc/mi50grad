/**
 * Optimized Fused P2P Allreduce + RMSNorm Kernel (v3)
 * 
 * FIXED: Memory access pattern now uses interleaved stride for coalescing.
 * Uses 256 threads (4 wavefronts) to match v1 baseline.
 * 
 * Key optimizations:
 *   1. 256 threads (4 wavefronts) - same as v1
 *   2. Interleaved memory access (stride 256) for coalescing
 *   3. Vectorized float4 loads (8 FP16 at a time)
 *   4. 2 __syncthreads() barriers (same as v1)
 *   5. Optimized LDS reduction
 *
 * Memory access pattern:
 *   Thread t accesses elements at: t*8, t*8+256, t*8+512, ...
 *   Adjacent threads access adjacent 8-element chunks.
 *   For hidden_size=5120: 256 threads × 20 elements = 5120 (perfect fit)
 *
 * Target: gfx906 (MI50/MI60)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/* ===================================================================
 * Optimized GPU kernel: fused P2P allreduce + RMSNorm v3
 * Four-wavefront design with interleaved memory access (matches v1)
 * =================================================================== */

/**
 * kernel_p2p_allreduce_rmsnorm_tp4_v3_kernel:
 *
 * Grid: (batch_size, 1, 1), Block: (256, 1, 1)
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_p2p_allreduce_rmsnorm_tp4_v3_kernel(
    __half* __restrict__ output,
    const __half* hidden,
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
    
    // Base pointers for this batch item
    const __half* h_ptr   = hidden + blockIdx.x * dim;
    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    const __half* p_peer1 = partial_peer1 + blockIdx.x * dim;
    const __half* p_peer2 = partial_peer2 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;
    
    // ================================================================
    // Phase 1: Reduce all partials + compute sum-of-squares
    // Interleaved access: thread t handles {t*8, t*8+256, t*8+512, ...}
    // Uses vectorized float4 loads (8 FP16 at a time)
    // ================================================================
    float sum_sq = 0.0f;
    
    // Vectorized path: each thread processes 8-element chunks with stride 256*8=2048
    // For dim=5120: 5120/2048 = 2.5, so 2 full iterations + tail
    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        // Vectorized loads (float4 = 128-bit = 8 FP16)
        float4 h4  = *reinterpret_cast<const float4*>(h_ptr + i);
        float4 pl4 = *reinterpret_cast<const float4*>(p_local + i);
        float4 p04 = *reinterpret_cast<const float4*>(p_peer0 + i);
        float4 p14 = *reinterpret_cast<const float4*>(p_peer1 + i);
        float4 p24 = *reinterpret_cast<const float4*>(p_peer2 + i);
        
        const __half2* h_h2  = reinterpret_cast<const __half2*>(&h4);
        const __half2* pl_h2 = reinterpret_cast<const __half2*>(&pl4);
        const __half2* p0_h2 = reinterpret_cast<const __half2*>(&p04);
        const __half2* p1_h2 = reinterpret_cast<const __half2*>(&p14);
        const __half2* p2_h2 = reinterpret_cast<const __half2*>(&p24);
        
        float4 sum4;
        __half2* sum_h2 = reinterpret_cast<__half2*>(&sum4);
        
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float v0 = __half2float(__low2half (h_h2[k]))
                     + __half2float(__low2half (pl_h2[k]))
                     + __half2float(__low2half (p0_h2[k]))
                     + __half2float(__low2half (p1_h2[k]))
                     + __half2float(__low2half (p2_h2[k]));
            float v1 = __half2float(__high2half(h_h2[k]))
                     + __half2float(__high2half(pl_h2[k]))
                     + __half2float(__high2half(p0_h2[k]))
                     + __half2float(__high2half(p1_h2[k]))
                     + __half2float(__high2half(p2_h2[k]));
            
            sum_h2[k] = __halves2half2(__float2half(v0), __float2half(v1));
            sum_sq += v0 * v0 + v1 * v1;
        }
        
        *reinterpret_cast<float4*>(out + i) = sum4;
    }
    
    // Scalar tail (for non-multiple-of-8 dims)
    {
        unsigned int vec_dim = (dim / 8) * 8;
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float v = __half2float(h_ptr[i])
                    + __half2float(p_local[i])
                    + __half2float(p_peer0[i])
                    + __half2float(p_peer1[i])
                    + __half2float(p_peer2[i]);
            out[i] = __float2half(v);
            sum_sq += v * v;
        }
    }
    
    // ================================================================
    // Phase 2: Intra-wavefront reduction (shuffle)
    // 6-step butterfly reduction across 64 lanes within each wavefront
    // ================================================================
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);
    
    // ================================================================
    // Phase 3: Cross-wavefront reduction via LDS (2 barriers)
    // ================================================================
    __shared__ float s_wf[4];
    if (lane == 0) s_wf[wf_id] = sum_sq;
    __syncthreads();
    
    if (threadIdx.x < 4) {
        float total = s_wf[0] + s_wf[1] + s_wf[2] + s_wf[3];
        s_wf[threadIdx.x] = total;
    }
    __syncthreads();
    float total_sq = s_wf[0];
    
    // Compute inverse RMS
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);
    
    // ================================================================
    // Phase 4: Apply RMSNorm + weight
    // Same interleaved access pattern as Phase 1
    // ================================================================
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

/**
 * kernel_p2p_allreduce_rmsnorm_tp2_v3_kernel:
 * TP=2 variant (same 256-thread design)
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_p2p_allreduce_rmsnorm_tp2_v3_kernel(
    __half* __restrict__ output,
    const __half* hidden,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ weight,
    unsigned int dim,
    float eps
) {
    unsigned int wf_id = threadIdx.x / 64;
    unsigned int lane   = threadIdx.x % 64;
    
    const __half* h_ptr   = hidden + blockIdx.x * dim;
    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;
    
    float sum_sq = 0.0f;
    
    // Interleaved access with vectorized loads (stride 256)
    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        float4 h4  = *reinterpret_cast<const float4*>(h_ptr + i);
        float4 pl4 = *reinterpret_cast<const float4*>(p_local + i);
        float4 p04 = *reinterpret_cast<const float4*>(p_peer0 + i);
        
        const __half2* h_h2  = reinterpret_cast<const __half2*>(&h4);
        const __half2* pl_h2 = reinterpret_cast<const __half2*>(&pl4);
        const __half2* p0_h2 = reinterpret_cast<const __half2*>(&p04);
        
        float4 sum4;
        __half2* sum_h2 = reinterpret_cast<__half2*>(&sum4);
        
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float v0 = __half2float(__low2half (h_h2[k]))
                     + __half2float(__low2half (pl_h2[k]))
                     + __half2float(__low2half (p0_h2[k]));
            float v1 = __half2float(__high2half(h_h2[k]))
                     + __half2float(__high2half(pl_h2[k]))
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
            float v = __half2float(h_ptr[i])
                    + __half2float(p_local[i])
                    + __half2float(p_peer0[i]);
            out[i] = __float2half(v);
            sum_sq += v * v;
        }
    }
    
    // Intra-wavefront reduction
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);
    
    // Cross-wavefront reduction
    __shared__ float s_wf[4];
    if (lane == 0) s_wf[wf_id] = sum_sq;
    __syncthreads();
    
    if (threadIdx.x < 4) {
        float total = s_wf[0] + s_wf[1] + s_wf[2] + s_wf[3];
        s_wf[threadIdx.x] = total;
    }
    __syncthreads();
    float total_sq = s_wf[0];
    
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);
    
    // Phase 4: Apply RMSNorm (stride 256)
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

/**
 * kernel_p2p_allreduce_rmsnorm_generic_v3_kernel:
 * Generic variant for non-standard hidden sizes (256 threads)
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_p2p_allreduce_rmsnorm_generic_v3_kernel(
    __half* __restrict__ output,
    const __half* hidden,
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
    
    const __half* h_ptr   = hidden + blockIdx.x * dim;
    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    const __half* p_peer1 = partial_peer1 + blockIdx.x * dim;
    const __half* p_peer2 = partial_peer2 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;
    
    float sum_sq = 0.0f;
    
    // Interleaved access with vectorized loads (stride 256)
    for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
        float4 h4  = *reinterpret_cast<const float4*>(h_ptr + i);
        float4 pl4 = *reinterpret_cast<const float4*>(p_local + i);
        float4 p04 = *reinterpret_cast<const float4*>(p_peer0 + i);
        float4 p14 = *reinterpret_cast<const float4*>(p_peer1 + i);
        float4 p24 = *reinterpret_cast<const float4*>(p_peer2 + i);
        
        const __half2* h_h2  = reinterpret_cast<const __half2*>(&h4);
        const __half2* pl_h2 = reinterpret_cast<const __half2*>(&pl4);
        const __half2* p0_h2 = reinterpret_cast<const __half2*>(&p04);
        const __half2* p1_h2 = reinterpret_cast<const __half2*>(&p14);
        const __half2* p2_h2 = reinterpret_cast<const __half2*>(&p24);
        
        float4 sum4;
        __half2* sum_h2 = reinterpret_cast<__half2*>(&sum4);
        
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float v0 = __half2float(__low2half (h_h2[k]))
                     + __half2float(__low2half (pl_h2[k]))
                     + __half2float(__low2half (p0_h2[k]))
                     + __half2float(__low2half (p1_h2[k]))
                     + __half2float(__low2half (p2_h2[k]));
            float v1 = __half2float(__high2half(h_h2[k]))
                     + __half2float(__high2half(pl_h2[k]))
                     + __half2float(__high2half(p0_h2[k]))
                     + __half2float(__high2half(p1_h2[k]))
                     + __half2float(__high2half(p2_h2[k]));
            
            sum_h2[k] = __halves2half2(__float2half(v0), __float2half(v1));
            sum_sq += v0 * v0 + v1 * v1;
        }
        
        *reinterpret_cast<float4*>(out + i) = sum4;
    }
    
    // Scalar tail
    {
        unsigned int vec_dim = (dim / 8) * 8;
        for (unsigned int i = vec_dim + threadIdx.x; i < dim; i += 256) {
            float v = __half2float(h_ptr[i])
                    + __half2float(p_local[i])
                    + __half2float(p_peer0[i])
                    + __half2float(p_peer1[i])
                    + __half2float(p_peer2[i]);
            out[i] = __float2half(v);
            sum_sq += v * v;
        }
    }
    
    // Intra-wavefront reduction
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);
    
    // Cross-wavefront reduction
    __shared__ float s_wf[4];
    if (lane == 0) s_wf[wf_id] = sum_sq;
    __syncthreads();
    
    if (threadIdx.x < 4) {
        float total = s_wf[0] + s_wf[1] + s_wf[2] + s_wf[3];
        s_wf[threadIdx.x] = total;
    }
    __syncthreads();
    float total_sq = s_wf[0];
    
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);
    
    // Phase 4: Apply RMSNorm with interleaved access (stride 256)
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
 * Host-callable C wrappers for Python ctypes dispatch
 * =================================================================== */

extern "C" int kernel_p2p_allreduce_rmsnorm_tp4_v3(
    void* output,
    const void* hidden,
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
    (void)hipGetLastError();
    hipLaunchKernelGGL(kernel_p2p_allreduce_rmsnorm_tp4_v3_kernel,
                       dim3(batch_size), dim3(256), 0, stream,
                       (__half*)output,
                       (const __half*)hidden,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)partial_peer1,
                       (const __half*)partial_peer2,
                       (const __half*)weight,
                       dim, eps);
    return (int)hipGetLastError();
}

extern "C" int kernel_p2p_allreduce_rmsnorm_tp2_v3(
    void* output,
    const void* hidden,
    const void* partial_local,
    const void* partial_peer0,
    const void* weight,
    unsigned int dim,
    unsigned int batch_size,
    float eps,
    hipStream_t stream)
{
    (void)hipGetLastError();
    hipLaunchKernelGGL(kernel_p2p_allreduce_rmsnorm_tp2_v3_kernel,
                       dim3(batch_size), dim3(256), 0, stream,
                       (__half*)output,
                       (const __half*)hidden,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)weight,
                       dim, eps);
    return (int)hipGetLastError();
}

extern "C" int kernel_p2p_allreduce_rmsnorm_generic_v3(
    void* output,
    const void* hidden,
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
    (void)hipGetLastError();
    hipLaunchKernelGGL(kernel_p2p_allreduce_rmsnorm_generic_v3_kernel,
                       dim3(batch_size), dim3(256), 0, stream,
                       (__half*)output,
                       (const __half*)hidden,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)partial_peer1,
                       (const __half*)partial_peer2,
                       (const __half*)weight,
                       dim, eps);
    return (int)hipGetLastError();
}
