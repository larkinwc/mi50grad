/**
 * Optimized Fused P2P Allreduce + RMSNorm Kernel (v2)
 *
 * Optimizations applied:
 *   1. Increased elements per thread: 16 FP16 elements/thread (vs 8 in v1)
 *      - Reduces thread count from 256 to 128 for hidden_size=5120
 *      - Fewer warps = fewer synchronization points
 *   2. Optimized memory access: explicit dwordx4 (128-bit) loads with alignment
 *      - Ensures coalesced 128-byte reads from peer BAR1 memory
 *   3. Minimized LDS usage: only 8 bytes (2 floats for warp sums, vs 16 in v1)
 *      - Reduced from 4 warps to 2 warps
 *   4. Improved instruction-level parallelism in accumulation loop
 *
 * Note: This version maintains 2 __syncthreads() barriers (same as v1) but
 *       reduces overall synchronization overhead via fewer warps (2 vs 4).
 *
 * Target: gfx906 (MI50/MI60)
 * Compilation: hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC
 *
 * Performance target: <= 50us per call (vs ~79us baseline)
 * Correctness: max_abs_error < 1e-3 vs reference
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/* ===================================================================
 * Optimized GPU kernel: fused P2P allreduce + RMSNorm v2
 * =================================================================== */

/**
 * kernel_p2p_allreduce_rmsnorm_tp4_v2_kernel:
 *
 * Optimizations:
 *   - 128 threads (2 warps) instead of 256 (4 warps)
 *   - Each thread processes 16 FP16 elements (2x more than v1)
 *   - dwordx4 vectorized loads for coalesced 128-byte memory access
 *   - Reduced LDS usage: 8 bytes (2 floats) vs 16 bytes (4 floats) in v1
 *
 * Note: Uses 2 __syncthreads() barriers (same as v1) but with reduced
 *       warp count for lower synchronization overhead.
 *
 * Grid: (batch_size, 1, 1), Block: (128, 1, 1)
 * For hidden_size=5120: each thread processes 40 iterations
 */
__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void kernel_p2p_allreduce_rmsnorm_tp4_v2_kernel(
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
    const unsigned int tid = threadIdx.x;
    const unsigned int wf_id = tid / 64;  // 0 or 1 (2 warps total)
    const unsigned int lane = tid % 64;
    
    // Base pointers for this batch item
    const __half* h_ptr   = hidden + blockIdx.x * dim;
    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    const __half* p_peer1 = partial_peer1 + blockIdx.x * dim;
    const __half* p_peer2 = partial_peer2 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;
    
    // ================================================================
    // Phase 1: Reduce all partials + compute sum-of-squares
    // Each thread processes 16 elements per iteration (dwordx4 x 4)
    // ================================================================
    float sum_sq = 0.0f;
    
    // Main loop: 16 elements per iteration, stride = 128 * 16 = 2048
    // For dim=5120: threads 0-31 cover all elements in ~2-3 iterations
    for (unsigned int i = tid * 16; i + 15 < dim; i += 128 * 16) {
        // Load 16 elements from each buffer using 4x dwordx4 (128-bit) loads
        // This ensures coalesced 128-byte aligned reads from peer BAR1 memory
        
        // Hidden: 4x dwordx4 loads = 16 FP16 elements
        float4 h0 = *reinterpret_cast<const float4*>(h_ptr + i);
        float4 h1 = *reinterpret_cast<const float4*>(h_ptr + i + 8);
        
        // Partial local
        float4 pl0 = *reinterpret_cast<const float4*>(p_local + i);
        float4 pl1 = *reinterpret_cast<const float4*>(p_local + i + 8);
        
        // Peer 0
        float4 p00 = *reinterpret_cast<const float4*>(p_peer0 + i);
        float4 p01 = *reinterpret_cast<const float4*>(p_peer0 + i + 8);
        
        // Peer 1
        float4 p10 = *reinterpret_cast<const float4*>(p_peer1 + i);
        float4 p11 = *reinterpret_cast<const float4*>(p_peer1 + i + 8);
        
        // Peer 2
        float4 p20 = *reinterpret_cast<const float4*>(p_peer2 + i);
        float4 p21 = *reinterpret_cast<const float4*>(p_peer2 + i + 8);
        
        // Cast to half2 for FP32 accumulation
        const __half2* h_h2_0  = reinterpret_cast<const __half2*>(&h0);
        const __half2* h_h2_1  = reinterpret_cast<const __half2*>(&h1);
        const __half2* pl_h2_0 = reinterpret_cast<const __half2*>(&pl0);
        const __half2* pl_h2_1 = reinterpret_cast<const __half2*>(&pl1);
        const __half2* p0_h2_0 = reinterpret_cast<const __half2*>(&p00);
        const __half2* p0_h2_1 = reinterpret_cast<const __half2*>(&p01);
        const __half2* p1_h2_0 = reinterpret_cast<const __half2*>(&p10);
        const __half2* p1_h2_1 = reinterpret_cast<const __half2*>(&p11);
        const __half2* p2_h2_0 = reinterpret_cast<const __half2*>(&p20);
        const __half2* p2_h2_1 = reinterpret_cast<const __half2*>(&p21);
        
        // Output storage
        float4 out0, out1;
        __half2* out_h2_0 = reinterpret_cast<__half2*>(&out0);
        __half2* out_h2_1 = reinterpret_cast<__half2*>(&out1);
        
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            // Accumulate 5 inputs (hidden + 4 partials) in FP32
            float v0 = __half2float(__low2half (h_h2_0[k]))
                     + __half2float(__low2half (pl_h2_0[k]))
                     + __half2float(__low2half (p0_h2_0[k]))
                     + __half2float(__low2half (p1_h2_0[k]))
                     + __half2float(__low2half (p2_h2_0[k]));
            float v1 = __half2float(__high2half(h_h2_0[k]))
                     + __half2float(__high2half(pl_h2_0[k]))
                     + __half2float(__high2half(p0_h2_0[k]))
                     + __half2float(__high2half(p1_h2_0[k]))
                     + __half2float(__high2half(p2_h2_0[k]));
            
            out_h2_0[k] = __halves2half2(__float2half(v0), __float2half(v1));
            sum_sq += v0 * v0 + v1 * v1;
        }
        
        #pragma unroll
        for (int k = 4; k < 8; k++) {
            float v0 = __half2float(__low2half (h_h2_1[k-4]))
                     + __half2float(__low2half (pl_h2_1[k-4]))
                     + __half2float(__low2half (p0_h2_1[k-4]))
                     + __half2float(__low2half (p1_h2_1[k-4]))
                     + __half2float(__low2half (p2_h2_1[k-4]));
            float v1 = __half2float(__high2half(h_h2_1[k-4]))
                     + __half2float(__high2half(pl_h2_1[k-4]))
                     + __half2float(__high2half(p0_h2_1[k-4]))
                     + __half2float(__high2half(p1_h2_1[k-4]))
                     + __half2float(__high2half(p2_h2_1[k-4]));
            
            out_h2_1[k-4] = __halves2half2(__float2half(v0), __float2half(v1));
            sum_sq += v0 * v0 + v1 * v1;
        }
        
        // Store reduced sum (temporarily, will normalize in Phase 3)
        *reinterpret_cast<float4*>(out + i) = out0;
        *reinterpret_cast<float4*>(out + i + 8) = out1;
    }
    
    // Scalar tail: elements in [(dim/16)*16, dim-1]
    {
        unsigned int vec_dim = (dim / 16) * 16;
        for (unsigned int i = vec_dim + tid; i < dim; i += 128) {
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
    // Phase 2: Warp reduction (6-step butterfly for wave64)
    // ================================================================
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);
    
    // ================================================================
    // Phase 3: Cross-warp reduction via LDS (2 warps -> 1 barrier)
    // Warp leaders write partial sums to LDS, then thread 0 computes total
    // ================================================================
    __shared__ float s_warp[2];  // Only 2 warps (vs 4 in v1)
    
    // Warp leaders write to LDS
    if (lane == 0) {
        s_warp[wf_id] = sum_sq;
    }
    __syncthreads();  // Single barrier
    
    // Thread 0 computes total and broadcasts implicitly via LDS
    if (tid == 0) {
        float total = s_warp[0] + s_warp[1];
        s_warp[0] = total;
    }
    __syncthreads();  // Ensure total is visible (needed for correctness)
    float total_sq = s_warp[0];
    
    // Compute inverse RMS
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);
    
    // ================================================================
    // Phase 4: Apply RMSNorm + weight
    // ================================================================
    for (unsigned int i = tid * 16; i + 15 < dim; i += 128 * 16) {
        // Load reduced sum (stored in Phase 1)
        float4 sum0 = *reinterpret_cast<const float4*>(out + i);
        float4 sum1 = *reinterpret_cast<const float4*>(out + i + 8);
        
        // Load weight
        float4 w0 = *reinterpret_cast<const float4*>(weight + i);
        float4 w1 = *reinterpret_cast<const float4*>(weight + i + 8);
        
        const __half2* sum_h2_0 = reinterpret_cast<const __half2*>(&sum0);
        const __half2* sum_h2_1 = reinterpret_cast<const __half2*>(&sum1);
        const __half2* w_h2_0   = reinterpret_cast<const __half2*>(&w0);
        const __half2* w_h2_1   = reinterpret_cast<const __half2*>(&w1);
        
        float4 out0, out1;
        __half2* out_h2_0 = reinterpret_cast<__half2*>(&out0);
        __half2* out_h2_1 = reinterpret_cast<__half2*>(&out1);
        
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float s0 = __half2float(__low2half (sum_h2_0[k]));
            float s1 = __half2float(__high2half(sum_h2_0[k]));
            float w0v = __half2float(__low2half (w_h2_0[k]));
            float w1v = __half2float(__high2half(w_h2_0[k]));
            
            out_h2_0[k] = __halves2half2(
                __float2half(s0 * rms_inv * w0v),
                __float2half(s1 * rms_inv * w1v)
            );
        }
        
        #pragma unroll
        for (int k = 4; k < 8; k++) {
            float s0 = __half2float(__low2half (sum_h2_1[k-4]));
            float s1 = __half2float(__high2half(sum_h2_1[k-4]));
            float w0v = __half2float(__low2half (w_h2_1[k-4]));
            float w1v = __half2float(__high2half(w_h2_1[k-4]));
            
            out_h2_1[k-4] = __halves2half2(
                __float2half(s0 * rms_inv * w0v),
                __float2half(s1 * rms_inv * w1v)
            );
        }
        
        *reinterpret_cast<float4*>(out + i) = out0;
        *reinterpret_cast<float4*>(out + i + 8) = out1;
    }
    
    // Scalar tail
    {
        unsigned int vec_dim = (dim / 16) * 16;
        for (unsigned int i = vec_dim + tid; i < dim; i += 128) {
            float val = __half2float(out[i]);
            float w   = __half2float(weight[i]);
            out[i] = __float2half(val * rms_inv * w);
        }
    }
}

/**
 * kernel_p2p_allreduce_rmsnorm_tp2_v2_kernel:
 * Optimized TP=2 variant (same optimizations as TP=4)
 */
__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void kernel_p2p_allreduce_rmsnorm_tp2_v2_kernel(
    __half* __restrict__ output,
    const __half* hidden,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ weight,
    unsigned int dim,
    float eps
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int wf_id = tid / 64;
    const unsigned int lane = tid % 64;
    
    const __half* h_ptr   = hidden + blockIdx.x * dim;
    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;
    
    float sum_sq = 0.0f;
    
    for (unsigned int i = tid * 16; i + 15 < dim; i += 128 * 16) {
        float4 h0 = *reinterpret_cast<const float4*>(h_ptr + i);
        float4 h1 = *reinterpret_cast<const float4*>(h_ptr + i + 8);
        float4 pl0 = *reinterpret_cast<const float4*>(p_local + i);
        float4 pl1 = *reinterpret_cast<const float4*>(p_local + i + 8);
        float4 p00 = *reinterpret_cast<const float4*>(p_peer0 + i);
        float4 p01 = *reinterpret_cast<const float4*>(p_peer0 + i + 8);
        
        const __half2* h_h2_0  = reinterpret_cast<const __half2*>(&h0);
        const __half2* h_h2_1  = reinterpret_cast<const __half2*>(&h1);
        const __half2* pl_h2_0 = reinterpret_cast<const __half2*>(&pl0);
        const __half2* pl_h2_1 = reinterpret_cast<const __half2*>(&pl1);
        const __half2* p0_h2_0 = reinterpret_cast<const __half2*>(&p00);
        const __half2* p0_h2_1 = reinterpret_cast<const __half2*>(&p01);
        
        float4 out0, out1;
        __half2* out_h2_0 = reinterpret_cast<__half2*>(&out0);
        __half2* out_h2_1 = reinterpret_cast<__half2*>(&out1);
        
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float v0 = __half2float(__low2half (h_h2_0[k]))
                     + __half2float(__low2half (pl_h2_0[k]))
                     + __half2float(__low2half (p0_h2_0[k]));
            float v1 = __half2float(__high2half(h_h2_0[k]))
                     + __half2float(__high2half(pl_h2_0[k]))
                     + __half2float(__high2half(p0_h2_0[k]));
            
            out_h2_0[k] = __halves2half2(__float2half(v0), __float2half(v1));
            sum_sq += v0 * v0 + v1 * v1;
        }
        
        #pragma unroll
        for (int k = 4; k < 8; k++) {
            float v0 = __half2float(__low2half (h_h2_1[k-4]))
                     + __half2float(__low2half (pl_h2_1[k-4]))
                     + __half2float(__low2half (p0_h2_1[k-4]));
            float v1 = __half2float(__high2half(h_h2_1[k-4]))
                     + __half2float(__high2half(pl_h2_1[k-4]))
                     + __half2float(__high2half(p0_h2_1[k-4]));
            
            out_h2_1[k-4] = __halves2half2(__float2half(v0), __float2half(v1));
            sum_sq += v0 * v0 + v1 * v1;
        }
        
        *reinterpret_cast<float4*>(out + i) = out0;
        *reinterpret_cast<float4*>(out + i + 8) = out1;
    }
    
    // Scalar tail
    {
        unsigned int vec_dim = (dim / 16) * 16;
        for (unsigned int i = vec_dim + tid; i < dim; i += 128) {
            float v = __half2float(h_ptr[i])
                    + __half2float(p_local[i])
                    + __half2float(p_peer0[i]);
            out[i] = __float2half(v);
            sum_sq += v * v;
        }
    }
    
    // Warp reduction
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);
    
    // Cross-warp reduction
    __shared__ float s_warp[2];
    if (lane == 0) s_warp[wf_id] = sum_sq;
    __syncthreads();
    if (tid == 0) {
        float total = s_warp[0] + s_warp[1];
        s_warp[0] = total;
    }
    __syncthreads();
    float total_sq = s_warp[0];
    
    float rms_inv = rsqrtf(total_sq / (float)dim + eps);
    
    // Apply RMSNorm + weight
    for (unsigned int i = tid * 16; i + 15 < dim; i += 128 * 16) {
        float4 sum0 = *reinterpret_cast<const float4*>(out + i);
        float4 sum1 = *reinterpret_cast<const float4*>(out + i + 8);
        float4 w0 = *reinterpret_cast<const float4*>(weight + i);
        float4 w1 = *reinterpret_cast<const float4*>(weight + i + 8);
        
        const __half2* sum_h2_0 = reinterpret_cast<const __half2*>(&sum0);
        const __half2* sum_h2_1 = reinterpret_cast<const __half2*>(&sum1);
        const __half2* w_h2_0   = reinterpret_cast<const __half2*>(&w0);
        const __half2* w_h2_1   = reinterpret_cast<const __half2*>(&w1);
        
        float4 out0, out1;
        __half2* out_h2_0 = reinterpret_cast<__half2*>(&out0);
        __half2* out_h2_1 = reinterpret_cast<__half2*>(&out1);
        
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float s0 = __half2float(__low2half (sum_h2_0[k]));
            float s1 = __half2float(__high2half(sum_h2_0[k]));
            float w0v = __half2float(__low2half (w_h2_0[k]));
            float w1v = __half2float(__high2half(w_h2_0[k]));
            
            out_h2_0[k] = __halves2half2(
                __float2half(s0 * rms_inv * w0v),
                __float2half(s1 * rms_inv * w1v)
            );
        }
        
        #pragma unroll
        for (int k = 4; k < 8; k++) {
            float s0 = __half2float(__low2half (sum_h2_1[k-4]));
            float s1 = __half2float(__high2half(sum_h2_1[k-4]));
            float w0v = __half2float(__low2half (w_h2_1[k-4]));
            float w1v = __half2float(__high2half(w_h2_1[k-4]));
            
            out_h2_1[k-4] = __halves2half2(
                __float2half(s0 * rms_inv * w0v),
                __float2half(s1 * rms_inv * w1v)
            );
        }
        
        *reinterpret_cast<float4*>(out + i) = out0;
        *reinterpret_cast<float4*>(out + i + 8) = out1;
    }
    
    // Scalar tail
    {
        unsigned int vec_dim = (dim / 16) * 16;
        for (unsigned int i = vec_dim + tid; i < dim; i += 128) {
            float val = __half2float(out[i]);
            float w   = __half2float(weight[i]);
            out[i] = __float2half(val * rms_inv * w);
        }
    }
}

/* ===================================================================
 * Host-callable C wrappers for Python ctypes dispatch
 * =================================================================== */

extern "C" int kernel_p2p_allreduce_rmsnorm_tp4_v2(
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
    hipLaunchKernelGGL(kernel_p2p_allreduce_rmsnorm_tp4_v2_kernel,
                       dim3(batch_size), dim3(128), 0, stream,
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

extern "C" int kernel_p2p_allreduce_rmsnorm_tp2_v2(
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
    hipLaunchKernelGGL(kernel_p2p_allreduce_rmsnorm_tp2_v2_kernel,
                       dim3(batch_size), dim3(128), 0, stream,
                       (__half*)output,
                       (const __half*)hidden,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)weight,
                       dim, eps);
    return (int)hipGetLastError();
}
