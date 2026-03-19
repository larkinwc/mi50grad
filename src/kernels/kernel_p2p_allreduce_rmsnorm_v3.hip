/**
 * Optimized Fused P2P Allreduce + RMSNorm Kernel (v3)
 * 
 * PROFILING-BASED OPTIMIZATIONS (from m2-profile-pcie4-bandwidth):
 * 
 * Key Finding: 10KB allreduce is LATENCY-BOUND, not bandwidth-bound
 * - Kernel launch overhead: 12.20 µs (fixed cost)
 * - Total 10KB latency (benchmark): 20.37 µs
 * - Current full allreduce: ~79 µs (59 µs overhead to eliminate)
 * - Vectorization provides NO benefit for small payloads
 * - Access pattern is irrelevant (sequential == random)
 * 
 * Optimizations applied in v3:
 *   1. Single-wavefront design (64 threads instead of 128/256)
 *      - Eliminates cross-wavefront synchronization
 *      - Reduces warp scheduling overhead
 *      - Each thread does 2x more work (80 elements vs 40)
 *   
 *   2. Reduced synchronization barriers
 *      - v2: 2 __syncthreads() barriers
 *      - v3: 1 __syncthreads() barrier (single warp, no cross-warp LDS needed)
 *   
 *   3. Optimized thread configuration
 *      - 64 threads (exactly 1 wavefront)
 *      - 80 FP16 elements per thread for hidden_size=5120
 *      - Total coverage: 64 × 80 = 5120 elements (perfect fit)
 *   
 *   4. Minimized instruction overhead
 *      - Use __shfl_sync for intra-wavefront reduction (no LDS)
 *      - Wave leader computes total, broadcasts via __shfl
 *      - Eliminates LDS latency and barriers
 *   
 *   5. Streamlined memory access
 *      - Simple scalar loads (vectorization doesn't help for latency-bound)
 *      - No complex addressing logic
 *      - Direct accumulation into registers
 *
 * Target: gfx906 (MI50/MI60)
 * Compilation: hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC
 *
 * Performance target: <= 40us per call (vs ~75us v2, ~79us baseline)
 * Correctness: max_abs_error < 5e-3 vs reference
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/* ===================================================================
 * Optimized GPU kernel: fused P2P allreduce + RMSNorm v3
 * Single-wavefront design for latency-bound 10KB payloads
 * =================================================================== */

/**
 * kernel_p2p_allreduce_rmsnorm_tp4_v3_kernel:
 *
 * OPTIMIZATIONS:
 *   - 64 threads (single wavefront) instead of 128/256
 *   - 80 elements per thread (vs 16 in v2, 40 in v1)
 *   - 1 __syncthreads() barrier (vs 2 in v2/v1)
 *   - No LDS usage - pure register + __shfl reduction
 *   - Streamlined code path for minimal instruction overhead
 *
 * PERFECT FIT FOR hidden_size=5120:
 *   64 threads × 80 elements = 5120 total (no scalar tail needed)
 *
 * Grid: (batch_size, 1, 1), Block: (64, 1, 1)
 */
__attribute__((amdgpu_flat_work_group_size(64, 64)))
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
    const unsigned int tid = threadIdx.x;
    const unsigned int tid80 = tid * 80;  // Each thread processes 80 elements (5120/64=80)
    const unsigned int lane = tid;  // tid == lane for single-wavefront
    
    // Base pointers for this batch item
    const __half* h_ptr   = hidden + blockIdx.x * dim;
    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    const __half* p_peer1 = partial_peer1 + blockIdx.x * dim;
    const __half* p_peer2 = partial_peer2 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;
    
    // ================================================================
    // Phase 1: Reduce all partials + compute sum-of-squares
    // Each thread processes 80 elements (stride = 64*80 = 5120)
    // For dim=5120: single iteration covers all elements
    // ================================================================
    float sum_sq = 0.0f;
    
    // Main loop: 80 elements per iteration
    // stride = num_threads * elements_per_thread = 64 * 80 = 5120
    for (unsigned int base_idx = tid80; base_idx < dim; base_idx += 5120) {
        #pragma unroll 80
        for (unsigned int k = 0; k < 80 && (base_idx + k) < dim; k++) {
            unsigned int idx = base_idx + k;
            
            // Load from all 5 buffers (hidden + 4 partials)
            float h  = __half2float(h_ptr[idx]);
            float pl = __half2float(p_local[idx]);
            float p0 = __half2float(p_peer0[idx]);
            float p1 = __half2float(p_peer1[idx]);
            float p2 = __half2float(p_peer2[idx]);
            
            // FP32 accumulation
            float sum = h + pl + p0 + p1 + p2;
            
            // Store reduced sum temporarily
            out[idx] = __float2half(sum);
            
            // Accumulate sum-of-squares
            sum_sq += sum * sum;
        }
    }
    
    // ================================================================
    // Phase 2: Intra-wavefront reduction (single warp, no LDS)
    // 6-step butterfly reduction across 64 lanes
    // ================================================================
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);
    
    // ================================================================
    // Phase 3: Wave leader computes total, broadcasts via __shfl
    // No LDS needed, no __syncthreads() for cross-warp sync
    // ================================================================
    if (lane == 0) {
        sum_sq = __shfl(sum_sq, 0);  // lane 0 already has total
    }
    
    // Compute inverse RMS
    float rms_inv = rsqrtf(sum_sq / (float)dim + eps);
    
    // ================================================================
    // Phase 4: Apply RMSNorm + weight
    // Each thread processes same 80 elements as Phase 1
    // ================================================================
    for (unsigned int base_idx = tid80; base_idx < dim; base_idx += 5120) {
        #pragma unroll 80
        for (unsigned int k = 0; k < 80 && (base_idx + k) < dim; k++) {
            unsigned int idx = base_idx + k;
            
            // Load reduced sum (stored in Phase 1) and weight
            float val = __half2float(out[idx]);
            float w   = __half2float(weight[idx]);
            
            // Apply RMSNorm and weight
            out[idx] = __float2half(val * rms_inv * w);
        }
    }
}

/**
 * kernel_p2p_allreduce_rmsnorm_tp2_v3_kernel:
 * TP=2 variant (same single-wavefront optimizations)
 * 64 threads × 80 elements = 5120 (perfect fit for hidden_size=5120)
 */
__attribute__((amdgpu_flat_work_group_size(64, 64)))
__global__ void kernel_p2p_allreduce_rmsnorm_tp2_v3_kernel(
    __half* __restrict__ output,
    const __half* hidden,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ weight,
    unsigned int dim,
    float eps
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tid80 = tid * 80;  // Each thread processes 80 elements (5120/64=80)
    const unsigned int lane = tid;
    
    const __half* h_ptr   = hidden + blockIdx.x * dim;
    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;
    
    float sum_sq = 0.0f;
    
    for (unsigned int base_idx = tid80; base_idx < dim; base_idx += 5120) {
        #pragma unroll 80
        for (unsigned int k = 0; k < 80 && (base_idx + k) < dim; k++) {
            unsigned int idx = base_idx + k;
            
            float h  = __half2float(h_ptr[idx]);
            float pl = __half2float(p_local[idx]);
            float p0 = __half2float(p_peer0[idx]);
            
            float sum = h + pl + p0;
            out[idx] = __float2half(sum);
            sum_sq += sum * sum;
        }
    }
    
    // Intra-wavefront reduction
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);
    
    if (lane == 0) {
        sum_sq = __shfl(sum_sq, 0);
    }
    
    float rms_inv = rsqrtf(sum_sq / (float)dim + eps);
    
    for (unsigned int base_idx = tid80; base_idx < dim; base_idx += 5120) {
        #pragma unroll 80
        for (unsigned int k = 0; k < 80 && (base_idx + k) < dim; k++) {
            unsigned int idx = base_idx + k;
            
            float val = __half2float(out[idx]);
            float w   = __half2float(weight[idx]);
            
            out[idx] = __float2half(val * rms_inv * w);
        }
    }
}

/**
 * kernel_p2p_allreduce_rmsnorm_generic_v3_kernel:
 * Generic variant for non-standard hidden sizes
 * Uses dynamic loop bounds instead of hardcoded 80 elements/thread
 */
__attribute__((amdgpu_flat_work_group_size(64, 64)))
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
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid;
    
    const __half* h_ptr   = hidden + blockIdx.x * dim;
    const __half* p_local = partial_local + blockIdx.x * dim;
    const __half* p_peer0 = partial_peer0 + blockIdx.x * dim;
    const __half* p_peer1 = partial_peer1 + blockIdx.x * dim;
    const __half* p_peer2 = partial_peer2 + blockIdx.x * dim;
    __half* out           = output + blockIdx.x * dim;
    
    float sum_sq = 0.0f;
    
    // Calculate elements per thread for this specific dim
    // elements_per_thread = ceil(dim / 64)
    unsigned int elements_per_thread = (dim + 63) / 64;
    
    for (unsigned int k = 0; k < elements_per_thread; k++) {
        unsigned int idx = tid + k * 64;
        if (idx >= dim) break;
        
        float h  = __half2float(h_ptr[idx]);
        float pl = __half2float(p_local[idx]);
        float p0 = __half2float(p_peer0[idx]);
        float p1 = __half2float(p_peer1[idx]);
        float p2 = __half2float(p_peer2[idx]);
        
        float sum = h + pl + p0 + p1 + p2;
        out[idx] = __float2half(sum);
        sum_sq += sum * sum;
    }
    
    // Intra-wavefront reduction
    sum_sq += __shfl_xor(sum_sq, 32);
    sum_sq += __shfl_xor(sum_sq, 16);
    sum_sq += __shfl_xor(sum_sq, 8);
    sum_sq += __shfl_xor(sum_sq, 4);
    sum_sq += __shfl_xor(sum_sq, 2);
    sum_sq += __shfl_xor(sum_sq, 1);
    
    if (lane == 0) {
        sum_sq = __shfl(sum_sq, 0);
    }
    
    float rms_inv = rsqrtf(sum_sq / (float)dim + eps);
    
    for (unsigned int k = 0; k < elements_per_thread; k++) {
        unsigned int idx = tid + k * 64;
        if (idx >= dim) break;
        
        float val = __half2float(out[idx]);
        float w   = __half2float(weight[idx]);
        
        out[idx] = __float2half(val * rms_inv * w);
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
                       dim3(batch_size), dim3(64), 0, stream,
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
                       dim3(batch_size), dim3(64), 0, stream,
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
                       dim3(batch_size), dim3(64), 0, stream,
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
