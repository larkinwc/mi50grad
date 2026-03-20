/**
 * Kernel-based P2P allreduce with INT8 block-wise compression.
 *
 * Milestone: compressed-allreduce
 * Feature: compressed-ar-standalone-kernel
 *
 * This kernel reduces P2P transfer volume by ~50% using block-wise INT8 quantization:
 *   - Block size: 32 FP16 elements
 *   - Per block: compute max(abs(x)), scale = max_abs / 127.0
 *   - Quantize: quantized[i] = round(x[i] / scale), clamp to [-127, 127]
 *   - Store: 32 INT8 values (32 bytes) + 1 FP16 scale (2 bytes) = 34 bytes
 *   - Uncompressed: 32 * 2 = 64 bytes
 *   - Compression ratio: 34/64 = 53.1%
 *
 * Memory layout for compressed data (5120 FP16 elements):
 *   - 160 blocks of 32 elements each
 *   - Data section: 5120 INT8 values = 5120 bytes
 *   - Scale section: 160 FP16 scales = 320 bytes
 *   - Total compressed: 5440 bytes (vs 10240 bytes uncompressed)
 *   - Layout: [INT8 data (5120B)] [FP16 scales (320B)]
 *
 * Algorithm:
 *   Phase 1: Each GPU quantizes its local partial buffer to INT8+scale format
 *   Phase 2: Each GPU reads 3 peer compressed partials via BAR1 P2P
 *   Phase 3: Dequantize each peer's data to FP32, sum all 4 partials
 *   Phase 4: Write FP16 result to hidden buffer
 *
 * Expected performance:
 *   - PCIe BAR1 read is the bottleneck (~12 GB/s effective)
 *   - Halving transfer volume: 10KB -> 5.4KB per call
 *   - Theoretical latency reduction: ~79us * (5.4/10.24) = ~42us + quantize/dequantize overhead
 *   - Realistic estimate: ~45-55us per call (quantize/dequantize adds ~5-10us)
 *
 * Target: gfx906 (MI50/MI60), compiled as SHARED LIBRARY with:
 *   hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC
 *
 * Exposes host-callable C functions for ctypes-based dispatch.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

/* ===================================================================
 * Constants
 * =================================================================== */
#define BLOCK_SIZE 32
#define QUANT_SCALE 127.0f

/* ===================================================================
 * Device Functions: Quantization and Dequantization
 * =================================================================== */

/**
 * Quantize a block of 32 FP16 values to INT8 + scale.
 * 
 * @param input     Input FP16 buffer (32 elements)
 * @param output    Output INT8 buffer (32 elements)
 * @param scale_out Output FP16 scale (1 element)
 */
__device__ __forceinline__
void quantize_block_fp16_to_int8(
    const __half* __restrict__ input,
    int8_t* __restrict__ output,
    __half* __restrict__ scale_out)
{
    // Find max absolute value in the block
    float max_abs = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float val = __half2float(input[i]);
        float abs_val = fabsf(val);
        max_abs = fmaxf(max_abs, abs_val);
    }
    
    // Compute scale: scale = max_abs / 127.0
    // Handle edge case of all zeros
    float scale = (max_abs > 0.0f) ? (max_abs / QUANT_SCALE) : 1.0f;
    *scale_out = __float2half(scale);
    
    // Quantize: quantized[i] = round(x[i] / scale), clamp to [-127, 127]
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float val = __half2float(input[i]);
        float q = roundf(val * inv_scale);
        q = fmaxf(-QUANT_SCALE, fminf(QUANT_SCALE, q));
        output[i] = (int8_t)q;
    }
}

/**
 * Dequantize a block of INT8 values back to FP32.
 * 
 * @param input     Input INT8 buffer (32 elements)
 * @param scale     FP16 scale for this block
 * @param output    Output FP32 buffer (32 elements)
 */
__device__ __forceinline__
void dequantize_block_int8_to_fp32(
    const int8_t* __restrict__ input,
    __half scale,
    float* __restrict__ output)
{
    float scale_f = __half2float(scale);
    for (int i = 0; i < BLOCK_SIZE; i++) {
        output[i] = (float)input[i] * scale_f;
    }
}

/* ===================================================================
 * GPU Kernels
 * =================================================================== */

/**
 * kernel_quantize_to_int8:
 *
 * Quantize FP16 partial buffer to INT8 compressed format.
 *
 * Grid: (num_blocks, 1, 1), Block: (256, 1, 1)
 * Each block handles one 32-element block.
 *
 * Args:
 *   input_fp16:   Input FP16 buffer (n elements)
 *   output_int8:  Output INT8 buffer (n elements, data section)
 *   output_scales: Output FP16 scales (num_blocks elements)
 *   n:            Total number of FP16 elements
 *   num_blocks:   Number of blocks (ceil(n / 32))
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_quantize_to_int8(
    const __half* __restrict__ input_fp16,
    int8_t* __restrict__ output_int8,
    __half* __restrict__ output_scales,
    unsigned int n,
    unsigned int num_blocks)
{
    unsigned int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    unsigned int base_idx = block_id * BLOCK_SIZE;
    
    // Load input block to shared memory
    __half block_data[BLOCK_SIZE];
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        unsigned int idx = base_idx + i;
        block_data[i] = (idx < n) ? input_fp16[idx] : __float2half(0.0f);
    }
    __syncthreads();
    
    // Thread 0 performs quantization
    if (threadIdx.x == 0) {
        int8_t quantized[BLOCK_SIZE];
        __half scale;
        quantize_block_fp16_to_int8(block_data, quantized, &scale);
        
        // Write output
        for (int i = 0; i < BLOCK_SIZE; i++) {
            unsigned int idx = base_idx + i;
            if (idx < n) {
                output_int8[idx] = quantized[i];
            }
        }
        output_scales[block_id] = scale;
    }
}

/**
 * kernel_dequantize_and_sum_fp32:
 *
 * Dequantize INT8 compressed buffers and accumulate sum in FP32.
 *
 * Each thread block processes one 32-element block.
 * Thread 0 reads all peer INT8 data, dequantizes to FP32, and accumulates.
 *
 * Args:
 *   local_int8:    Local GPU INT8 data (n elements)
 *   local_scales:  Local GPU FP16 scales (num_blocks elements)
 *   peer0_int8:    Peer 0 INT8 data (P2P pointer, n elements)
 *   peer0_scales:  Peer 0 FP16 scales (P2P pointer, num_blocks elements)
 *   peer1_int8:    Peer 1 INT8 data (P2P pointer, n elements)
 *   peer1_scales:  Peer 1 FP16 scales (P2P pointer, num_blocks elements)
 *   peer2_int8:    Peer 2 INT8 data (P2P pointer, n elements)
 *   peer2_scales:  Peer 2 FP16 scales (P2P pointer, num_blocks elements)
 *   output_fp16:   Output FP16 buffer (n elements)
 *   n:             Total number of FP16 elements
 *   num_blocks:    Number of blocks (ceil(n / 32))
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_dequantize_and_sum_fp32(
    const int8_t* __restrict__ local_int8,
    const __half* __restrict__ local_scales,
    const int8_t* __restrict__ peer0_int8,
    const __half* __restrict__ peer0_scales,
    const int8_t* __restrict__ peer1_int8,
    const __half* __restrict__ peer1_scales,
    const int8_t* __restrict__ peer2_int8,
    const __half* __restrict__ peer2_scales,
    __half* __restrict__ output_fp16,
    unsigned int n,
    unsigned int num_blocks)
{
    unsigned int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    unsigned int base_idx = block_id * BLOCK_SIZE;
    
    // Allocate shared memory for dequantized values
    float local_fp32[BLOCK_SIZE];
    float peer0_fp32[BLOCK_SIZE];
    float peer1_fp32[BLOCK_SIZE];
    float peer2_fp32[BLOCK_SIZE];
    float sum_fp32[BLOCK_SIZE];
    
    // Thread 0 reads and dequantizes all 4 sources
    if (threadIdx.x == 0) {
        // Read scales
        __half scale_local = local_scales[block_id];
        __half scale_peer0 = peer0_scales[block_id];
        __half scale_peer1 = peer1_scales[block_id];
        __half scale_peer2 = peer2_scales[block_id];
        
        // Dequantize local
        dequantize_block_int8_to_fp32(&local_int8[base_idx], scale_local, local_fp32);
        
        // Dequantize peers (via P2P BAR1 reads)
        dequantize_block_int8_to_fp32(&peer0_int8[base_idx], scale_peer0, peer0_fp32);
        dequantize_block_int8_to_fp32(&peer1_int8[base_idx], scale_peer1, peer1_fp32);
        dequantize_block_int8_to_fp32(&peer2_int8[base_idx], scale_peer2, peer2_fp32);
        
        // Sum all 4 partials in FP32
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum_fp32[i] = local_fp32[i] + peer0_fp32[i] + peer1_fp32[i] + peer2_fp32[i];
        }
        
        // Write result as FP16
        for (int i = 0; i < BLOCK_SIZE; i++) {
            unsigned int idx = base_idx + i;
            if (idx < n) {
                output_fp16[idx] = __float2half(sum_fp32[i]);
            }
        }
    }
}

/**
 * kernel_dequantize_sum_with_residual_fp32:
 *
 * Dequantize INT8 compressed buffers, accumulate sum, and add residual hidden.
 * This is the main allreduce+residual kernel for TP=4.
 *
 * Args:
 *   hidden:        Hidden buffer (read-modify-write, n elements FP16)
 *   local_int8:    Local GPU INT8 data (n elements)
 *   local_scales:  Local GPU FP16 scales (num_blocks elements)
 *   peer0_int8:    Peer 0 INT8 data (P2P pointer, n elements)
 *   peer0_scales:  Peer 0 FP16 scales (P2P pointer, num_blocks elements)
 *   peer1_int8:    Peer 1 INT8 data (P2P pointer, n elements)
 *   peer1_scales:  Peer 1 FP16 scales (P2P pointer, num_blocks elements)
 *   peer2_int8:    Peer 2 INT8 data (P2P pointer, n elements)
 *   peer2_scales:  Peer 2 FP16 scales (P2P pointer, num_blocks elements)
 *   n:             Total number of FP16 elements
 *   num_blocks:    Number of blocks (ceil(n / 32))
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_dequantize_sum_with_residual_fp32(
    __half* __restrict__ hidden,
    const int8_t* __restrict__ local_int8,
    const __half* __restrict__ local_scales,
    const int8_t* __restrict__ peer0_int8,
    const __half* __restrict__ peer0_scales,
    const int8_t* __restrict__ peer1_int8,
    const __half* __restrict__ peer1_scales,
    const int8_t* __restrict__ peer2_int8,
    const __half* __restrict__ peer2_scales,
    unsigned int n,
    unsigned int num_blocks)
{
    unsigned int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    unsigned int base_idx = block_id * BLOCK_SIZE;
    
    // Thread 0 processes this block
    if (threadIdx.x == 0) {
        // Read scales
        __half scale_local = local_scales[block_id];
        __half scale_peer0 = peer0_scales[block_id];
        __half scale_peer1 = peer1_scales[block_id];
        __half scale_peer2 = peer2_scales[block_id];
        
        float local_fp32[BLOCK_SIZE];
        float peer0_fp32[BLOCK_SIZE];
        float peer1_fp32[BLOCK_SIZE];
        float peer2_fp32[BLOCK_SIZE];
        
        // Dequantize local
        dequantize_block_int8_to_fp32(&local_int8[base_idx], scale_local, local_fp32);
        
        // Dequantize peers (via P2P BAR1 reads)
        dequantize_block_int8_to_fp32(&peer0_int8[base_idx], scale_peer0, peer0_fp32);
        dequantize_block_int8_to_fp32(&peer1_int8[base_idx], scale_peer1, peer1_fp32);
        dequantize_block_int8_to_fp32(&peer2_int8[base_idx], scale_peer2, peer2_fp32);
        
        // Sum all 4 partials + hidden in FP32
        for (int i = 0; i < BLOCK_SIZE; i++) {
            unsigned int idx = base_idx + i;
            if (idx < n) {
                float h = __half2float(hidden[idx]);
                float sum = h + local_fp32[i] + peer0_fp32[i] + peer1_fp32[i] + peer2_fp32[i];
                hidden[idx] = __float2half(sum);
            }
        }
    }
}

/* ===================================================================
 * Host-callable C wrappers for Python ctypes dispatch
 * =================================================================== */

/**
 * kernel_p2p_allreduce_compressed_residual_tp4:
 *
 * TP=4 compressed allreduce with residual add.
 *
 * This function:
 *   1. Quantizes local partial to INT8 format (in-place in compressed buffer)
 *   2. Reads peer INT8 compressed data via P2P BAR1
 *   3. Dequantizes all 4 partials and sums with hidden
 *   4. Writes FP16 result to hidden buffer
 *
 * CRITICAL: This combined function requires all GPUs to launch simultaneously
 * (via threading.Barrier) because it combines quantize + allreduce in one call.
 * Each GPU tries to read peer compressed data before peers have written it.
 *
 * DEPRECATED: Use kernel_p2p_allreduce_compressed_quantize() + 
 *             kernel_p2p_allreduce_compressed_residual_tp4_read() separately
 *             with stream synchronization between them for correct execution.
 *
 * Args:
 *   hidden:          This GPU's hidden buffer (FP16, read-modify-write)
 *   partial_local:   This GPU's partial GEMV result (FP16 input)
 *   compressed_local: Local INT8 compressed buffer (output from quantize)
 *                     Layout: [INT8 data (n bytes)] [FP16 scales (num_blocks*2 bytes)]
 *   compressed_peer0: Peer 0 INT8 compressed buffer (P2P pointer)
 *   compressed_peer1: Peer 1 INT8 compressed buffer (P2P pointer)
 *   compressed_peer2: Peer 2 INT8 compressed buffer (P2P pointer)
 *   n:               Number of FP16 elements
 *   num_blocks:      Number of blocks (ceil(n / 32))
 *   stream:          HIP stream to launch on
 *
 * Returns: hipError_t (0 = success)
 */
extern "C" int kernel_p2p_allreduce_compressed_residual_tp4(
    void* hidden,
    const void* partial_local,
    void* compressed_local,
    const void* compressed_peer0,
    const void* compressed_peer1,
    const void* compressed_peer2,
    unsigned int n,
    unsigned int num_blocks,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    
    // Calculate pointers within compressed buffers
    // Layout: [INT8 data (n bytes)] [FP16 scales (num_blocks*2 bytes)]
    int8_t* local_int8 = (int8_t*)compressed_local;
    __half* local_scales = (__half*)((char*)compressed_local + n);
    
    const int8_t* peer0_int8 = (const int8_t*)compressed_peer0;
    const __half* peer0_scales = (const __half*)((const char*)compressed_peer0 + n);
    
    const int8_t* peer1_int8 = (const int8_t*)compressed_peer1;
    const __half* peer1_scales = (const __half*)((const char*)compressed_peer1 + n);
    
    const int8_t* peer2_int8 = (const int8_t*)compressed_peer2;
    const __half* peer2_scales = (const __half*)((const char*)compressed_peer2 + n);
    
    // Step 1: Quantize local partial to INT8
    unsigned int quantize_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_quantize_to_int8,
                       dim3(quantize_blocks), dim3(256), 0, stream,
                       (const __half*)partial_local,
                       local_int8,
                       local_scales,
                       n,
                       quantize_blocks);
    
    // Add threadfence to ensure all writes are visible to other GPUs
    // NOTE: This is not sufficient for cross-GPU synchronization!
    // The caller MUST use threading.Barrier to launch all GPUs simultaneously,
    // OR split into two separate kernel launches with stream sync between them.
    hipLaunchKernelGGL(kernel_dequantize_sum_with_residual_fp32,
                       dim3(num_blocks), dim3(256), 0, stream,
                       (__half*)hidden,
                       local_int8,
                       local_scales,
                       peer0_int8,
                       peer0_scales,
                       peer1_int8,
                       peer1_scales,
                       peer2_int8,
                       peer2_scales,
                       n,
                       num_blocks);
    
    return (int)hipGetLastError();
}

/**
 * kernel_p2p_allreduce_compressed_quantize:
 *
 * Quantize FP16 partial to INT8 compressed format.
 * This is Phase 1 of the compressed allreduce.
 *
 * Args:
 *   partial_local:   This GPU's partial GEMV result (FP16 input)
 *   compressed_local: Local INT8 compressed buffer (output)
 *                     Layout: [INT8 data (n bytes)] [FP16 scales (num_blocks*2 bytes)]
 *   n:               Number of FP16 elements
 *   num_blocks:      Number of blocks (ceil(n / 32))
 *   stream:          HIP stream to launch on
 *
 * Returns: hipError_t (0 = success)
 */
extern "C" int kernel_p2p_allreduce_compressed_quantize(
    const void* partial_local,
    void* compressed_local,
    unsigned int n,
    unsigned int num_blocks,
    hipStream_t stream)
{
    (void)hipGetLastError();
    
    // Calculate pointers within compressed buffers
    int8_t* local_int8 = (int8_t*)compressed_local;
    __half* local_scales = (__half*)((char*)compressed_local + n);
    
    // Quantize local partial to INT8
    unsigned int quantize_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_quantize_to_int8,
                       dim3(quantize_blocks), dim3(256), 0, stream,
                       (const __half*)partial_local,
                       local_int8,
                       local_scales,
                       n,
                       quantize_blocks);
    
    return (int)hipGetLastError();
}

/**
 * kernel_p2p_allreduce_compressed_residual_tp4_read:
 *
 * TP=4 compressed allreduce read phase with residual add.
 * This is Phase 2 of the compressed allreduce - assumes quantize has already
 * been completed on all GPUs (via stream synchronization).
 *
 * Args:
 *   hidden:          This GPU's hidden buffer (FP16, read-modify-write)
 *   compressed_local: Local INT8 compressed buffer (already quantized)
 *   compressed_peer0: Peer 0 INT8 compressed buffer (P2P pointer)
 *   compressed_peer1: Peer 1 INT8 compressed buffer (P2P pointer)
 *   compressed_peer2: Peer 2 INT8 compressed buffer (P2P pointer)
 *   n:               Number of FP16 elements
 *   num_blocks:      Number of blocks
 *   stream:          HIP stream
 *
 * Returns: hipError_t (0 = success)
 */
extern "C" int kernel_p2p_allreduce_compressed_residual_tp4_read(
    void* hidden,
    const void* compressed_local,
    const void* compressed_peer0,
    const void* compressed_peer1,
    const void* compressed_peer2,
    unsigned int n,
    unsigned int num_blocks,
    hipStream_t stream)
{
    (void)hipGetLastError();
    
    // Calculate pointers within compressed buffers
    int8_t* local_int8 = (int8_t*)compressed_local;
    __half* local_scales = (__half*)((char*)compressed_local + n);
    
    const int8_t* peer0_int8 = (const int8_t*)compressed_peer0;
    const __half* peer0_scales = (const __half*)((const char*)compressed_peer0 + n);
    
    const int8_t* peer1_int8 = (const int8_t*)compressed_peer1;
    const __half* peer1_scales = (const __half*)((const char*)compressed_peer1 + n);
    
    const int8_t* peer2_int8 = (const int8_t*)compressed_peer2;
    const __half* peer2_scales = (const __half*)((const char*)compressed_peer2 + n);
    
    // Dequantize and sum with residual
    hipLaunchKernelGGL(kernel_dequantize_sum_with_residual_fp32,
                       dim3(num_blocks), dim3(256), 0, stream,
                       (__half*)hidden,
                       local_int8,
                       local_scales,
                       peer0_int8,
                       peer0_scales,
                       peer1_int8,
                       peer1_scales,
                       peer2_int8,
                       peer2_scales,
                       n,
                       num_blocks);
    
    return (int)hipGetLastError();
}

/**
 * kernel_p2p_allreduce_compressed_sum_tp4:
 *
 * TP=4 compressed allreduce WITHOUT residual add (sum-only path).
 *
 * Args:
 *   partial_local:   This GPU's partial buffer (FP16, output = full sum)
 *   compressed_local: Local INT8 compressed buffer
 *   compressed_peer0: Peer 0 INT8 compressed buffer (P2P pointer)
 *   compressed_peer1: Peer 1 INT8 compressed buffer (P2P pointer)
 *   compressed_peer2: Peer 2 INT8 compressed buffer (P2P pointer)
 *   n:               Number of FP16 elements
 *   num_blocks:      Number of blocks
 *   stream:          HIP stream
 *
 * Returns: hipError_t (0 = success)
 */
/**
 * kernel_p2p_allreduce_compressed_sum_tp4:
 *
 * TP=4 compressed allreduce WITHOUT residual add (sum-only path).
 *
 * Args:
 *   partial_local:   This GPU's partial buffer (FP16, output = full sum)
 *   compressed_local: Local INT8 compressed buffer
 *   compressed_peer0: Peer 0 INT8 compressed buffer (P2P pointer)
 *   compressed_peer1: Peer 1 INT8 compressed buffer (P2P pointer)
 *   compressed_peer2: Peer 2 INT8 compressed buffer (P2P pointer)
 *   n:               Number of FP16 elements
 *   num_blocks:      Number of blocks
 *   stream:          HIP stream
 *
 * Returns: hipError_t (0 = success)
 */
extern "C" int kernel_p2p_allreduce_compressed_sum_tp4(
    void* partial_local,
    void* compressed_local,
    const void* compressed_peer0,
    const void* compressed_peer1,
    const void* compressed_peer2,
    unsigned int n,
    unsigned int num_blocks,
    hipStream_t stream)
{
    (void)hipGetLastError();
    
    int8_t* local_int8 = (int8_t*)compressed_local;
    __half* local_scales = (__half*)((char*)compressed_local + n);
    
    const int8_t* peer0_int8 = (const int8_t*)compressed_peer0;
    const __half* peer0_scales = (const __half*)((const char*)compressed_peer0 + n);
    
    const int8_t* peer1_int8 = (const int8_t*)compressed_peer1;
    const __half* peer1_scales = (const __half*)((const char*)compressed_peer1 + n);
    
    const int8_t* peer2_int8 = (const int8_t*)compressed_peer2;
    const __half* peer2_scales = (const __half*)((const char*)compressed_peer2 + n);
    
    // Step 1: Quantize
    unsigned int quantize_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_quantize_to_int8,
                       dim3(quantize_blocks), dim3(256), 0, stream,
                       (const __half*)partial_local,
                       local_int8,
                       local_scales,
                       n,
                       quantize_blocks);
    
    // Step 2: Dequantize and sum (no residual)
    hipLaunchKernelGGL(kernel_dequantize_and_sum_fp32,
                       dim3(num_blocks), dim3(256), 0, stream,
                       local_int8,
                       local_scales,
                       peer0_int8,
                       peer0_scales,
                       peer1_int8,
                       peer1_scales,
                       peer2_int8,
                       peer2_scales,
                       (__half*)partial_local,
                       n,
                       num_blocks);
    
    return (int)hipGetLastError();
}

/**
 * kernel_p2p_allreduce_compressed_sum_tp4_read:
 *
 * TP=4 compressed allreduce read phase WITHOUT residual add.
 * This is Phase 2 of the compressed allreduce - assumes quantize has already
 * been completed on all GPUs (via stream synchronization).
 *
 * Args:
 *   partial_local:   This GPU's partial buffer (FP16, output = full sum)
 *   compressed_local: Local INT8 compressed buffer (already quantized)
 *   compressed_peer0: Peer 0 INT8 compressed buffer (P2P pointer)
 *   compressed_peer1: Peer 1 INT8 compressed buffer (P2P pointer)
 *   compressed_peer2: Peer 2 INT8 compressed buffer (P2P pointer)
 *   n:               Number of FP16 elements
 *   num_blocks:      Number of blocks
 *   stream:          HIP stream
 *
 * Returns: hipError_t (0 = success)
 */
extern "C" int kernel_p2p_allreduce_compressed_sum_tp4_read(
    void* partial_local,
    const void* compressed_local,
    const void* compressed_peer0,
    const void* compressed_peer1,
    const void* compressed_peer2,
    unsigned int n,
    unsigned int num_blocks,
    hipStream_t stream)
{
    (void)hipGetLastError();
    
    int8_t* local_int8 = (int8_t*)compressed_local;
    __half* local_scales = (__half*)((char*)compressed_local + n);
    
    const int8_t* peer0_int8 = (const int8_t*)compressed_peer0;
    const __half* peer0_scales = (const __half*)((const char*)compressed_peer0 + n);
    
    const int8_t* peer1_int8 = (const int8_t*)compressed_peer1;
    const __half* peer1_scales = (const __half*)((const char*)compressed_peer1 + n);
    
    const int8_t* peer2_int8 = (const int8_t*)compressed_peer2;
    const __half* peer2_scales = (const __half*)((const char*)compressed_peer2 + n);
    
    // Dequantize and sum (no residual)
    hipLaunchKernelGGL(kernel_dequantize_and_sum_fp32,
                       dim3(num_blocks), dim3(256), 0, stream,
                       local_int8,
                       local_scales,
                       peer0_int8,
                       peer0_scales,
                       peer1_int8,
                       peer1_scales,
                       peer2_int8,
                       peer2_scales,
                       (__half*)partial_local,
                       n,
                       num_blocks);
    
    return (int)hipGetLastError();
}
