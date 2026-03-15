/**
 * Fused P2P gather-reduce-residual kernel for TP inference.
 *
 * This kernel reads partial GEMV results from multiple GPU memory spaces via
 * P2P access (enabled via hipDeviceEnablePeerAccess), sums them with the local
 * hidden state (residual), and writes the result back to each GPU's hidden buffer.
 *
 * Feature: fused-p2p-gemv-epilogue
 *
 * Design (Option B - Separate Fused Kernel):
 *   - Each GPU launches its own fused kernel
 *   - Each GPU's kernel reads the OTHER GPU partial results via P2P pointers
 *   - Sums all partials + hidden, writes result to d_hidden on THAT GPU
 *   - All 4 GPUs write correct result simultaneously (no broadcast needed!)
 *
 * For TP=4, hidden_size=5120 (10KB):
 *   - Each GPU reads local partial (10KB) + 3 remote partials (30KB) + local hidden (10KB)
 *   - Total reads per GPU: 50KB
 *   - All 4 GPUs in parallel: effective bandwidth = 4 × 50KB = 200KB / allreduce latency
 *   - vs current: GPU0 reads 3 remote (30KB), reduces, then broadcasts to 3 GPUs (30KB)
 *     Sequential: 60KB / P2P bandwidth
 *   - The parallel approach should be 2-4x faster for small payloads
 *
 * Each GPU can access peer memory if P2P was enabled between all pairs.
 * On gfx906 (PCIe), this is "soft P2P" via host memory mapping, but still
 * much faster than the CPU-mediated path.
 *
 * Target: gfx906 (MI50/MI60), compiled as SHARED LIBRARY with:
 *   hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/**
 * fused_p2p_reduce_residual_tp2:
 *   Runs on BOTH GPUs simultaneously (each GPU reads the other's partial via P2P).
 *   hidden[i] = hidden[i] + partial_local[i] + partial_peer0[i]
 *
 * Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes 2 FP16 elements (half2 packing).
 * FP32 accumulation for correctness.
 *
 * partial_local: this GPU's GEMV partial result (local memory)
 * partial_peer0: the other GPU's partial result (P2P pointer)
 * hidden: this GPU's hidden state (local memory, read-modify-write)
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void fused_p2p_reduce_residual_tp2_kernel(
    __half* __restrict__ hidden,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 h  = *(__half2*)(hidden + idx);
        __half2 pl = *(__half2*)(partial_local + idx);
        __half2 p0 = *(__half2*)(partial_peer0 + idx);

        float h0  = __half2float(__low2half(h));
        float h1  = __half2float(__high2half(h));
        float pl0 = __half2float(__low2half(pl));
        float pl1 = __half2float(__high2half(pl));
        float p00 = __half2float(__low2half(p0));
        float p01 = __half2float(__high2half(p0));

        *(__half2*)(hidden + idx) = __halves2half2(
            __float2half(h0 + pl0 + p00),
            __float2half(h1 + pl1 + p01)
        );
    } else if (idx < n) {
        float h  = __half2float(hidden[idx]);
        float pl = __half2float(partial_local[idx]);
        float p0 = __half2float(partial_peer0[idx]);
        hidden[idx] = __float2half(h + pl + p0);
    }
}

/**
 * fused_p2p_reduce_residual_tp4:
 *   Runs on ALL 4 GPUs simultaneously.
 *   hidden[i] = hidden[i] + partial_local[i] + partial_peer0[i] + partial_peer1[i] + partial_peer2[i]
 *
 * Each GPU passes:
 *   - hidden: its own hidden buffer
 *   - partial_local: its own partial GEMV result
 *   - partial_peer{0,1,2}: the OTHER 3 GPUs' partial buffers (P2P pointers)
 *
 * Result: All 4 GPUs update their hidden[i] to the correct allreduced value simultaneously.
 * No broadcast needed since each GPU computes the full sum independently.
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void fused_p2p_reduce_residual_tp4_kernel(
    __half* __restrict__ hidden,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ partial_peer1,
    const __half* __restrict__ partial_peer2,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 h  = *(__half2*)(hidden + idx);
        __half2 pl = *(__half2*)(partial_local + idx);
        __half2 p0 = *(__half2*)(partial_peer0 + idx);
        __half2 p1 = *(__half2*)(partial_peer1 + idx);
        __half2 p2 = *(__half2*)(partial_peer2 + idx);

        float h0 = __half2float(__low2half(h));
        float h1 = __half2float(__high2half(h));

        *(__half2*)(hidden + idx) = __halves2half2(
            __float2half(h0
                + __half2float(__low2half(pl))
                + __half2float(__low2half(p0))
                + __half2float(__low2half(p1))
                + __half2float(__low2half(p2))),
            __float2half(h1
                + __half2float(__high2half(pl))
                + __half2float(__high2half(p0))
                + __half2float(__high2half(p1))
                + __half2float(__high2half(p2)))
        );
    } else if (idx < n) {
        float h  = __half2float(hidden[idx]);
        float pl = __half2float(partial_local[idx]);
        float p0 = __half2float(partial_peer0[idx]);
        float p1 = __half2float(partial_peer1[idx]);
        float p2 = __half2float(partial_peer2[idx]);
        hidden[idx] = __float2half(h + pl + p0 + p1 + p2);
    }
}

/**
 * fused_p2p_reduce_only_tp4:
 *   Like _residual but without adding hidden (for _allreduce_sum path).
 *   partial_local[i] = partial_local[i] + partial_peer0[i] + partial_peer1[i] + partial_peer2[i]
 *
 * Each GPU computes the full reduced sum independently into its own partial buffer.
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void fused_p2p_reduce_only_tp4_kernel(
    __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ partial_peer1,
    const __half* __restrict__ partial_peer2,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 pl = *(__half2*)(partial_local + idx);
        __half2 p0 = *(__half2*)(partial_peer0 + idx);
        __half2 p1 = *(__half2*)(partial_peer1 + idx);
        __half2 p2 = *(__half2*)(partial_peer2 + idx);

        *(__half2*)(partial_local + idx) = __halves2half2(
            __float2half(
                __half2float(__low2half(pl))
                + __half2float(__low2half(p0))
                + __half2float(__low2half(p1))
                + __half2float(__low2half(p2))),
            __float2half(
                __half2float(__high2half(pl))
                + __half2float(__high2half(p0))
                + __half2float(__high2half(p1))
                + __half2float(__high2half(p2)))
        );
    } else if (idx < n) {
        float pl = __half2float(partial_local[idx]);
        float p0 = __half2float(partial_peer0[idx]);
        float p1 = __half2float(partial_peer1[idx]);
        float p2 = __half2float(partial_peer2[idx]);
        partial_local[idx] = __float2half(pl + p0 + p1 + p2);
    }
}

/* ===================================================================
 * Host-callable C wrappers for Python ctypes dispatch.
 * These use hipLaunchKernelGGL (HIP C++ runtime launch).
 * Returns hipError_t (0 = success).
 *
 * Each function launches the kernel on the CALLING device.
 * The caller must have set the device context before calling.
 * =================================================================== */

extern "C" int fused_p2p_reduce_residual_tp2(
    void* hidden,
    const void* partial_local,
    const void* partial_peer0,
    unsigned int n,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256-1) / (2*256);
    hipLaunchKernelGGL(fused_p2p_reduce_residual_tp2_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)hidden,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       n);
    return (int)hipGetLastError();
}

extern "C" int fused_p2p_reduce_residual_tp4(
    void* hidden,
    const void* partial_local,
    const void* partial_peer0,
    const void* partial_peer1,
    const void* partial_peer2,
    unsigned int n,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256-1) / (2*256);
    hipLaunchKernelGGL(fused_p2p_reduce_residual_tp4_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)hidden,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)partial_peer1,
                       (const __half*)partial_peer2,
                       n);
    return (int)hipGetLastError();
}

extern "C" int fused_p2p_reduce_only_tp4(
    void* partial_local,
    const void* partial_peer0,
    const void* partial_peer1,
    const void* partial_peer2,
    unsigned int n,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256-1) / (2*256);
    hipLaunchKernelGGL(fused_p2p_reduce_only_tp4_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)partial_peer1,
                       (const __half*)partial_peer2,
                       n);
    return (int)hipGetLastError();
}
