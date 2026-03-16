/**
 * Kernel-based P2P allreduce for TP inference.
 *
 * Milestone: kernel-p2p-allreduce
 *
 * This kernel eliminates host round-trips entirely by having each GPU run a
 * single kernel that:
 *   1. Reads peer GPU partial buffers directly via BAR1-mapped P2P pointers
 *      (enabled by hipDeviceEnablePeerAccess at init time)
 *   2. Reduces all partials + local hidden state
 *   3. Writes the result to the local hidden buffer
 *
 * Each GPU runs its own kernel simultaneously — no gather/broadcast, no
 * intermediate buffers, no sequential host synchronization points.
 *
 * Current star-topology allreduce per call:
 *   hipSetDevice x4, hipMemcpyPeerAsync gather x3, hipStreamSynchronize,
 *   reduce kernel on GPU0, hipStreamSynchronize,
 *   hipMemcpyPeerAsync broadcast x3, sync all = ~119us
 *
 * Target approach (this kernel):
 *   Each GPU launches kernel_p2p_allreduce_residual_tp4 once.
 *   Kernel reads: local hidden + local partial + 3 remote partials (via P2P)
 *   Kernel writes: local hidden = hidden + sum(all 4 partials)
 *   Single launch per GPU, single sync per GPU = ~1 kernel + 1 event sync
 *
 * On gfx906/PCIe, P2P access uses BAR1 aperture after hipDeviceEnablePeerAccess.
 * Remote reads are ~12 GB/s (PCIe, not XGMI), but for 10KB payloads the
 * fixed kernel launch latency dominates over bandwidth cost.
 *
 * Target: gfx906 (MI50/MI60), compiled as SHARED LIBRARY with:
 *   hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC
 *
 * Exposes host-callable C functions (kernel_p2p_allreduce_residual_tp*) that
 * use hipLaunchKernelGGL for reliable ctypes-based dispatch.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/* ===================================================================
 * GPU kernels: each runs on the local GPU, reads peer memory via P2P
 * =================================================================== */

/**
 * kernel_p2p_allreduce_residual_tp4_kernel:
 *
 *   hidden[i] = hidden[i] + partial_local[i]
 *             + partial_peer0[i] + partial_peer1[i] + partial_peer2[i]
 *
 * Runs on EACH GPU simultaneously. Each GPU:
 *   - hidden:        its own hidden buffer (local memory, read-modify-write)
 *   - partial_local: its own partial GEMV result (local memory)
 *   - partial_peer0, partial_peer1, partial_peer2:
 *                    the other 3 GPUs' partial buffers (P2P pointers via BAR1)
 *
 * Grid: (ceil(n/2/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes 2 FP16 elements (half2 packing).
 * FP32 accumulation for correctness (mandatory on gfx906).
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_p2p_allreduce_residual_tp4_kernel(
    __half* __restrict__ hidden,
    const __half* __restrict__ partial_local,
    const __half* __restrict__ partial_peer0,
    const __half* __restrict__ partial_peer1,
    const __half* __restrict__ partial_peer2,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        // Load local data
        __half2 h  = *(__half2*)(hidden + idx);
        __half2 pl = *(__half2*)(partial_local + idx);
        // Load remote data via P2P (BAR1-mapped on gfx906/PCIe)
        __half2 p0 = *(__half2*)(partial_peer0 + idx);
        __half2 p1 = *(__half2*)(partial_peer1 + idx);
        __half2 p2 = *(__half2*)(partial_peer2 + idx);

        // FP32 accumulation (mandatory on gfx906)
        float h0  = __half2float(__low2half(h));
        float h1  = __half2float(__high2half(h));

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
        // Handle odd element at end
        float h  = __half2float(hidden[idx]);
        float pl = __half2float(partial_local[idx]);
        float p0 = __half2float(partial_peer0[idx]);
        float p1 = __half2float(partial_peer1[idx]);
        float p2 = __half2float(partial_peer2[idx]);
        hidden[idx] = __float2half(h + pl + p0 + p1 + p2);
    }
}

/**
 * kernel_p2p_allreduce_residual_tp2_kernel:
 *
 *   hidden[i] = hidden[i] + partial_local[i] + partial_peer0[i]
 *
 * Runs on BOTH GPUs simultaneously.
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_p2p_allreduce_residual_tp2_kernel(
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

        *(__half2*)(hidden + idx) = __halves2half2(
            __float2half(h0
                + __half2float(__low2half(pl))
                + __half2float(__low2half(p0))),
            __float2half(h1
                + __half2float(__high2half(pl))
                + __half2float(__high2half(p0)))
        );
    } else if (idx < n) {
        float h  = __half2float(hidden[idx]);
        float pl = __half2float(partial_local[idx]);
        float p0 = __half2float(partial_peer0[idx]);
        hidden[idx] = __float2half(h + pl + p0);
    }
}

/**
 * kernel_p2p_allreduce_sum_tp4_kernel:
 *   Allreduce WITHOUT hidden add (for _allreduce_sum path).
 *
 *   partial_local[i] = partial_local[i] + partial_peer0[i]
 *                    + partial_peer1[i] + partial_peer2[i]
 *
 * Each GPU writes the full reduced sum into its own partial buffer.
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void kernel_p2p_allreduce_sum_tp4_kernel(
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
 * These use hipLaunchKernelGGL (HIP C++ runtime launch) which is
 * reliable when called from a shared library loaded via ctypes.
 *
 * Each function launches the kernel on the CALLING device (the device
 * must be set by the caller via hipSetDevice before calling).
 * Returns hipError_t (0 = success).
 * =================================================================== */

/**
 * kernel_p2p_allreduce_residual_tp4:
 *
 * Launch the TP=4 kernel P2P allreduce + residual add.
 *
 * Args:
 *   hidden:        this GPU's hidden buffer (local, read-modify-write)
 *   partial_local: this GPU's partial GEMV result (local)
 *   partial_peer0: first peer GPU's partial buffer (P2P pointer)
 *   partial_peer1: second peer GPU's partial buffer (P2P pointer)
 *   partial_peer2: third peer GPU's partial buffer (P2P pointer)
 *   n:             number of FP16 elements
 *   stream:        HIP stream to launch on
 */
extern "C" int kernel_p2p_allreduce_residual_tp4(
    void* hidden,
    const void* partial_local,
    const void* partial_peer0,
    const void* partial_peer1,
    const void* partial_peer2,
    unsigned int n,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256 - 1) / (2*256);
    hipLaunchKernelGGL(kernel_p2p_allreduce_residual_tp4_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)hidden,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)partial_peer1,
                       (const __half*)partial_peer2,
                       n);
    return (int)hipGetLastError();
}

/**
 * kernel_p2p_allreduce_residual_tp2:
 *
 * Launch the TP=2 kernel P2P allreduce + residual add.
 */
extern "C" int kernel_p2p_allreduce_residual_tp2(
    void* hidden,
    const void* partial_local,
    const void* partial_peer0,
    unsigned int n,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256 - 1) / (2*256);
    hipLaunchKernelGGL(kernel_p2p_allreduce_residual_tp2_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)hidden,
                       (const __half*)partial_local,
                       (const __half*)partial_peer0,
                       n);
    return (int)hipGetLastError();
}

/**
 * kernel_p2p_allreduce_sum_tp4:
 *
 * Launch the TP=4 kernel P2P allreduce sum (no residual add).
 */
extern "C" int kernel_p2p_allreduce_sum_tp4(
    void* partial_local,
    const void* partial_peer0,
    const void* partial_peer1,
    const void* partial_peer2,
    unsigned int n,
    hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256 - 1) / (2*256);
    hipLaunchKernelGGL(kernel_p2p_allreduce_sum_tp4_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)partial_local,
                       (const __half*)partial_peer0,
                       (const __half*)partial_peer1,
                       (const __half*)partial_peer2,
                       n);
    return (int)hipGetLastError();
}
