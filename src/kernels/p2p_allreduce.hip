/**
 * P2P allreduce kernels for TP inference.
 *
 * Replaces the CPU-mediated allreduce (fast_allreduce.c) with:
 *   1. hipMemcpyPeerAsync gather of partials to GPU0
 *   2. On-device HIP reduce kernel on GPU0 (sum FP16 partials + optional hidden for residual)
 *   3. hipMemcpyPeerAsync broadcast result from GPU0 to all GPUs
 *   4. Single synchronization per GPU stream
 *
 * For small 10KB payloads (hidden_size=5120 FP16), this eliminates the
 * host-roundtrip latency overhead (D2H + CPU add + H2D per GPU).
 *
 * Target: gfx906 (MI50/MI60), compiled as SHARED LIBRARY with:
 *   hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC
 *
 * Exposes host-callable C functions that launch the GPU kernels via
 * hipLaunchKernelGGL (HIP C++ runtime launch, more reliable than
 * hipModuleLaunchKernel from Python ctypes).
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/**
 * p2p_reduce_sum_residual_tp2:
 *   hidden[i] = hidden[i] + partial0[i] + partial1[i]
 *
 * Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes 2 FP16 elements (half2 packing).
 * FP32 accumulation for correctness.
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_reduce_sum_residual_tp2_kernel(
    __half* __restrict__ hidden,
    const __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 h = *(__half2*)(hidden + idx);
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);

        float h0 = __half2float(__low2half(h));
        float h1 = __half2float(__high2half(h));

        *(__half2*)(hidden + idx) = __halves2half2(
            __float2half(h0 + __half2float(__low2half(p0))
                            + __half2float(__low2half(p1))),
            __float2half(h1 + __half2float(__high2half(p0))
                            + __half2float(__high2half(p1)))
        );
    } else if (idx < n) {
        float h = __half2float(hidden[idx]);
        float p0 = __half2float(partial0[idx]);
        float p1 = __half2float(partial1[idx]);
        hidden[idx] = __float2half(h + p0 + p1);
    }
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_reduce_sum_residual_tp3_kernel(
    __half* __restrict__ hidden,
    const __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    const __half* __restrict__ partial2,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 h = *(__half2*)(hidden + idx);
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);
        __half2 p2 = *(__half2*)(partial2 + idx);

        float h0 = __half2float(__low2half(h));
        float h1 = __half2float(__high2half(h));

        *(__half2*)(hidden + idx) = __halves2half2(
            __float2half(h0 + __half2float(__low2half(p0))
                            + __half2float(__low2half(p1))
                            + __half2float(__low2half(p2))),
            __float2half(h1 + __half2float(__high2half(p0))
                            + __half2float(__high2half(p1))
                            + __half2float(__high2half(p2)))
        );
    } else if (idx < n) {
        float h = __half2float(hidden[idx]);
        float p0 = __half2float(partial0[idx]);
        float p1 = __half2float(partial1[idx]);
        float p2 = __half2float(partial2[idx]);
        hidden[idx] = __float2half(h + p0 + p1 + p2);
    }
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_reduce_sum_residual_tp4_kernel(
    __half* __restrict__ hidden,
    const __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    const __half* __restrict__ partial2,
    const __half* __restrict__ partial3,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 h = *(__half2*)(hidden + idx);
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);
        __half2 p2 = *(__half2*)(partial2 + idx);
        __half2 p3 = *(__half2*)(partial3 + idx);

        float h0 = __half2float(__low2half(h));
        float h1 = __half2float(__high2half(h));

        *(__half2*)(hidden + idx) = __halves2half2(
            __float2half(h0 + __half2float(__low2half(p0))
                            + __half2float(__low2half(p1))
                            + __half2float(__low2half(p2))
                            + __half2float(__low2half(p3))),
            __float2half(h1 + __half2float(__high2half(p0))
                            + __half2float(__high2half(p1))
                            + __half2float(__high2half(p2))
                            + __half2float(__high2half(p3)))
        );
    } else if (idx < n) {
        float h = __half2float(hidden[idx]);
        float p0 = __half2float(partial0[idx]);
        float p1 = __half2float(partial1[idx]);
        float p2 = __half2float(partial2[idx]);
        float p3 = __half2float(partial3[idx]);
        hidden[idx] = __float2half(h + p0 + p1 + p2 + p3);
    }
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_reduce_sum_only_tp2_kernel(
    __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);

        *(__half2*)(partial0 + idx) = __halves2half2(
            __float2half(__half2float(__low2half(p0)) + __half2float(__low2half(p1))),
            __float2half(__half2float(__high2half(p0)) + __half2float(__high2half(p1)))
        );
    } else if (idx < n) {
        partial0[idx] = __float2half(
            __half2float(partial0[idx]) + __half2float(partial1[idx]));
    }
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_reduce_sum_only_tp3_kernel(
    __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    const __half* __restrict__ partial2,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);
        __half2 p2 = *(__half2*)(partial2 + idx);

        *(__half2*)(partial0 + idx) = __halves2half2(
            __float2half(__half2float(__low2half(p0))
                        + __half2float(__low2half(p1))
                        + __half2float(__low2half(p2))),
            __float2half(__half2float(__high2half(p0))
                        + __half2float(__high2half(p1))
                        + __half2float(__high2half(p2)))
        );
    } else if (idx < n) {
        partial0[idx] = __float2half(
            __half2float(partial0[idx])
            + __half2float(partial1[idx])
            + __half2float(partial2[idx]));
    }
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_reduce_sum_only_tp4_kernel(
    __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    const __half* __restrict__ partial2,
    const __half* __restrict__ partial3,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);
        __half2 p2 = *(__half2*)(partial2 + idx);
        __half2 p3 = *(__half2*)(partial3 + idx);

        *(__half2*)(partial0 + idx) = __halves2half2(
            __float2half(__half2float(__low2half(p0))
                        + __half2float(__low2half(p1))
                        + __half2float(__low2half(p2))
                        + __half2float(__low2half(p3))),
            __float2half(__half2float(__high2half(p0))
                        + __half2float(__high2half(p1))
                        + __half2float(__high2half(p2))
                        + __half2float(__high2half(p3)))
        );
    } else if (idx < n) {
        partial0[idx] = __float2half(
            __half2float(partial0[idx])
            + __half2float(partial1[idx])
            + __half2float(partial2[idx])
            + __half2float(partial3[idx]));
    }
}

/* ===================================================================
 * Host-callable C wrappers for Python ctypes dispatch.
 * These use hipLaunchKernelGGL (HIP C++ runtime launch) which is
 * reliable when called from a shared library loaded via ctypes.
 * Returns hipError_t (0 = success).
 * =================================================================== */

extern "C" int p2p_reduce_residual_tp2(
    void* hidden, const void* partial0, const void* partial1,
    unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256-1) / (2*256);
    hipLaunchKernelGGL(p2p_reduce_sum_residual_tp2_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)hidden,
                       (const __half*)partial0,
                       (const __half*)partial1, n);
    return (int)hipGetLastError();
}

extern "C" int p2p_reduce_residual_tp3(
    void* hidden,
    const void* partial0, const void* partial1, const void* partial2,
    unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256-1) / (2*256);
    hipLaunchKernelGGL(p2p_reduce_sum_residual_tp3_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)hidden,
                       (const __half*)partial0,
                       (const __half*)partial1,
                       (const __half*)partial2, n);
    return (int)hipGetLastError();
}

extern "C" int p2p_reduce_residual_tp4(
    void* hidden,
    const void* partial0, const void* partial1,
    const void* partial2, const void* partial3,
    unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256-1) / (2*256);
    hipLaunchKernelGGL(p2p_reduce_sum_residual_tp4_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)hidden,
                       (const __half*)partial0,
                       (const __half*)partial1,
                       (const __half*)partial2,
                       (const __half*)partial3, n);
    return (int)hipGetLastError();
}

extern "C" int p2p_reduce_sum_tp2(
    void* partial0, const void* partial1,
    unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256-1) / (2*256);
    hipLaunchKernelGGL(p2p_reduce_sum_only_tp2_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)partial0,
                       (const __half*)partial1, n);
    return (int)hipGetLastError();
}

extern "C" int p2p_reduce_sum_tp3(
    void* partial0, const void* partial1, const void* partial2,
    unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256-1) / (2*256);
    hipLaunchKernelGGL(p2p_reduce_sum_only_tp3_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)partial0,
                       (const __half*)partial1,
                       (const __half*)partial2, n);
    return (int)hipGetLastError();
}

extern "C" int p2p_reduce_sum_tp4(
    void* partial0, const void* partial1,
    const void* partial2, const void* partial3,
    unsigned int n, hipStream_t stream)
{
    (void)hipGetLastError();  // Clear any stale error
    unsigned int blocks = (n + 2*256-1) / (2*256);
    hipLaunchKernelGGL(p2p_reduce_sum_only_tp4_kernel,
                       dim3(blocks), dim3(256), 0, stream,
                       (__half*)partial0,
                       (const __half*)partial1,
                       (const __half*)partial2,
                       (const __half*)partial3, n);
    return (int)hipGetLastError();
}


#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/**
 * reduce_sum_fp16_tp2:
 *   dst[i] = hidden[i] + partial0[i] + partial1[i]
 *
 * Grid: (ceil(n/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes 2 FP16 elements (half2 packing).
 * FP32 accumulation for correctness.
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
extern "C" __global__ void p2p_reduce_sum_residual_tp2(
    __half* __restrict__ hidden,
    const __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        // FP32 accumulation for correctness
        __half2 h = *(__half2*)(hidden + idx);
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);

        float h0 = __half2float(__low2half(h));
        float h1 = __half2float(__high2half(h));
        float p00 = __half2float(__low2half(p0));
        float p01 = __half2float(__high2half(p0));
        float p10 = __half2float(__low2half(p1));
        float p11 = __half2float(__high2half(p1));

        *(__half2*)(hidden + idx) = __halves2half2(
            __float2half(h0 + p00 + p10),
            __float2half(h1 + p01 + p11)
        );
    } else if (idx < n) {
        float h = __half2float(hidden[idx]);
        float p0 = __half2float(partial0[idx]);
        float p1 = __half2float(partial1[idx]);
        hidden[idx] = __float2half(h + p0 + p1);
    }
}

/**
 * reduce_sum_fp16_tp3:
 *   dst[i] = hidden[i] + partial0[i] + partial1[i] + partial2[i]
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
extern "C" __global__ void p2p_reduce_sum_residual_tp3(
    __half* __restrict__ hidden,
    const __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    const __half* __restrict__ partial2,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 h = *(__half2*)(hidden + idx);
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);
        __half2 p2 = *(__half2*)(partial2 + idx);

        float h0 = __half2float(__low2half(h));
        float h1 = __half2float(__high2half(h));

        *(__half2*)(hidden + idx) = __halves2half2(
            __float2half(h0 + __half2float(__low2half(p0))
                            + __half2float(__low2half(p1))
                            + __half2float(__low2half(p2))),
            __float2half(h1 + __half2float(__high2half(p0))
                            + __half2float(__high2half(p1))
                            + __half2float(__high2half(p2)))
        );
    } else if (idx < n) {
        float h = __half2float(hidden[idx]);
        float p0 = __half2float(partial0[idx]);
        float p1 = __half2float(partial1[idx]);
        float p2 = __half2float(partial2[idx]);
        hidden[idx] = __float2half(h + p0 + p1 + p2);
    }
}

/**
 * reduce_sum_fp16_tp4:
 *   dst[i] = hidden[i] + partial0[i] + partial1[i] + partial2[i] + partial3[i]
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
extern "C" __global__ void p2p_reduce_sum_residual_tp4(
    __half* __restrict__ hidden,
    const __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    const __half* __restrict__ partial2,
    const __half* __restrict__ partial3,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 h = *(__half2*)(hidden + idx);
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);
        __half2 p2 = *(__half2*)(partial2 + idx);
        __half2 p3 = *(__half2*)(partial3 + idx);

        float h0 = __half2float(__low2half(h));
        float h1 = __half2float(__high2half(h));

        *(__half2*)(hidden + idx) = __halves2half2(
            __float2half(h0 + __half2float(__low2half(p0))
                            + __half2float(__low2half(p1))
                            + __half2float(__low2half(p2))
                            + __half2float(__low2half(p3))),
            __float2half(h1 + __half2float(__high2half(p0))
                            + __half2float(__high2half(p1))
                            + __half2float(__high2half(p2))
                            + __half2float(__high2half(p3)))
        );
    } else if (idx < n) {
        float h = __half2float(hidden[idx]);
        float p0 = __half2float(partial0[idx]);
        float p1 = __half2float(partial1[idx]);
        float p2 = __half2float(partial2[idx]);
        float p3 = __half2float(partial3[idx]);
        hidden[idx] = __float2half(h + p0 + p1 + p2 + p3);
    }
}

/**
 * reduce_sum_only_tp2 / tp3 / tp4:
 *   Allreduce WITHOUT hidden add (for _allreduce_sum path)
 *   dst[i] = partial0[i] + partial1[i] [+ partial2[i] [+ partial3[i]]]
 *
 * Overwrites partial0 (used as dst on GPU0).
 */
__attribute__((amdgpu_flat_work_group_size(256, 256)))
extern "C" __global__ void p2p_reduce_sum_only_tp2(
    __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);

        float p00 = __half2float(__low2half(p0));
        float p01 = __half2float(__high2half(p0));
        float p10 = __half2float(__low2half(p1));
        float p11 = __half2float(__high2half(p1));

        *(__half2*)(partial0 + idx) = __halves2half2(
            __float2half(p00 + p10),
            __float2half(p01 + p11)
        );
    } else if (idx < n) {
        partial0[idx] = __float2half(
            __half2float(partial0[idx]) + __half2float(partial1[idx]));
    }
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
extern "C" __global__ void p2p_reduce_sum_only_tp3(
    __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    const __half* __restrict__ partial2,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);
        __half2 p2 = *(__half2*)(partial2 + idx);

        *(__half2*)(partial0 + idx) = __halves2half2(
            __float2half(__half2float(__low2half(p0))
                        + __half2float(__low2half(p1))
                        + __half2float(__low2half(p2))),
            __float2half(__half2float(__high2half(p0))
                        + __half2float(__high2half(p1))
                        + __half2float(__high2half(p2)))
        );
    } else if (idx < n) {
        partial0[idx] = __float2half(
            __half2float(partial0[idx])
            + __half2float(partial1[idx])
            + __half2float(partial2[idx]));
    }
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
extern "C" __global__ void p2p_reduce_sum_only_tp4(
    __half* __restrict__ partial0,
    const __half* __restrict__ partial1,
    const __half* __restrict__ partial2,
    const __half* __restrict__ partial3,
    unsigned int n
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __half2 p0 = *(__half2*)(partial0 + idx);
        __half2 p1 = *(__half2*)(partial1 + idx);
        __half2 p2 = *(__half2*)(partial2 + idx);
        __half2 p3 = *(__half2*)(partial3 + idx);

        *(__half2*)(partial0 + idx) = __halves2half2(
            __float2half(__half2float(__low2half(p0))
                        + __half2float(__low2half(p1))
                        + __half2float(__low2half(p2))
                        + __half2float(__low2half(p3))),
            __float2half(__half2float(__high2half(p0))
                        + __half2float(__high2half(p1))
                        + __half2float(__high2half(p2))
                        + __half2float(__high2half(p3)))
        );
    } else if (idx < n) {
        partial0[idx] = __float2half(
            __half2float(partial0[idx])
            + __half2float(partial1[idx])
            + __half2float(partial2[idx])
            + __half2float(partial3[idx]));
    }
}
