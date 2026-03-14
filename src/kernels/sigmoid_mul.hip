/**
 * Elementwise sigmoid-gated multiplication: out[i] *= sigmoid(gate[i])
 *
 * Grid: (ceil(N/256), 1, 1), Block: (256, 1, 1)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void sigmoid_mul_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ gate,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float o = __half2float(out[idx]);
        float g = __half2float(gate[idx]);
        float sig = 1.0f / (1.0f + expf(-g));
        out[idx] = __float2half(o * sig);
    }
}
