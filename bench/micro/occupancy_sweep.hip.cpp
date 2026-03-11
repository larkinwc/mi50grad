// Occupancy Sweep: vary VGPR pressure to measure waves/CU vs throughput tradeoff
//
// Build: hipcc -O3 --offload-arch=gfx906 -o occupancy_sweep occupancy_sweep.hip.cpp
// Usage: ./occupancy_sweep [gpu_id=0]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// gfx906 has 256 VGPRs per SIMD, 4 SIMDs per CU
// Wavefront = 64 threads
// VGPRs/wave -> max waves/SIMD:
//   24 VGPRs -> 10 waves/SIMD (capped at 10)
//   28 VGPRs -> 9
//   32 VGPRs -> 8
//   36 VGPRs -> 7
//   40 VGPRs -> 6
//   48 VGPRs -> 5
//   56 VGPRs -> 4
//   64 VGPRs -> 4
//   84 VGPRs -> 3
//   128 VGPRs -> 2
//   256 VGPRs -> 1

// We use template parameter to control VGPR usage via array size
template<int NVGPRS>
__global__ void kernel_compute(const float* __restrict__ src, float* __restrict__ dst, int n) {
    // Force NVGPRS VGPRs by using an array
    float regs[NVGPRS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Initialize
    for (int r = 0; r < NVGPRS; r++)
        regs[r] = src[tid % n] + (float)r;

    // Compute loop to keep all registers live
    for (int iter = 0; iter < 100; iter++) {
        for (int r = 0; r < NVGPRS; r++) {
            regs[r] = regs[r] * 1.0001f + regs[(r+1) % NVGPRS] * 0.0001f;
        }
    }

    // Write out to prevent DCE
    float sum = 0.0f;
    for (int r = 0; r < NVGPRS; r++) sum += regs[r];
    if (tid < n) dst[tid] = sum;
}

template<int NVGPRS>
void run_test(float* d_src, float* d_dst, int n, int gpu_id) {
    int block = 64;  // 1 wavefront per workgroup for cleaner measurement
    int grid = 60 * 4;  // enough to fill all CUs

    // Warmup
    for (int i = 0; i < 5; i++)
        kernel_compute<NVGPRS><<<grid, block>>>(d_src, d_dst, n);
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; i++)
        kernel_compute<NVGPRS><<<grid, block>>>(d_src, d_dst, n);
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / 20;
    printf("  VGPRs ~%-4d: %.3f ms\n", NVGPRS, ms);
}

int main(int argc, char** argv) {
    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(gpu_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, gpu_id));
    printf("=== Occupancy Sweep ===\n");
    printf("Device: %s (%s), %d CUs\n\n", props.name, props.gcnArchName,
           props.multiProcessorCount);

    int n = 1024 * 1024;
    float *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst, n * sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0, n * sizeof(float)));

    printf("Varying register pressure (more VGPRs = fewer waves/SIMD):\n");
    run_test<8>(d_src, d_dst, n, gpu_id);
    run_test<16>(d_src, d_dst, n, gpu_id);
    run_test<24>(d_src, d_dst, n, gpu_id);
    run_test<32>(d_src, d_dst, n, gpu_id);
    run_test<48>(d_src, d_dst, n, gpu_id);
    run_test<64>(d_src, d_dst, n, gpu_id);
    run_test<96>(d_src, d_dst, n, gpu_id);
    run_test<128>(d_src, d_dst, n, gpu_id);

    printf("\n");
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    return 0;
}
