// LDS Bandwidth + Bank Conflict Microbenchmark for gfx906
// Tests LDS throughput with and without XOR swizzle patterns.
//
// Build: hipcc -O3 --offload-arch=gfx906 -o lds_bandwidth lds_bandwidth.hip.cpp
// Usage: ./lds_bandwidth [gpu_id=0]

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

// LDS is 64KB per CU on gfx906, 32 banks, 4 bytes each

// Test 1: Sequential access (no conflicts)
__global__ void lds_sequential(float* out, int iters) {
    __shared__ float smem[8192]; // 32KB
    int tid = threadIdx.x;

    // Init
    for (int i = tid; i < 8192; i += blockDim.x)
        smem[i] = 1.0f;
    __syncthreads();

    float sum = 0.0f;
    for (int iter = 0; iter < iters; iter++) {
        // Each thread reads sequential addresses (stride 1)
        int idx = (tid + iter * blockDim.x) % 8192;
        sum += smem[idx];
    }

    if (tid == 0) out[0] = sum;
}

// Test 2: Strided access (bank conflicts)
__global__ void lds_strided(float* out, int iters, int stride) {
    __shared__ float smem[8192];
    int tid = threadIdx.x;

    for (int i = tid; i < 8192; i += blockDim.x)
        smem[i] = 1.0f;
    __syncthreads();

    float sum = 0.0f;
    for (int iter = 0; iter < iters; iter++) {
        int idx = (tid * stride + iter) % 8192;
        sum += smem[idx];
    }

    if (tid == 0) out[0] = sum;
}

// Test 3: XOR swizzled access (should eliminate conflicts)
__global__ void lds_xor_swizzle(float* out, int iters) {
    __shared__ float smem[8192];
    int tid = threadIdx.x;

    for (int i = tid; i < 8192; i += blockDim.x)
        smem[i] = 1.0f;
    __syncthreads();

    float sum = 0.0f;
    for (int iter = 0; iter < iters; iter++) {
        // XOR swizzle: phys_col = col ^ (row % 32)
        int row = (tid + iter) / 64;
        int col = (tid + iter) % 64;
        int swizzled_col = col ^ (row & 31);
        int idx = row * 64 + swizzled_col;
        idx = idx % 8192;
        sum += smem[idx];
    }

    if (tid == 0) out[0] = sum;
}

// Test 4: Read-write ping-pong
__global__ void lds_readwrite(float* out, int iters) {
    __shared__ float smem_a[4096];
    __shared__ float smem_b[4096];
    int tid = threadIdx.x;

    for (int i = tid; i < 4096; i += blockDim.x) {
        smem_a[i] = 1.0f;
        smem_b[i] = 0.0f;
    }
    __syncthreads();

    for (int iter = 0; iter < iters; iter++) {
        int idx = (tid + iter * blockDim.x) % 4096;
        smem_b[idx] = smem_a[idx] + 1.0f;
        __syncthreads();
        smem_a[idx] = smem_b[idx];
        __syncthreads();
    }

    if (tid == 0) out[0] = smem_a[0];
}

template<typename F>
double time_kernel(F launcher, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) launcher();
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) launcher();
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

int main(int argc, char** argv) {
    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(gpu_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, gpu_id));

    printf("=== LDS Bandwidth Benchmark ===\n");
    printf("Device: %s (%s)\n", props.name, props.gcnArchName);
    printf("CUs: %d, LDS/CU: %d KB\n\n", props.multiProcessorCount,
           (int)(props.sharedMemPerBlock / 1024));

    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));

    int block = 256;  // 4 wavefronts
    int inner_iters = 10000;
    int warmup = 5, bench_iters = 20;

    printf("%-30s %10s\n", "Test", "Time (ms)");
    printf("%-30s %10s\n", "----", "---------");

    // Sequential
    {
        double ms = time_kernel([&](){
            lds_sequential<<<1, block>>>(d_out, inner_iters);
        }, warmup, bench_iters);
        printf("%-30s %10.3f\n", "Sequential (stride 1)", ms);
    }

    // Various strides
    for (int stride : {1, 2, 4, 8, 16, 32}) {
        char name[64];
        snprintf(name, sizeof(name), "Strided (stride %d)", stride);
        double ms = time_kernel([&](){
            lds_strided<<<1, block>>>(d_out, inner_iters, stride);
        }, warmup, bench_iters);
        printf("%-30s %10.3f\n", name, ms);
    }

    // XOR swizzle
    {
        double ms = time_kernel([&](){
            lds_xor_swizzle<<<1, block>>>(d_out, inner_iters);
        }, warmup, bench_iters);
        printf("%-30s %10.3f\n", "XOR swizzled", ms);
    }

    // Read-write
    {
        double ms = time_kernel([&](){
            lds_readwrite<<<1, block>>>(d_out, inner_iters / 10);
        }, warmup, bench_iters);
        printf("%-30s %10.3f\n", "Read-Write ping-pong", ms);
    }

    printf("\n");
    HIP_CHECK(hipFree(d_out));

    return 0;
}
