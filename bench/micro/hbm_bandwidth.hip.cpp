// HBM Bandwidth Microbenchmark for gfx906
// Tests sequential read, write, and copy at various vector widths.
//
// Build: hipcc -O3 --offload-arch=gfx906 -o hbm_bandwidth hbm_bandwidth.hip.cpp
// Usage: ./hbm_bandwidth [gpu_id=0] [size_mb=256]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Kernels: bandwidth measurement
// ============================================================

// Read-only: sum all elements (prevents optimization away)
__global__ void kernel_read_f32(const float* __restrict__ src, float* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        sum += src[i];
    }
    if (tid == 0) out[0] = sum;
}

__global__ void kernel_read_f32x4(const float4* __restrict__ src, float* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float4 v = src[i];
        sum += v.x + v.y + v.z + v.w;
    }
    if (tid == 0) out[0] = sum;
}

// Write-only
__global__ void kernel_write_f32(float* __restrict__ dst, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        dst[i] = 1.0f;
    }
}

__global__ void kernel_write_f32x4(float4* __restrict__ dst, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float4 val = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    for (int i = tid; i < n; i += stride) {
        dst[i] = val;
    }
}

// Copy
__global__ void kernel_copy_f32(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void kernel_copy_f32x4(const float4* __restrict__ src, float4* __restrict__ dst, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// ============================================================

struct BenchResult {
    const char* name;
    double gb_s;
    double pct_peak;
};

template<typename F>
double time_kernel(F launcher, int warmup, int iters) {
    // Warmup
    for (int i = 0; i < warmup; i++) launcher();
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) launcher();
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ms / iters;
}

int main(int argc, char** argv) {
    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    int size_mb = argc > 2 ? atoi(argv[2]) : 256;

    HIP_CHECK(hipSetDevice(gpu_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, gpu_id));
    double peak_bw = 2.0 * (props.memoryClockRate * 1000.0) * (props.memoryBusWidth / 8.0) / 1e9;

    printf("=== HBM Bandwidth Benchmark ===\n");
    printf("Device: %s (%s)\n", props.name, props.gcnArchName);
    printf("Theoretical peak: %.1f GB/s\n", peak_bw);
    printf("Buffer size: %d MB\n\n", size_mb);

    size_t size_bytes = (size_t)size_mb * 1024 * 1024;
    int n_f32 = size_bytes / sizeof(float);
    int n_f32x4 = size_bytes / sizeof(float4);

    float *d_a, *d_b, *d_out;
    HIP_CHECK(hipMalloc(&d_a, size_bytes));
    HIP_CHECK(hipMalloc(&d_b, size_bytes));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
    HIP_CHECK(hipMemset(d_a, 0, size_bytes));
    HIP_CHECK(hipMemset(d_b, 0, size_bytes));

    int block = 256;
    int grid = (n_f32 + block - 1) / block;
    if (grid > 60 * 4) grid = 60 * 4;  // 60 CUs, ~4 waves/CU

    int grid4 = (n_f32x4 + block - 1) / block;
    if (grid4 > 60 * 4) grid4 = 60 * 4;

    int warmup = 5, iters = 20;

    printf("%-25s %10s %10s\n", "Test", "GB/s", "% Peak");
    printf("%-25s %10s %10s\n", "----", "----", "------");

    // Read f32
    {
        double ms = time_kernel([&](){
            kernel_read_f32<<<grid, block>>>(d_a, d_out, n_f32);
        }, warmup, iters);
        double gb_s = size_bytes / (ms * 1e6);
        printf("%-25s %10.1f %9.1f%%\n", "Read (float)", gb_s, gb_s/peak_bw*100);
    }

    // Read f32x4
    {
        double ms = time_kernel([&](){
            kernel_read_f32x4<<<grid4, block>>>((float4*)d_a, d_out, n_f32x4);
        }, warmup, iters);
        double gb_s = size_bytes / (ms * 1e6);
        printf("%-25s %10.1f %9.1f%%\n", "Read (float4)", gb_s, gb_s/peak_bw*100);
    }

    // Write f32
    {
        double ms = time_kernel([&](){
            kernel_write_f32<<<grid, block>>>(d_a, n_f32);
        }, warmup, iters);
        double gb_s = size_bytes / (ms * 1e6);
        printf("%-25s %10.1f %9.1f%%\n", "Write (float)", gb_s, gb_s/peak_bw*100);
    }

    // Write f32x4
    {
        double ms = time_kernel([&](){
            kernel_write_f32x4<<<grid4, block>>>((float4*)d_a, n_f32x4);
        }, warmup, iters);
        double gb_s = size_bytes / (ms * 1e6);
        printf("%-25s %10.1f %9.1f%%\n", "Write (float4)", gb_s, gb_s/peak_bw*100);
    }

    // Copy f32
    {
        double ms = time_kernel([&](){
            kernel_copy_f32<<<grid, block>>>(d_a, d_b, n_f32);
        }, warmup, iters);
        double gb_s = 2.0 * size_bytes / (ms * 1e6);  // read + write
        printf("%-25s %10.1f %9.1f%%\n", "Copy (float)", gb_s, gb_s/peak_bw*100);
    }

    // Copy f32x4
    {
        double ms = time_kernel([&](){
            kernel_copy_f32x4<<<grid4, block>>>((float4*)d_a, (float4*)d_b, n_f32x4);
        }, warmup, iters);
        double gb_s = 2.0 * size_bytes / (ms * 1e6);
        printf("%-25s %10.1f %9.1f%%\n", "Copy (float4)", gb_s, gb_s/peak_bw*100);
    }

    // hipMemcpy D2D baseline
    {
        double ms = time_kernel([&](){
            HIP_CHECK(hipMemcpy(d_b, d_a, size_bytes, hipMemcpyDeviceToDevice));
        }, warmup, iters);
        double gb_s = 2.0 * size_bytes / (ms * 1e6);
        printf("%-25s %10.1f %9.1f%%\n", "hipMemcpy D2D", gb_s, gb_s/peak_bw*100);
    }

    printf("\n");
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_out));

    return 0;
}
