// Kernel Dispatch Latency Microbenchmark
// Measures overhead of hipModuleLaunchKernel for empty/minimal kernels.
//
// Build: hipcc -O3 --offload-arch=gfx906 -o dispatch_latency dispatch_latency.hip.cpp
// Usage: ./dispatch_latency [gpu_id=0] [num_launches=10000]

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

__global__ void kernel_empty() {}

__global__ void kernel_tiny(float* out) {
    if (threadIdx.x == 0) *out = 1.0f;
}

int main(int argc, char** argv) {
    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    int num_launches = argc > 2 ? atoi(argv[2]) : 10000;

    HIP_CHECK(hipSetDevice(gpu_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, gpu_id));
    printf("=== Dispatch Latency Benchmark ===\n");
    printf("Device: %s (%s)\n", props.name, props.gcnArchName);
    printf("Launches: %d\n\n", num_launches);

    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));

    // Warmup
    for (int i = 0; i < 100; i++) {
        kernel_empty<<<1, 64>>>();
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Test 1: Empty kernel, launch+sync each time
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_launches; i++) {
            kernel_empty<<<1, 64>>>();
            HIP_CHECK(hipDeviceSynchronize());
        }
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        printf("Empty kernel (launch+sync):  %.2f us/launch\n", us / num_launches);
    }

    // Test 2: Empty kernel, batch launch then sync
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_launches; i++) {
            kernel_empty<<<1, 64>>>();
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        printf("Empty kernel (batch+sync):   %.2f us/launch\n", us / num_launches);
    }

    // Test 3: Tiny kernel with memory write
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_launches; i++) {
            kernel_tiny<<<1, 64>>>(d_out);
            HIP_CHECK(hipDeviceSynchronize());
        }
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        printf("Tiny kernel (launch+sync):   %.2f us/launch\n", us / num_launches);
    }

    // Test 4: Multiple workgroups
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_launches; i++) {
            kernel_empty<<<60, 64>>>();  // 60 CUs
            HIP_CHECK(hipDeviceSynchronize());
        }
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        printf("Empty 60-WG (launch+sync):   %.2f us/launch\n", us / num_launches);
    }

    printf("\n");
    HIP_CHECK(hipFree(d_out));

    return 0;
}
