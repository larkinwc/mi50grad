// HIP host code for launching ISA probe kernels
// Compiles assembly .s -> .hsaco, loads via HIP, launches, reports results.
//
// Build: hipcc -o run_probes run_probes.hip.cpp
// Usage: ./run_probes [num_iters=100000] [gpu_id=0] [hsaco_dir=.]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                err, hipGetErrorString(err), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

struct ProbeResult {
    uint32_t elapsed_lo;
    uint32_t elapsed_hi;
    uint32_t acc_value;  // anti-DCE
};

struct ProbeInfo {
    const char* name;
    const char* asm_file;
    const char* kernel_name;
    int ops_per_iter;         // total ops per iteration (8 instructions * ops/instruction)
    const char* description;
};

static ProbeInfo probes[] = {
    {"dot2_f32_f16",  "probe_dot2_f32_f16.s",  "probe_dot2_f32_f16",
     8 * 2, "v_dot2_f32_f16: 2xFP16 dot -> FP32"},
    {"dot4_i32_i8",   "probe_dot4_i32_i8.s",   "probe_dot4_i32_i8",
     8 * 4, "v_dot4_i32_i8: 4xINT8 dot -> INT32"},
    {"dot8_i32_i4",   "probe_dot8_i32_i4.s",   "probe_dot8_i32_i4",
     8 * 8, "v_dot8_i32_i4: 8xINT4 dot -> INT32"},
    {"pk_fma_f16",    "probe_pk_fma_f16.s",     "probe_pk_fma_f16",
     8 * 2, "v_pk_fma_f16: packed 2xFP16 FMA"},
    {"fmac_f32",      "probe_fmac_f32.s",       "probe_fmac_f32",
     8 * 1, "v_fmac_f32: FP32 FMA"},
    {"exp_f32",       "probe_exp_f32.s",        "probe_exp_f32",
     8 * 1, "v_exp_f32: MUFU transcendental"},
    {"dpp",           "probe_dpp.s",            "probe_dpp",
     8,    "DPP mov/add with row_shr/bcast"},
};
static const int NUM_PROBES = sizeof(probes) / sizeof(probes[0]);

static bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static int build_hsaco(const char* asm_file, const char* hsaco_file) {
    char obj_file[256];
    snprintf(obj_file, sizeof(obj_file), "%.*s.o",
             (int)(strlen(asm_file) - 2), asm_file);

    char cmd[1024];
    // Assemble
    snprintf(cmd, sizeof(cmd),
             "/opt/rocm/llvm/bin/llvm-mc --triple=amdgcn-amd-amdhsa --mcpu=gfx906 "
             "--filetype=obj %s -o %s 2>&1",
             asm_file, obj_file);
    printf("  [asm] %s\n", cmd);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "  Assembly FAILED for %s (rc=%d)\n", asm_file, rc);
        return rc;
    }

    // Link
    snprintf(cmd, sizeof(cmd),
             "/opt/rocm/llvm/bin/ld.lld -shared %s -o %s 2>&1",
             obj_file, hsaco_file);
    printf("  [link] %s\n", cmd);
    rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "  Linking FAILED for %s (rc=%d)\n", obj_file, rc);
    }
    return rc;
}

int main(int argc, char** argv) {
    uint32_t num_iters = 100000;
    int gpu_id = 0;
    const char* hsaco_dir = nullptr;  // directory with pre-built .hsaco files

    if (argc > 1) num_iters = atoi(argv[1]);
    if (argc > 2) gpu_id = atoi(argv[2]);
    if (argc > 3) hsaco_dir = argv[3];

    printf("=== mi50grad ISA Probe Suite ===\n");
    printf("Iterations: %u\n", num_iters);
    printf("GPU ID: %d\n\n", gpu_id);

    HIP_CHECK(hipSetDevice(gpu_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, gpu_id));
    printf("Device: %s\n", props.name);
    printf("GCN Arch: %s\n", props.gcnArchName);
    printf("CUs: %d\n", props.multiProcessorCount);
    printf("Clock: %d MHz\n", props.clockRate / 1000);
    printf("Memory Clock: %d MHz\n", props.memoryClockRate / 1000);
    printf("Memory Bus: %d bits\n", props.memoryBusWidth);
    double peak_bw = 2.0 * (props.memoryClockRate * 1000.0) * (props.memoryBusWidth / 8.0) / 1e9;
    printf("Theoretical HBM BW: %.1f GB/s\n\n", peak_bw);

    // Allocate output buffer on device
    ProbeResult* d_result;
    ProbeResult h_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(ProbeResult)));

    printf("%-20s %-8s %-15s %-15s %-12s\n",
           "Probe", "Status", "Cycles", "Instr/Cyc/CU", "Description");
    printf("%-20s %-8s %-15s %-15s %-12s\n",
           "----", "------", "------", "------------", "-----------");

    for (int i = 0; i < NUM_PROBES; i++) {
        ProbeInfo& p = probes[i];
        printf("\n--- %s ---\n", p.description);

        // Find or build HSACO
        char hsaco_file[512];
        bool found = false;

        // Check pre-built HSACO directory first
        if (hsaco_dir) {
            snprintf(hsaco_file, sizeof(hsaco_file), "%s/%.*s.hsaco",
                     hsaco_dir, (int)(strlen(p.asm_file) - 2), p.asm_file);
            if (file_exists(hsaco_file)) {
                printf("  [pre-built] %s\n", hsaco_file);
                found = true;
            }
        }

        // Fallback: try to build from .s
        if (!found) {
            snprintf(hsaco_file, sizeof(hsaco_file), "%.*s.hsaco",
                     (int)(strlen(p.asm_file) - 2), p.asm_file);
            if (build_hsaco(p.asm_file, hsaco_file) != 0) {
                printf("%-20s %-8s (assembly failed - instruction may not be supported)\n",
                       p.name, "FAIL");
                continue;
            }
        }

        // Load module
        hipModule_t module;
        hipError_t err = hipModuleLoad(&module, hsaco_file);
        if (err != hipSuccess) {
            printf("%-20s %-8s (module load failed: %s)\n",
                   p.name, "FAIL", hipGetErrorString(err));
            continue;
        }

        // Get kernel function
        hipFunction_t kernel;
        err = hipModuleGetFunction(&kernel, module, p.kernel_name);
        if (err != hipSuccess) {
            printf("%-20s %-8s (kernel not found: %s)\n",
                   p.name, "FAIL", hipGetErrorString(err));
            (void)hipModuleUnload(module);
            continue;
        }

        // Clear result
        HIP_CHECK(hipMemset(d_result, 0, sizeof(ProbeResult)));

        // Launch: 1 workgroup, 64 threads (1 wavefront)
        // Pass args via kernelParams (array of host pointers to each arg value)
        uint32_t h_num_iters = num_iters;
        uint32_t h_pad = 0;
        uint64_t h_output_ptr = (uint64_t)d_result;
        void* kernelParams[] = { &h_num_iters, &h_pad, &h_output_ptr };

        err = hipModuleLaunchKernel(kernel,
                                     1, 1, 1,    // grid
                                     64, 1, 1,   // block (1 wavefront)
                                     0,          // shared mem
                                     0,          // stream
                                     kernelParams,
                                     nullptr);

        if (err != hipSuccess) {
            printf("%-20s %-8s (launch failed: %s — instruction may TRAP)\n",
                   p.name, "TRAP?", hipGetErrorString(err));
            (void)hipModuleUnload(module);
            continue;
        }

        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("%-20s %-8s (sync failed: %s — instruction likely TRAPPED)\n",
                   p.name, "TRAP", hipGetErrorString(err));
            (void)hipModuleUnload(module);
            continue;
        }

        // Read result
        HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(ProbeResult), hipMemcpyDeviceToHost));

        uint64_t elapsed = ((uint64_t)h_result.elapsed_hi << 32) | h_result.elapsed_lo;

        if (elapsed == 0) {
            printf("%-20s %-8s (zero elapsed — timing may be broken)\n",
                   p.name, "WARN");
        } else {
            // Instructions per cycle:
            // 8 instructions per iteration, num_iters iterations
            // elapsed is in GPU clock cycles
            double total_instr = 8.0 * num_iters;
            double ipc = total_instr / (double)elapsed;

            printf("%-20s %-8s %-15lu %-15.4f %s\n",
                   p.name, "OK", (unsigned long)elapsed, ipc, p.description);

            // Also compute effective FLOPS/IOPS
            double total_ops = (double)p.ops_per_iter * num_iters * 64; // 64 lanes
            double gpu_clock_hz = props.clockRate * 1000.0; // clockRate is in kHz
            double seconds = (double)elapsed / gpu_clock_hz;
            double tops = total_ops / seconds / 1e12;
            printf("  -> %.3f T ops/s (%.1f%% of theoretical)\n",
                   tops, tops / 26.8 * 100.0);
        }

        (void)hipModuleUnload(module);
    }

    printf("\n=== Probe Suite Complete ===\n");

    HIP_CHECK(hipFree(d_result));

    return 0;
}
