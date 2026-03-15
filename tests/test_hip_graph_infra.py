#!/usr/bin/env python3
"""
tests/test_hip_graph_infra.py — HIP Graph API infrastructure tests for gfx906.

Tests:
  1. HIP graph API availability on gfx906 (hipGraphCreate, hipStreamBeginCapture, etc.)
  2. Capture a simple single-kernel graph and replay it, verifying correctness
  3. Capture a multi-kernel graph (e.g., a sequence of elementwise ops) and replay
  4. Test mutable parameter update via hipGraphExecKernelNodeSetParams
  5. Benchmark graph replay vs direct launch for a compute segment

Validation assertions fulfilled:
  VAL-GRAPH-001: HIP graph capture succeeds on gfx906
  VAL-GRAPH-002: Mutable parameters update correctly between replays
  VAL-GRAPH-003: Graph replay correctness (single + multi kernel)

USAGE:
    # Stop vLLM first, then:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    #     -e HIP_VISIBLE_DEVICES=0 \\
    #     -v /opt/mi50grad:/opt/mi50grad \\
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_hip_graph_infra.py'
"""

import sys
import os
import ctypes
import time
import math
import numpy as np
from pathlib import Path

# Unbuffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

from src.runtime.hip_dispatch import HIPRuntime
from src.runtime.hip_graph_dispatch import (
    HIPGraphRuntime, GraphSegment,
    hipStreamCaptureModeRelaxed, hipStreamCaptureModeGlobal,
    hipGraphNodeTypeKernel, hipKernelNodeParams
)

# ============================================================
# Config
# ============================================================
DEVICE_ID = 0
BENCH_WARMUP = 10
BENCH_ITERS  = 200

# ============================================================
# Test utilities
# ============================================================

def print_header(title, width=72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_pass(msg=""):
    print(f"  PASS{': ' + msg if msg else ''}")


def print_fail(msg=""):
    print(f"  FAIL{': ' + msg if msg else ''}")


results = {}  # test_name → bool


def record(name, passed, msg=""):
    results[name] = passed
    if passed:
        print_pass(msg)
    else:
        print_fail(msg)


# ============================================================
# Helper: Load a simple HIP kernel for testing
# ============================================================

SIMPLE_KERNEL_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Simple FP16 scale kernel: y[i] = scale * x[i]
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void scale_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    float scale,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = __half((float)in[i] * scale);
    }
}

// FP16 add kernel: z[i] = x[i] + y[i]
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void add_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ y,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = __half((float)x[i] + (float)y[i]);
    }
}

// FP16 AXPY kernel: y[i] = alpha * x[i] + beta * y[i]
// alpha at params[2], beta at params[3] — these will be the "mutable" params
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void axpy_fp16(
    __half* __restrict__ y,     // in+out
    const __half* __restrict__ x,
    float alpha,
    float beta,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = (float)x[i];
        float yi = (float)y[i];
        y[i] = __half(alpha * xi + beta * yi);
    }
}
"""

def build_test_kernel(hip: HIPRuntime) -> tuple:
    """Build the test kernel as HSACO and load it via hipModuleLoad.

    Returns (module_handle, func_scale, func_add, func_axpy, hsaco_path).
    """
    import subprocess

    kernel_dir = Path("/opt/mi50grad/build/test_kernels")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    src_path   = kernel_dir / "test_graph_kernels.hip"
    hsaco_path = kernel_dir / "test_graph_kernels.hsaco"

    src_path.write_text(SIMPLE_KERNEL_SRC)

    # Build HSACO (code object loadable via hipModuleLoad)
    ret = subprocess.run(
        ["/opt/rocm/bin/hipcc", "--genco",
         "--offload-arch=gfx906",
         "-O3",
         "-o", str(hsaco_path), str(src_path)],
        capture_output=True, text=True
    )
    if ret.returncode != 0:
        print(f"  hipcc stdout: {ret.stdout}")
        print(f"  hipcc stderr: {ret.stderr}")
        raise RuntimeError(f"hipcc --genco failed: {ret.returncode}")

    print(f"  HSACO built: {hsaco_path} ({hsaco_path.stat().st_size} bytes)")

    # Load as HIP module
    module = hip.module_load(str(hsaco_path))
    func_scale  = hip.module_get_function(module, "scale_fp16")
    func_add    = hip.module_get_function(module, "add_fp16")
    func_axpy   = hip.module_get_function(module, "axpy_fp16")
    return module, func_scale, func_add, func_axpy, str(hsaco_path)


# ============================================================
# TEST 1: HIP Graph API availability check
# ============================================================

def test_graph_api_availability(graph_rt: HIPGraphRuntime):
    """VAL-GRAPH-001 precondition: verify all needed HIP graph functions are accessible."""
    print_header("TEST 1: HIP Graph API Availability")

    functions_needed = [
        "hipGraphCreate",
        "hipStreamBeginCapture",
        "hipStreamEndCapture",
        "hipGraphInstantiate",
        "hipGraphLaunch",
        "hipGraphExecKernelNodeSetParams",
        "hipGraphExecDestroy",
        "hipGraphDestroy",
        "hipGraphGetNodes",
        "hipGraphNodeGetType",
        "hipGraphKernelNodeGetParams",
    ]

    all_ok = True
    for fn in functions_needed:
        available = hasattr(graph_rt._lib, fn)
        status = "OK" if available else "MISSING"
        print(f"  {fn}: {status}")
        if not available:
            all_ok = False

    # Try creating and destroying a graph
    try:
        g = graph_rt.graph_create()
        graph_rt.graph_destroy(g)
        print("  hipGraphCreate + hipGraphDestroy: OK")
    except Exception as e:
        print(f"  hipGraphCreate + hipGraphDestroy: FAILED ({e})")
        all_ok = False

    record("api_availability", all_ok,
           "All HIP graph functions available on gfx906/ROCm 7.1" if all_ok
           else "Some functions missing")
    return all_ok


# ============================================================
# TEST 2: Single-kernel graph capture and replay
# ============================================================

def test_single_kernel_graph(hip: HIPRuntime, graph_rt: HIPGraphRuntime,
                              func_scale: int):
    """VAL-GRAPH-001 + VAL-GRAPH-003: capture scale_fp16 as a graph and replay."""
    print_header("TEST 2: Single-Kernel Graph Capture and Replay")

    N = 1024
    SCALE = 2.5

    # Allocate device buffers
    d_in  = hip.malloc(N * 2)
    d_out = hip.malloc(N * 2)

    # Fill input with known values
    host_in = np.random.randn(N).astype(np.float16)
    hip.memcpy_h2d(d_in, host_in.tobytes(), N * 2)
    hip.memset(d_out, 0, N * 2)

    # Build ctypes params for scale_fp16(out, in, scale, N)
    p_out   = ctypes.c_uint64(d_out)
    p_in    = ctypes.c_uint64(d_in)
    p_scale = ctypes.c_float(SCALE)
    p_N     = ctypes.c_int(N)
    params  = [p_out, p_in, p_scale, p_N]

    n_params = len(params)
    params_arr = (ctypes.c_void_p * n_params)()
    for i, p in enumerate(params):
        params_arr[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)

    grid  = ((N + 255) // 256, 1, 1)
    block = (256, 1, 1)

    # Create a stream for capture
    stream = hip.stream_create()

    try:
        # === Direct launch (reference) ===
        hip.memset(d_out, 0, N * 2)
        hip.launch_kernel(func_scale, grid, block, params, stream=stream)
        hip.stream_synchronize(stream)
        result_direct = np.frombuffer(ctypes.create_string_buffer(N * 2), dtype=np.float16).copy()
        buf = ctypes.create_string_buffer(N * 2)
        hip.memcpy_d2h(buf, d_out, N * 2)
        result_direct = np.frombuffer(buf.raw, dtype=np.float16).copy()

        expected = (host_in.astype(np.float32) * SCALE).astype(np.float16)
        err_direct = float(np.max(np.abs(result_direct.astype(np.float32) -
                                          expected.astype(np.float32))))
        print(f"  Direct launch max_abs_err: {err_direct:.2e}")

        # === Graph capture ===
        hip.memset(d_out, 0, N * 2)

        seg = GraphSegment(graph_rt, DEVICE_ID)
        seg.begin_capture(stream, mode=hipStreamCaptureModeRelaxed)

        # Capture the kernel launch
        hip.launch_kernel(func_scale, grid, block, params, stream=stream)

        seg.end_capture(stream)
        seg.instantiate()

        n_nodes = seg.num_kernel_nodes()
        print(f"  Graph captured: {n_nodes} kernel node(s)")

        # === Graph replay ===
        hip.memset(d_out, 0, N * 2)
        seg.replay(stream)
        hip.stream_synchronize(stream)

        buf2 = ctypes.create_string_buffer(N * 2)
        hip.memcpy_d2h(buf2, d_out, N * 2)
        result_graph = np.frombuffer(buf2.raw, dtype=np.float16).copy()

        err_graph = float(np.max(np.abs(result_graph.astype(np.float32) -
                                         expected.astype(np.float32))))
        print(f"  Graph replay max_abs_err: {err_graph:.2e}")

        passed = (n_nodes == 1 and err_graph < 1e-2)
        record("single_kernel_graph", passed,
               f"capture OK, replay max_err={err_graph:.2e}" if passed
               else f"FAILED: nodes={n_nodes}, err={err_graph:.2e}")

        seg.cleanup()

    finally:
        hip.stream_destroy(stream)
        hip.free(d_in)
        hip.free(d_out)


# ============================================================
# TEST 3: Multi-kernel graph capture and replay
# ============================================================

def test_multi_kernel_graph(hip: HIPRuntime, graph_rt: HIPGraphRuntime,
                             func_scale: int, func_add: int):
    """VAL-GRAPH-003: capture scale → add → scale sequence as a graph and replay."""
    print_header("TEST 3: Multi-Kernel Graph Capture and Replay")

    N = 2048
    SCALE1 = 1.5
    SCALE2 = 0.7

    d_x   = hip.malloc(N * 2)
    d_y   = hip.malloc(N * 2)
    d_tmp = hip.malloc(N * 2)
    d_out = hip.malloc(N * 2)

    host_x = np.random.randn(N).astype(np.float16)
    host_y = np.random.randn(N).astype(np.float16)
    hip.memcpy_h2d(d_x, host_x.tobytes(), N * 2)
    hip.memcpy_h2d(d_y, host_y.tobytes(), N * 2)
    hip.memset(d_tmp, 0, N * 2)
    hip.memset(d_out, 0, N * 2)

    grid  = ((N + 255) // 256, 1, 1)
    block = (256, 1, 1)

    # Params: scale1: tmp = x * SCALE1
    p_tmp   = ctypes.c_uint64(d_tmp)
    p_x     = ctypes.c_uint64(d_x)
    p_y     = ctypes.c_uint64(d_y)
    p_out   = ctypes.c_uint64(d_out)
    p_s1    = ctypes.c_float(SCALE1)
    p_s2    = ctypes.c_float(SCALE2)
    p_N     = ctypes.c_int(N)
    params_scale1 = [p_tmp, p_x, p_s1, p_N]    # scale: tmp = x * 1.5
    params_add    = [p_tmp, p_tmp, p_y, p_N]    # add: tmp = tmp + y
    params_scale2 = [p_out, p_tmp, p_s2, p_N]   # scale: out = tmp * 0.7

    # Compute reference
    expected = ((host_x.astype(np.float32) * SCALE1 +
                 host_y.astype(np.float32)) * SCALE2).astype(np.float16)

    stream = hip.stream_create()

    try:
        seg = GraphSegment(graph_rt, DEVICE_ID)
        seg.begin_capture(stream, mode=hipStreamCaptureModeRelaxed)

        hip.launch_kernel(func_scale, grid, block, params_scale1, stream=stream)
        hip.launch_kernel(func_add,   grid, block, params_add,    stream=stream)
        hip.launch_kernel(func_scale, grid, block, params_scale2, stream=stream)

        seg.end_capture(stream)
        seg.instantiate()

        n_nodes = seg.num_kernel_nodes()
        print(f"  Graph captured: {n_nodes} kernel node(s)")

        # Replay
        hip.memset(d_tmp, 0, N * 2)
        hip.memset(d_out, 0, N * 2)
        seg.replay(stream)
        hip.stream_synchronize(stream)

        buf = ctypes.create_string_buffer(N * 2)
        hip.memcpy_d2h(buf, d_out, N * 2)
        result = np.frombuffer(buf.raw, dtype=np.float16).copy()

        err = float(np.max(np.abs(result.astype(np.float32) -
                                   expected.astype(np.float32))))
        print(f"  Multi-kernel graph replay max_abs_err: {err:.2e}")

        # Also check intermediate tmp buffer
        buf_tmp = ctypes.create_string_buffer(N * 2)
        hip.memcpy_d2h(buf_tmp, d_tmp, N * 2)
        result_tmp = np.frombuffer(buf_tmp.raw, dtype=np.float16).copy()
        expected_tmp = (host_x.astype(np.float32) * SCALE1 +
                        host_y.astype(np.float32)).astype(np.float16)
        err_tmp = float(np.max(np.abs(result_tmp.astype(np.float32) -
                                       expected_tmp.astype(np.float32))))
        print(f"  Intermediate buffer max_abs_err: {err_tmp:.2e}")

        passed = (n_nodes == 3 and err < 5e-3)
        record("multi_kernel_graph", passed,
               f"{n_nodes} nodes captured, max_err={err:.2e}" if passed
               else f"FAILED: nodes={n_nodes}, err={err:.2e}")

        seg.cleanup()

    finally:
        hip.stream_destroy(stream)
        hip.free(d_x)
        hip.free(d_y)
        hip.free(d_tmp)
        hip.free(d_out)


# ============================================================
# TEST 4: Mutable parameter update via hipGraphExecKernelNodeSetParams
# ============================================================

def test_mutable_params(hip: HIPRuntime, graph_rt: HIPGraphRuntime,
                         func_axpy: int):
    """VAL-GRAPH-002: capture axpy, then update alpha/beta between replays.

    axpy kernel: y[i] = alpha * x[i] + beta * y_old[i]
    We capture with alpha=1.0, beta=0.0 (copy x to y).
    Then update to alpha=2.0, beta=1.0 (y = 2*x + y_old).
    """
    print_header("TEST 4: Mutable Parameter Update (hipGraphExecKernelNodeSetParams)")

    N = 512
    ALPHA_INIT = 1.0
    BETA_INIT  = 0.0
    ALPHA_NEW  = 2.0
    BETA_NEW   = 1.0

    d_y = hip.malloc(N * 2)
    d_x = hip.malloc(N * 2)

    host_x = np.ones(N, dtype=np.float16) * 3.0
    host_y_init = np.ones(N, dtype=np.float16) * 5.0

    hip.memcpy_h2d(d_x, host_x.tobytes(), N * 2)
    hip.memcpy_h2d(d_y, host_y_init.tobytes(), N * 2)

    grid  = ((N + 255) // 256, 1, 1)
    block = (256, 1, 1)

    # Build ctypes params: axpy(y, x, alpha, beta, N)
    p_y     = ctypes.c_uint64(d_y)
    p_x     = ctypes.c_uint64(d_x)
    p_alpha = ctypes.c_float(ALPHA_INIT)
    p_beta  = ctypes.c_float(BETA_INIT)
    p_N     = ctypes.c_int(N)
    params  = [p_y, p_x, p_alpha, p_beta, p_N]

    stream = hip.stream_create()

    try:
        # === Capture graph with initial alpha/beta ===
        seg = GraphSegment(graph_rt, DEVICE_ID)
        seg.begin_capture(stream, mode=hipStreamCaptureModeRelaxed)
        hip.launch_kernel(func_axpy, grid, block, params, stream=stream)
        seg.end_capture(stream)
        seg.instantiate()

        n_nodes = seg.num_kernel_nodes()
        print(f"  Graph captured: {n_nodes} kernel node(s)")
        if n_nodes != 1:
            record("mutable_params", False, f"Expected 1 kernel node, got {n_nodes}")
            return

        kernel_nodes = seg._kernel_nodes
        node = kernel_nodes[0]

        # === Step 1: Replay with initial params (alpha=1.0, beta=0.0) ===
        # Expected: y = 1.0 * x + 0.0 * y_old = x = [3.0, 3.0, ...]
        hip.memcpy_h2d(d_y, host_y_init.tobytes(), N * 2)
        seg.replay(stream)
        hip.stream_synchronize(stream)

        buf = ctypes.create_string_buffer(N * 2)
        hip.memcpy_d2h(buf, d_y, N * 2)
        result1 = np.frombuffer(buf.raw, dtype=np.float16).copy()
        expected1 = np.ones(N, dtype=np.float32) * (ALPHA_INIT * 3.0 + BETA_INIT * 5.0)
        err1 = float(np.max(np.abs(result1.astype(np.float32) - expected1)))
        print(f"  Step 1 (alpha=1.0, beta=0.0) result[0]={float(result1[0]):.3f}, "
              f"expected={expected1[0]:.3f}, err={err1:.2e}")

        # === Step 2: Update alpha and beta in graph, replay ===
        # Build updated hipKernelNodeParams with new alpha and beta
        cur_params = seg.get_kernel_params(node)
        print(f"  Original kernel func ptr: {cur_params.func}")

        # We need to create new ctypes param values with the updated alpha/beta
        p_alpha_new = ctypes.c_float(ALPHA_NEW)
        p_beta_new  = ctypes.c_float(BETA_NEW)
        params_new  = [p_y, p_x, p_alpha_new, p_beta_new, p_N]

        n_params = len(params_new)
        params_arr_new = (ctypes.c_void_p * n_params)()
        for i, p in enumerate(params_new):
            params_arr_new[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)

        # Build new hipKernelNodeParams
        new_kparams = hipKernelNodeParams()
        new_kparams.blockDimX = block[0]
        new_kparams.blockDimY = block[1]
        new_kparams.blockDimZ = block[2]
        new_kparams.gridDimX  = grid[0]
        new_kparams.gridDimY  = grid[1]
        new_kparams.gridDimZ  = grid[2]
        new_kparams.func      = cur_params.func  # same function
        new_kparams.kernelParams = ctypes.cast(
            params_arr_new, ctypes.c_void_p
        ).value
        new_kparams.sharedMemBytes = 0
        new_kparams.extra = None

        seg.update_kernel_params(node, new_kparams)
        print(f"  Updated kernel params: alpha={ALPHA_NEW}, beta={BETA_NEW}")

        # Replay with updated params
        # d_y currently holds result1 = [3.0, ...]; now: y = 2.0 * x + 1.0 * y_old
        # (y_old = result of step 1 = 3.0)
        # expected: 2.0 * 3.0 + 1.0 * 3.0 = 9.0
        seg.replay(stream)
        hip.stream_synchronize(stream)

        buf2 = ctypes.create_string_buffer(N * 2)
        hip.memcpy_d2h(buf2, d_y, N * 2)
        result2 = np.frombuffer(buf2.raw, dtype=np.float16).copy()
        expected2 = np.ones(N, dtype=np.float32) * (ALPHA_NEW * 3.0 + BETA_NEW * 3.0)  # 9.0
        err2 = float(np.max(np.abs(result2.astype(np.float32) - expected2)))
        print(f"  Step 2 (alpha=2.0, beta=1.0) result[0]={float(result2[0]):.3f}, "
              f"expected={expected2[0]:.3f}, err={err2:.2e}")

        passed = (err1 < 1e-2 and err2 < 1e-2)
        record("mutable_params", passed,
               f"step1_err={err1:.2e}, step2_err={err2:.2e}" if passed
               else f"FAILED: step1_err={err1:.2e}, step2_err={err2:.2e}")

        # Keep ctypes objects alive until after replay
        _ = (params_arr_new, p_alpha_new, p_beta_new)

        seg.cleanup()

    finally:
        hip.stream_destroy(stream)
        hip.free(d_x)
        hip.free(d_y)


# ============================================================
# TEST 5: Multi-step mutable parameter correctness (simulate RoPE pos updates)
# ============================================================

def test_multi_step_mutable(hip: HIPRuntime, graph_rt: HIPGraphRuntime,
                              func_scale: int):
    """VAL-GRAPH-002: Simulate position-dependent parameter updates across 10 decode steps.

    Use scale kernel with different scale factors per step (like RoPE cos/sin offsets).
    Capture once, then update the scale between replays and verify correct output.
    """
    print_header("TEST 5: Multi-Step Mutable Parameter Updates (10 steps)")

    N = 1024
    STEPS = 10

    d_in  = hip.malloc(N * 2)
    d_out = hip.malloc(N * 2)

    # Fill input with ascending values
    host_in = np.arange(N, dtype=np.float16) * 0.001 + 0.1
    hip.memcpy_h2d(d_in, host_in.tobytes(), N * 2)

    grid  = ((N + 255) // 256, 1, 1)
    block = (256, 1, 1)

    # Build mutable ctypes params
    p_out   = ctypes.c_uint64(d_out)
    p_in    = ctypes.c_uint64(d_in)
    p_scale = ctypes.c_float(1.0)  # initial scale (will be updated)
    p_N     = ctypes.c_int(N)
    params  = [p_out, p_in, p_scale, p_N]

    # Keep params alive
    param_objects = list(params)

    stream = hip.stream_create()

    try:
        # Capture graph with initial scale=1.0
        seg = GraphSegment(graph_rt, DEVICE_ID)
        seg.begin_capture(stream, mode=hipStreamCaptureModeRelaxed)
        hip.launch_kernel(func_scale, grid, block, params, stream=stream)
        seg.end_capture(stream)
        seg.instantiate()

        kernel_nodes = seg._kernel_nodes
        if not kernel_nodes:
            record("multi_step_mutable", False, "No kernel nodes captured")
            return

        node = kernel_nodes[0]
        cur_params = seg.get_kernel_params(node)

        # Run 10 steps, each with a different scale
        all_pass = True
        for step in range(STEPS):
            scale_value = 1.0 + step * 0.5  # 1.0, 1.5, 2.0, ..., 5.5

            # Build new params with updated scale
            p_scale_new = ctypes.c_float(scale_value)
            params_new  = [p_out, p_in, p_scale_new, p_N]
            n = len(params_new)
            params_arr = (ctypes.c_void_p * n)()
            for i, p in enumerate(params_new):
                params_arr[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)

            new_kparams = hipKernelNodeParams()
            new_kparams.blockDimX = block[0]
            new_kparams.blockDimY = block[1]
            new_kparams.blockDimZ = block[2]
            new_kparams.gridDimX  = grid[0]
            new_kparams.gridDimY  = grid[1]
            new_kparams.gridDimZ  = grid[2]
            new_kparams.func      = cur_params.func
            new_kparams.kernelParams = ctypes.cast(
                params_arr, ctypes.c_void_p
            ).value
            new_kparams.sharedMemBytes = 0
            new_kparams.extra = None

            seg.update_kernel_params(node, new_kparams)

            hip.memset(d_out, 0, N * 2)
            seg.replay(stream)
            hip.stream_synchronize(stream)

            buf = ctypes.create_string_buffer(N * 2)
            hip.memcpy_d2h(buf, d_out, N * 2)
            result = np.frombuffer(buf.raw, dtype=np.float16).copy()

            expected = (host_in.astype(np.float32) * scale_value).astype(np.float16)
            err = float(np.max(np.abs(result.astype(np.float32) -
                                       expected.astype(np.float32))))
            step_pass = err < 1e-2
            if not step_pass:
                all_pass = False
            print(f"  Step {step+1:2d}: scale={scale_value:.1f}, "
                  f"result[0]={float(result[0]):.4f}, "
                  f"expected={float(expected[0]):.4f}, "
                  f"err={err:.2e}  {'OK' if step_pass else 'FAIL'}")

            # Keep objects alive
            _ = (p_scale_new, params_arr, new_kparams, params_new)

        record("multi_step_mutable", all_pass,
               "All 10 steps correct with updated params" if all_pass
               else "Some steps incorrect after param update")

        seg.cleanup()

    finally:
        hip.stream_destroy(stream)
        hip.free(d_in)
        hip.free(d_out)


# ============================================================
# TEST 6: Benchmark graph replay vs direct launch
# ============================================================

def test_benchmark_graph_vs_direct(hip: HIPRuntime, graph_rt: HIPGraphRuntime,
                                    func_scale: int, func_add: int):
    """VAL-GRAPH-003: Benchmark graph replay vs direct launch.

    Measures per-replay vs per-direct-launch overhead for a multi-kernel sequence.
    """
    print_header("TEST 6: Benchmark — Graph Replay vs Direct Launch")

    N = 5120  # Typical hidden_size for Qwen3.5-27B

    # Number of kernels in sequence (simulating a compute segment between allreduces)
    # A typical attention segment has ~7-9 kernel launches
    N_KERNELS = 8

    # Allocate buffers
    bufs = [hip.malloc(N * 2) for _ in range(N_KERNELS + 1)]
    host_data = np.random.randn(N).astype(np.float16)
    hip.memcpy_h2d(bufs[0], host_data.tobytes(), N * 2)

    grid  = ((N + 255) // 256, 1, 1)
    block = (256, 1, 1)

    # Build params for alternating scale/add operations
    scale_val = 1.1
    params_list = []
    for k in range(N_KERNELS):
        if k % 2 == 0:
            # scale: bufs[k+1] = bufs[k] * scale_val
            p_out = ctypes.c_uint64(bufs[k + 1])
            p_in  = ctypes.c_uint64(bufs[k])
            p_s   = ctypes.c_float(scale_val)
            p_N   = ctypes.c_int(N)
            params_list.append([p_out, p_in, p_s, p_N])
        else:
            # add: bufs[k+1] = bufs[k] + bufs[k-1]
            p_out  = ctypes.c_uint64(bufs[k + 1])
            p_x    = ctypes.c_uint64(bufs[k])
            p_y    = ctypes.c_uint64(bufs[k - 1] if k > 0 else bufs[0])
            p_N    = ctypes.c_int(N)
            params_list.append([p_out, p_x, p_y, p_N])

    funcs = [(func_scale if i % 2 == 0 else func_add) for i in range(N_KERNELS)]

    stream = hip.stream_create()
    event_start = hip.event_create()
    event_stop  = hip.event_create()

    try:
        # === Warmup direct launch ===
        for _ in range(BENCH_WARMUP):
            for k in range(N_KERNELS):
                hip.launch_kernel(funcs[k], grid, block, params_list[k], stream=stream)
        hip.stream_synchronize(stream)

        # === Benchmark direct launch ===
        hip.event_record(event_start, stream)
        for _ in range(BENCH_ITERS):
            for k in range(N_KERNELS):
                hip.launch_kernel(funcs[k], grid, block, params_list[k], stream=stream)
        hip.event_record(event_stop, stream)
        hip.stream_synchronize(stream)
        t_direct_ms = hip.event_elapsed_time(event_start, event_stop)
        t_direct_us = t_direct_ms * 1000.0 / BENCH_ITERS

        print(f"  Direct launch ({N_KERNELS} kernels): {t_direct_us:.2f} us/iter")

        # === Capture graph ===
        seg = GraphSegment(graph_rt, DEVICE_ID)
        seg.begin_capture(stream, mode=hipStreamCaptureModeRelaxed)
        for k in range(N_KERNELS):
            hip.launch_kernel(funcs[k], grid, block, params_list[k], stream=stream)
        seg.end_capture(stream)
        seg.instantiate()
        n_nodes = seg.num_kernel_nodes()
        print(f"  Graph captured: {n_nodes} kernel nodes")

        # === Warmup graph replay ===
        for _ in range(BENCH_WARMUP):
            seg.replay(stream)
        hip.stream_synchronize(stream)

        # === Benchmark graph replay ===
        hip.event_record(event_start, stream)
        for _ in range(BENCH_ITERS):
            seg.replay(stream)
        hip.event_record(event_stop, stream)
        hip.stream_synchronize(stream)
        t_graph_ms = hip.event_elapsed_time(event_start, event_stop)
        t_graph_us = t_graph_ms * 1000.0 / BENCH_ITERS

        print(f"  Graph replay   ({N_KERNELS} kernels): {t_graph_us:.2f} us/iter")

        speedup = t_direct_us / t_graph_us if t_graph_us > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x  (graph vs direct)")

        # Verify graph replay output matches direct
        # First collect reference via direct launch
        hip.memcpy_h2d(bufs[0], host_data.tobytes(), N * 2)
        for k in range(N_KERNELS):
            hip.launch_kernel(funcs[k], grid, block, params_list[k], stream=stream)
        hip.stream_synchronize(stream)
        buf_ref = ctypes.create_string_buffer(N * 2)
        hip.memcpy_d2h(buf_ref, bufs[N_KERNELS], N * 2)
        ref_out = np.frombuffer(buf_ref.raw, dtype=np.float16).copy()

        # Reset input and replay graph
        hip.memcpy_h2d(bufs[0], host_data.tobytes(), N * 2)
        seg.replay(stream)
        hip.stream_synchronize(stream)
        buf_graph = ctypes.create_string_buffer(N * 2)
        hip.memcpy_d2h(buf_graph, bufs[N_KERNELS], N * 2)
        graph_out = np.frombuffer(buf_graph.raw, dtype=np.float16).copy()

        err = float(np.max(np.abs(ref_out.astype(np.float32) -
                                   graph_out.astype(np.float32))))
        print(f"  Output correctness max_abs_err: {err:.2e}")

        print()
        print(f"  Performance summary:")
        print(f"    Direct launch: {t_direct_us:.2f} us/iter ({N_KERNELS} kernels)")
        print(f"    Graph replay:  {t_graph_us:.2f} us/iter ({N_KERNELS} kernels)")
        print(f"    Speedup:       {speedup:.2f}x")
        print(f"    Correctness:   max_err={err:.2e}")

        # Graph replay should be faster than direct for 8 kernel launches
        # (even if by just 10% — the main benefit is reducing host overhead)
        passed = (err < 5e-3 and n_nodes == N_KERNELS)
        speedup_note = ""
        if speedup >= 1.1:
            speedup_note = f"  graph is {speedup:.2f}x faster (overhead reduction demonstrated)"
        elif speedup >= 0.9:
            speedup_note = f"  graph at parity ({speedup:.2f}x) — GPU time dominates over launch overhead"
        else:
            speedup_note = f"  WARNING: direct launch faster ({speedup:.2f}x) — unexpected"

        print(f"  {speedup_note}")

        record("benchmark", passed,
               f"correctness OK, speedup={speedup:.2f}x" if passed
               else f"FAILED: err={err:.2e}, nodes={n_nodes}")

        seg.cleanup()

    finally:
        hip.event_destroy(event_start)
        hip.event_destroy(event_stop)
        hip.stream_destroy(stream)
        for buf in bufs:
            hip.free(buf)


# ============================================================
# TEST 7: Graph capture on null stream (default stream)
# ============================================================

def test_null_stream_capture(hip: HIPRuntime, graph_rt: HIPGraphRuntime,
                               func_scale: int):
    """Test capturing on the default (null) stream with hipStreamCaptureModeGlobal.

    For the actual decode path, we need to capture on the default stream
    (stream=0) since all existing kernels use it. This tests if that's possible.
    """
    print_header("TEST 7: Null Stream (Default Stream) Capture")

    N = 512
    SCALE = 3.0

    d_in  = hip.malloc(N * 2)
    d_out = hip.malloc(N * 2)

    host_in = np.arange(N, dtype=np.float16) * 0.01
    hip.memcpy_h2d(d_in, host_in.tobytes(), N * 2)
    hip.memset(d_out, 0, N * 2)

    grid  = ((N + 255) // 256, 1, 1)
    block = (256, 1, 1)

    p_out   = ctypes.c_uint64(d_out)
    p_in    = ctypes.c_uint64(d_in)
    p_scale = ctypes.c_float(SCALE)
    p_N     = ctypes.c_int(N)
    params  = [p_out, p_in, p_scale, p_N]

    # Try capturing on null stream (stream=0) with global mode
    # This may fail if other operations are in flight on other streams
    success = False
    err_msg = ""

    try:
        # Create a non-default stream for capture (recommended approach)
        capture_stream = hip.stream_create()

        seg = GraphSegment(graph_rt, DEVICE_ID)
        seg.begin_capture(capture_stream, mode=hipStreamCaptureModeRelaxed)
        hip.launch_kernel(func_scale, grid, block, params, stream=capture_stream)
        seg.end_capture(capture_stream)
        seg.instantiate()

        # Now replay on the null (default) stream
        hip.memset(d_out, 0, N * 2)
        seg.replay(stream=0)  # replay on null stream
        hip.synchronize()

        buf = ctypes.create_string_buffer(N * 2)
        hip.memcpy_d2h(buf, d_out, N * 2)
        result = np.frombuffer(buf.raw, dtype=np.float16).copy()
        expected = (host_in.astype(np.float32) * SCALE).astype(np.float16)
        err = float(np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32))))
        print(f"  Capture on non-default stream, replay on null stream: OK")
        print(f"  max_abs_err: {err:.2e}")

        success = err < 1e-2
        hip.stream_destroy(capture_stream)
        seg.cleanup()

    except Exception as e:
        err_msg = str(e)
        print(f"  Capture/replay failed: {e}")
        success = False

    record("null_stream_capture", success,
           "Capture on non-default + replay on null stream works" if success
           else f"FAILED: {err_msg}")

    hip.free(d_in)
    hip.free(d_out)


# ============================================================
# TEST 8: Multiple graph replays with different mutable params (simulate decode steps)
# ============================================================

def test_simulate_decode_steps(hip: HIPRuntime, graph_rt: HIPGraphRuntime,
                                func_scale: int, func_add: int, func_axpy: int):
    """Simulate a decode-like sequence with mutable params updated each step.

    Models the pattern used in graph decode path:
      - Capture once: [RMSNorm approx] → [scale with cos/sin-like factor] → [accumulate]
      - Per step: update the mutable scale factor (simulating RoPE cos/sin offset)
      - Verify each step produces correct output

    This is the closest simulation to what the real decode graph path needs.
    """
    print_header("TEST 8: Simulated Decode Steps with Mutable Param Updates")

    N = 5120
    STEPS = 12  # More than 10 to ensure stability

    d_hidden  = hip.malloc(N * 2)  # input: current hidden state
    d_rope    = hip.malloc(N * 2)  # "RoPE-scaled" hidden
    d_out     = hip.malloc(N * 2)  # output

    # Initialize hidden with some values
    host_hidden = np.random.randn(N).astype(np.float16) * 0.1

    grid  = ((N + 255) // 256, 1, 1)
    block = (256, 1, 1)

    # Simulate decode: each step applies a different "rope factor" then adds
    # Params for step 1:
    #   rope: d_rope = d_hidden * rope_cos_val   (simulates RoPE with cos factor)
    #   out:  d_out  = d_rope + d_hidden          (residual add)

    p_hidden = ctypes.c_uint64(d_hidden)
    p_rope   = ctypes.c_uint64(d_rope)
    p_out    = ctypes.c_uint64(d_out)
    p_N      = ctypes.c_int(N)
    p_cos    = ctypes.c_float(1.0)  # mutable: cos factor (changes per step)
    p_one    = ctypes.c_float(1.0)  # fixed: 1.0 for residual

    params_rope = [p_rope, p_hidden, p_cos, p_N]  # scale: rope = hidden * cos
    params_add  = [p_out, p_rope, p_hidden, p_N]  # add: out = rope + hidden

    stream = hip.stream_create()

    try:
        # Capture the graph
        seg = GraphSegment(graph_rt, DEVICE_ID)
        seg.begin_capture(stream, mode=hipStreamCaptureModeRelaxed)
        hip.launch_kernel(func_scale, grid, block, params_rope, stream=stream)
        hip.launch_kernel(func_add,   grid, block, params_add,  stream=stream)
        seg.end_capture(stream)
        seg.instantiate()

        n_nodes = seg.num_kernel_nodes()
        print(f"  Decode-like graph: {n_nodes} kernel nodes")

        # Find the scale node (it's the first kernel node, for the rope operation)
        scale_nodes = [n for n in seg._kernel_nodes]  # should be 2 nodes

        if n_nodes != 2:
            record("simulate_decode", False, f"Expected 2 nodes, got {n_nodes}")
            return

        rope_node = scale_nodes[0]  # scale kernel
        cur_params = seg.get_kernel_params(rope_node)

        all_pass = True
        for step in range(STEPS):
            # Simulate different RoPE factor per token position
            cos_factor = math.cos(step * 0.1)  # varies per position like real RoPE

            # Update hidden with "current embedding" (new token each step)
            hip.memcpy_h2d(d_hidden,
                           (host_hidden + step * 0.01).astype(np.float16).tobytes(),
                           N * 2)

            # Update mutable cos factor in the graph
            p_cos_new = ctypes.c_float(cos_factor)
            params_rope_new = [p_rope, p_hidden, p_cos_new, p_N]
            n = len(params_rope_new)
            arr = (ctypes.c_void_p * n)()
            for i, p in enumerate(params_rope_new):
                arr[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)

            new_kp = hipKernelNodeParams()
            new_kp.blockDimX = block[0]; new_kp.blockDimY = block[1]; new_kp.blockDimZ = block[2]
            new_kp.gridDimX  = grid[0];  new_kp.gridDimY  = grid[1];  new_kp.gridDimZ  = grid[2]
            new_kp.func = cur_params.func
            new_kp.kernelParams = ctypes.cast(arr, ctypes.c_void_p).value
            new_kp.sharedMemBytes = 0
            new_kp.extra = None

            seg.update_kernel_params(rope_node, new_kp)

            hip.memset(d_out, 0, N * 2)
            seg.replay(stream)
            hip.stream_synchronize(stream)

            buf = ctypes.create_string_buffer(N * 2)
            hip.memcpy_d2h(buf, d_out, N * 2)
            result = np.frombuffer(buf.raw, dtype=np.float16).copy()

            hidden_step = (host_hidden + step * 0.01).astype(np.float32)
            expected_rope = hidden_step * cos_factor
            expected = (expected_rope + hidden_step).astype(np.float16)
            err = float(np.max(np.abs(result.astype(np.float32) -
                                       expected.astype(np.float32))))

            step_pass = err < 5e-3
            if not step_pass:
                all_pass = False
            print(f"  Step {step+1:2d}: cos={cos_factor:.4f}, "
                  f"result[0]={float(result[0]):.4f}, "
                  f"expected={float(expected[0]):.4f}, "
                  f"err={err:.2e}  {'OK' if step_pass else 'FAIL'}")

            # Keep objects alive during replay
            _ = (p_cos_new, params_rope_new, arr, new_kp)

        record("simulate_decode", all_pass,
               f"All {STEPS} decode-like steps correct" if all_pass
               else "Some steps failed")

        seg.cleanup()

    finally:
        hip.stream_destroy(stream)
        hip.free(d_hidden)
        hip.free(d_rope)
        hip.free(d_out)


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("=" * 72)
    print("  HIP Graph Infrastructure Tests — gfx906 (MI50)")
    print("=" * 72)
    print(f"  Device: {DEVICE_ID}")
    print(f"  Bench warmup: {BENCH_WARMUP}, iters: {BENCH_ITERS}")

    # Initialize HIP
    hip = HIPRuntime()
    hip.init()
    hip.set_device(DEVICE_ID)

    count = ctypes.c_int(0)
    hip._lib.hipGetDeviceCount(ctypes.byref(count))
    print(f"  Detected {count.value} GPU(s)")

    # Initialize Graph runtime
    graph_rt = HIPGraphRuntime()
    print(f"  HIPGraphRuntime initialized")

    # Build test kernels
    print()
    print("  Building test kernels (hipcc)...")
    try:
        module, func_scale, func_add, func_axpy, so_path = build_test_kernel(hip)
        print(f"  Kernels built: {so_path}")
    except Exception as e:
        print(f"  FATAL: Failed to build test kernels: {e}")
        sys.exit(1)

    # Run tests
    test_graph_api_availability(graph_rt)
    test_single_kernel_graph(hip, graph_rt, func_scale)
    test_multi_kernel_graph(hip, graph_rt, func_scale, func_add)
    test_mutable_params(hip, graph_rt, func_axpy)
    test_multi_step_mutable(hip, graph_rt, func_scale)
    test_benchmark_graph_vs_direct(hip, graph_rt, func_scale, func_add)
    test_null_stream_capture(hip, graph_rt, func_scale)
    test_simulate_decode_steps(hip, graph_rt, func_scale, func_add, func_axpy)

    # Summary
    print()
    print("=" * 72)
    print("  TEST SUMMARY")
    print("=" * 72)

    n_pass = 0
    n_fail = 0
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:35s}: {status}")
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    print()
    print(f"  Total: {n_pass} PASS, {n_fail} FAIL")

    # Validation assertion summary
    print()
    print("=" * 72)
    print("  VALIDATION ASSERTIONS")
    print("=" * 72)

    api_ok = results.get("api_availability", False)
    capture_ok = (results.get("single_kernel_graph", False) and
                  results.get("multi_kernel_graph", False))
    mutable_ok = (results.get("mutable_params", False) and
                  results.get("multi_step_mutable", False) and
                  results.get("simulate_decode", False))
    bench_ok   = results.get("benchmark", False)

    print(f"  VAL-GRAPH-001 (HIP graph capture succeeds on gfx906): "
          f"{'PASS' if api_ok and capture_ok else 'FAIL'}")
    print(f"  VAL-GRAPH-002 (Mutable params update correctly across steps): "
          f"{'PASS' if mutable_ok else 'FAIL'}")
    print(f"  VAL-GRAPH-003 (Graph replay produces correct results): "
          f"{'PASS' if capture_ok and mutable_ok else 'FAIL'}")

    # Critical: HIP graph basics must work for the decode path feature
    critical_pass = (api_ok and capture_ok and mutable_ok)

    print()
    if critical_pass:
        print("  OVERALL: ALL CRITICAL CHECKS PASSED")
        print()
        print("  HIP graph infrastructure is functional on gfx906/ROCm 7.1.")
        print("  Ready for graph-decode-path feature implementation.")
    else:
        print("  OVERALL: CRITICAL CHECKS FAILED")
        print()
        if not api_ok:
            print("  ERROR: HIP Graph APIs not available — cannot proceed with graph decode")
        if not capture_ok:
            print("  ERROR: Graph capture failed — check ROCm version and GPU support")
        if not mutable_ok:
            print("  ERROR: Mutable param updates failed — graph decode won't work correctly")
        sys.exit(1)

    # Cleanup
    hip.module_unload(module)
    hip.synchronize()
    print()


if __name__ == "__main__":
    main()
