"""
hip_graph_dispatch.py — HIP Graph capture and replay infrastructure for decode.

Provides:
- HIPGraphRuntime: ctypes bindings for HIP Graph API (hipGraphCreate, etc.)
- GraphSegment: captures a sequence of kernel launches on one GPU between allreduce points
- GraphDecodeStep: manages the full decode step as alternating graph-replay and
  host-allreduce phases

Usage pattern:
    1. Create a GraphDecodeStep with the TPInferenceEngine
    2. On first call to decode_step_graph(), it captures per-GPU compute graphs
    3. On subsequent calls, it replays graphs with updated mutable params
       (RoPE cos/sin offsets, attention seq_len)

IMPORTANT: HIP graphs are per-device. Each GPU's compute segment must be captured
as a separate graph. Allreduce (cross-GPU P2P) stays host-orchestrated between
graph replay segments, as cross-device operations are not capturable in a single graph.

Key APIs used:
  hipStreamBeginCapture / hipStreamEndCapture: capture mode
  hipGraphInstantiate: compile graph → executable
  hipGraphLaunch: replay graph
  hipGraphExecKernelNodeSetParams: update mutable kernel args between replays
  hipGraphExecDestroy / hipGraphDestroy: cleanup
"""

import ctypes
import os
from typing import Optional, List, Dict, Any

# ============================================================
# HIP Graph API constants
# ============================================================

# hipStreamCaptureMode
hipStreamCaptureModeGlobal      = 0
hipStreamCaptureModeThreadLocal = 1
hipStreamCaptureModeRelaxed     = 2

# hipGraphNodeType
hipGraphNodeTypeKernel      = 0
hipGraphNodeTypeMemcpy      = 1
hipGraphNodeTypeMemset      = 2
hipGraphNodeTypeHost        = 3
hipGraphNodeTypeGraph       = 4
hipGraphNodeTypeEmpty       = 5
hipGraphNodeTypeWaitEvent   = 6
hipGraphNodeTypeEventRecord = 7

# hipStreamCaptureStatus
hipStreamCaptureStatusNone   = 0
hipStreamCaptureStatusActive = 1


# ============================================================
# hipKernelNodeParams ctypes struct
# ============================================================

class hipKernelNodeParams(ctypes.Structure):
    """hipKernelNodeParams for hipGraphAddKernelNode / hipGraphExecKernelNodeSetParams."""
    _fields_ = [
        ("blockDimX",  ctypes.c_uint),
        ("blockDimY",  ctypes.c_uint),
        ("blockDimZ",  ctypes.c_uint),
        ("extra",      ctypes.c_void_p),
        ("func",       ctypes.c_void_p),
        ("gridDimX",   ctypes.c_uint),
        ("gridDimY",   ctypes.c_uint),
        ("gridDimZ",   ctypes.c_uint),
        ("kernelParams", ctypes.c_void_p),
        ("sharedMemBytes", ctypes.c_uint),
    ]


# ============================================================
# HIP Graph Runtime bindings
# ============================================================

class HIPGraphRuntime:
    """ctypes bindings for HIP Graph API.

    Wraps the HIP graph functions needed for capture and replay:
      - hipStreamBeginCapture / hipStreamEndCapture
      - hipGraphInstantiate / hipGraphLaunch
      - hipGraphGetNodes / hipGraphNodeGetType / hipGraphKernelNodeGetParams
      - hipGraphExecKernelNodeSetParams
      - hipGraphExecDestroy / hipGraphDestroy
    """

    def __init__(self, hip_lib_path: str = "/opt/rocm/lib/libamdhip64.so"):
        self._lib = ctypes.CDLL(hip_lib_path)
        self._setup_functions()

    def _setup_functions(self):
        lib = self._lib

        # hipStreamBeginCapture(stream, mode)
        lib.hipStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.hipStreamBeginCapture.restype = ctypes.c_int

        # hipStreamEndCapture(stream, graph*)
        lib.hipStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.hipStreamEndCapture.restype = ctypes.c_int

        # hipGraphCreate(graph*, flags)
        lib.hipGraphCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.hipGraphCreate.restype = ctypes.c_int

        # hipGraphDestroy(graph)
        lib.hipGraphDestroy.argtypes = [ctypes.c_void_p]
        lib.hipGraphDestroy.restype = ctypes.c_int

        # hipGraphInstantiate(graphExec*, graph, errorNode*, logBuffer, bufferSize)
        lib.hipGraphInstantiate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # pGraphExec
            ctypes.c_void_p,                   # graph
            ctypes.POINTER(ctypes.c_void_p),   # pErrorNode (can be NULL)
            ctypes.c_char_p,                   # pLogBuffer (can be NULL)
            ctypes.c_size_t,                   # bufferSize
        ]
        lib.hipGraphInstantiate.restype = ctypes.c_int

        # hipGraphLaunch(graphExec, stream)
        lib.hipGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.hipGraphLaunch.restype = ctypes.c_int

        # hipGraphExecDestroy(graphExec)
        lib.hipGraphExecDestroy.argtypes = [ctypes.c_void_p]
        lib.hipGraphExecDestroy.restype = ctypes.c_int

        # hipGraphGetNodes(graph, nodes*, numNodes*)
        lib.hipGraphGetNodes.argtypes = [
            ctypes.c_void_p,                      # graph
            ctypes.POINTER(ctypes.c_void_p),      # nodes (out, can be NULL for count)
            ctypes.POINTER(ctypes.c_size_t),      # numNodes (in/out)
        ]
        lib.hipGraphGetNodes.restype = ctypes.c_int

        # hipGraphNodeGetType(node, type*)
        lib.hipGraphNodeGetType.argtypes = [
            ctypes.c_void_p,                # node
            ctypes.POINTER(ctypes.c_int),   # type (out)
        ]
        lib.hipGraphNodeGetType.restype = ctypes.c_int

        # hipGraphKernelNodeGetParams(node, params*)
        lib.hipGraphKernelNodeGetParams.argtypes = [
            ctypes.c_void_p,                            # node
            ctypes.POINTER(hipKernelNodeParams),        # params (out)
        ]
        lib.hipGraphKernelNodeGetParams.restype = ctypes.c_int

        # hipGraphExecKernelNodeSetParams(graphExec, node, params*)
        lib.hipGraphExecKernelNodeSetParams.argtypes = [
            ctypes.c_void_p,                            # graphExec
            ctypes.c_void_p,                            # node
            ctypes.POINTER(hipKernelNodeParams),        # params (in)
        ]
        lib.hipGraphExecKernelNodeSetParams.restype = ctypes.c_int

        # hipGraphKernelNodeSetParams(node, params*)
        lib.hipGraphKernelNodeSetParams.argtypes = [
            ctypes.c_void_p,                            # node
            ctypes.POINTER(hipKernelNodeParams),        # params (in)
        ]
        lib.hipGraphKernelNodeSetParams.restype = ctypes.c_int

        # hipStreamGetCaptureInfo(stream, status*, id*) — check if stream is in capture mode
        try:
            lib.hipStreamGetCaptureInfo.argtypes = [
                ctypes.c_void_p,               # stream
                ctypes.POINTER(ctypes.c_int),  # captureStatus (out)
                ctypes.POINTER(ctypes.c_uint64),  # id (out, can be NULL)
            ]
            lib.hipStreamGetCaptureInfo.restype = ctypes.c_int
            self._has_get_capture_info = True
        except AttributeError:
            self._has_get_capture_info = False

    def _check(self, err: int, msg: str = ""):
        if err != 0:
            raise RuntimeError(f"HIP error {err}: {msg}")

    def stream_begin_capture(self, stream: int,
                              mode: int = hipStreamCaptureModeRelaxed):
        """Begin capturing operations on a stream into a graph."""
        self._check(
            self._lib.hipStreamBeginCapture(
                ctypes.c_void_p(stream), ctypes.c_int(mode)
            ),
            "hipStreamBeginCapture"
        )

    def stream_end_capture(self, stream: int) -> int:
        """End stream capture and return the captured graph handle."""
        graph = ctypes.c_void_p(0)
        self._check(
            self._lib.hipStreamEndCapture(
                ctypes.c_void_p(stream), ctypes.byref(graph)
            ),
            "hipStreamEndCapture"
        )
        return graph.value

    def graph_create(self) -> int:
        """Create an empty graph."""
        graph = ctypes.c_void_p(0)
        self._check(
            self._lib.hipGraphCreate(ctypes.byref(graph), ctypes.c_uint(0)),
            "hipGraphCreate"
        )
        return graph.value

    def graph_destroy(self, graph: int):
        """Destroy a graph (template, not executable)."""
        self._check(
            self._lib.hipGraphDestroy(ctypes.c_void_p(graph)),
            "hipGraphDestroy"
        )

    def graph_instantiate(self, graph: int) -> int:
        """Instantiate (compile) a graph into an executable graph."""
        graph_exec = ctypes.c_void_p(0)
        self._check(
            self._lib.hipGraphInstantiate(
                ctypes.byref(graph_exec),
                ctypes.c_void_p(graph),
                None,   # errorNode
                None,   # logBuffer
                0,      # bufferSize
            ),
            "hipGraphInstantiate"
        )
        return graph_exec.value

    def graph_launch(self, graph_exec: int, stream: int = 0):
        """Launch (replay) an instantiated graph on a stream."""
        self._check(
            self._lib.hipGraphLaunch(
                ctypes.c_void_p(graph_exec), ctypes.c_void_p(stream)
            ),
            "hipGraphLaunch"
        )

    def graph_exec_destroy(self, graph_exec: int):
        """Destroy an executable graph."""
        self._check(
            self._lib.hipGraphExecDestroy(ctypes.c_void_p(graph_exec)),
            "hipGraphExecDestroy"
        )

    def graph_get_nodes(self, graph: int) -> list:
        """Get all nodes in a graph. Returns list of node handles."""
        # First get count
        num = ctypes.c_size_t(0)
        self._check(
            self._lib.hipGraphGetNodes(
                ctypes.c_void_p(graph), None, ctypes.byref(num)
            ),
            "hipGraphGetNodes (count)"
        )
        count = num.value
        if count == 0:
            return []

        # Then get actual nodes
        nodes_arr = (ctypes.c_void_p * count)()
        self._check(
            self._lib.hipGraphGetNodes(
                ctypes.c_void_p(graph),
                ctypes.cast(nodes_arr, ctypes.POINTER(ctypes.c_void_p)),
                ctypes.byref(num)
            ),
            "hipGraphGetNodes (fill)"
        )
        return [nodes_arr[i] for i in range(count)]

    def node_get_type(self, node: int) -> int:
        """Get the type of a graph node."""
        node_type = ctypes.c_int(0)
        self._check(
            self._lib.hipGraphNodeGetType(
                ctypes.c_void_p(node), ctypes.byref(node_type)
            ),
            "hipGraphNodeGetType"
        )
        return node_type.value

    def kernel_node_get_params(self, node: int) -> hipKernelNodeParams:
        """Get parameters of a kernel node."""
        params = hipKernelNodeParams()
        self._check(
            self._lib.hipGraphKernelNodeGetParams(
                ctypes.c_void_p(node), ctypes.byref(params)
            ),
            "hipGraphKernelNodeGetParams"
        )
        return params

    def exec_kernel_node_set_params(self, graph_exec: int, node: int,
                                     params: hipKernelNodeParams):
        """Update kernel node parameters in an executable graph between replays."""
        self._check(
            self._lib.hipGraphExecKernelNodeSetParams(
                ctypes.c_void_p(graph_exec),
                ctypes.c_void_p(node),
                ctypes.byref(params)
            ),
            "hipGraphExecKernelNodeSetParams"
        )


# ============================================================
# GraphSegment: captures one GPU's compute between allreduce points
# ============================================================

class MutableParamRef:
    """Reference to a mutable kernel parameter within a captured graph.

    Holds:
      - node: graph node handle containing the kernel
      - param_index: index into the kernel's params array
      - params: the full hipKernelNodeParams for that node (for update)
      - params_array: the C array of void* pointers to ctypes values
    """
    def __init__(self, node: int, params: hipKernelNodeParams):
        self.node = node
        self.params = params


class GraphSegment:
    """A captured HIP graph segment for one GPU between allreduce points.

    Workflow:
      1. segment.begin_capture(stream) — put the stream in capture mode
      2. Launch kernels on the stream normally (they get captured, not executed)
      3. segment.end_capture(stream) — finalize the graph
      4. segment.instantiate() — compile the graph to an executable
      5. segment.find_mutable_nodes(kernel_func_handles) — identify nodes to update
      6. Loop:
         a. segment.update_param(node, param_idx, new_val) — update mutable args
         b. segment.replay(stream) — re-execute the captured graph

    The captured graph replaces the original kernel launches with near-zero
    host overhead (single hipGraphLaunch call vs ~N hipModuleLaunchKernel calls).
    """

    def __init__(self, graph_rt: HIPGraphRuntime, device_id: int):
        self._graph_rt = graph_rt
        self._device_id = device_id
        self._graph = None       # template graph handle
        self._graph_exec = None  # instantiated (executable) graph handle
        self._nodes = []         # all nodes in the graph
        self._kernel_nodes = []  # kernel nodes only (list of node handles)
        self._kernel_node_params = {}  # node_handle → hipKernelNodeParams

    def begin_capture(self, stream: int,
                      mode: int = hipStreamCaptureModeRelaxed):
        """Begin capturing operations submitted to stream into this segment."""
        self._graph_rt.stream_begin_capture(stream, mode)

    def end_capture(self, stream: int):
        """End capture and store the graph template."""
        self._graph = self._graph_rt.stream_end_capture(stream)

    def instantiate(self):
        """Compile the captured graph into an executable graph."""
        if self._graph is None:
            raise RuntimeError("GraphSegment: must call end_capture() before instantiate()")
        self._graph_exec = self._graph_rt.graph_instantiate(self._graph)

        # Enumerate and classify nodes
        self._nodes = self._graph_rt.graph_get_nodes(self._graph)
        self._kernel_nodes = []
        self._kernel_node_params = {}
        for node in self._nodes:
            ntype = self._graph_rt.node_get_type(node)
            if ntype == hipGraphNodeTypeKernel:
                self._kernel_nodes.append(node)
                params = self._graph_rt.kernel_node_get_params(node)
                self._kernel_node_params[node] = params

    def num_kernel_nodes(self) -> int:
        return len(self._kernel_nodes)

    def find_kernel_nodes_by_func(self, func_handles: list) -> list:
        """Return kernel nodes whose func matches any of the given handles.

        Used to identify RoPE/QKNorm nodes and decode-attention nodes that
        have mutable parameters (cos/sin pointers, seq_len).
        """
        func_set = set(func_handles)
        matching = []
        for node in self._kernel_nodes:
            p = self._kernel_node_params[node]
            if p.func in func_set:
                matching.append(node)
        return matching

    def update_kernel_params(self, node: int, new_params: hipKernelNodeParams):
        """Update a kernel node's parameters in the executable graph.

        This allows changing mutable args (RoPE offsets, seq_len) between
        graph replays without re-capturing. Uses hipGraphExecKernelNodeSetParams.
        """
        if self._graph_exec is None:
            raise RuntimeError("GraphSegment: must call instantiate() before update_kernel_params()")
        self._graph_rt.exec_kernel_node_set_params(
            self._graph_exec, node, new_params
        )
        # Also update our cached params
        self._kernel_node_params[node] = new_params

    def get_kernel_params(self, node: int) -> hipKernelNodeParams:
        """Get the current parameters for a kernel node."""
        return self._kernel_node_params[node]

    def replay(self, stream: int = 0):
        """Replay the captured graph on the given stream."""
        if self._graph_exec is None:
            raise RuntimeError("GraphSegment: must call instantiate() before replay()")
        self._graph_rt.graph_launch(self._graph_exec, stream)

    def cleanup(self):
        """Destroy graph resources."""
        if self._graph_exec is not None:
            self._graph_rt.graph_exec_destroy(self._graph_exec)
            self._graph_exec = None
        if self._graph is not None:
            self._graph_rt.graph_destroy(self._graph)
            self._graph = None
        self._nodes.clear()
        self._kernel_nodes.clear()
        self._kernel_node_params.clear()


# ============================================================
# GraphDecodeStep: orchestrates full decode with graph replay
# ============================================================

class GraphDecodeStep:
    """Manages per-GPU decode graphs for the full 64-layer decode step.

    Structure per layer:
      Segment A (attention compute):
        RMSNorm → Q/KV GEMV → QKNorm/RoPE → KV append → decode attention
        → sigmoid gate → O-proj
      [Host allreduce (P2P)]
      Segment B (FFN compute):
        FFN RMSNorm → gate+up+silu → down_proj
      [Host allreduce (P2P)]

    Total: num_layers × 2 segments × tp_size GPUs graph segments

    On first decode call: captures all graph segments
    On subsequent calls: replays graphs with updated mutable params
    """

    def __init__(self, tp_engine, graph_rt: Optional[HIPGraphRuntime] = None):
        """
        Args:
            tp_engine: TPInferenceEngine instance
            graph_rt: HIPGraphRuntime instance (created if None)
        """
        self._tp_engine = tp_engine
        self._graph_rt = graph_rt or HIPGraphRuntime()
        self._captured = False

        # Per-GPU, per-layer, per-segment graph objects
        # Indexed as: _attn_graphs[gpu_idx][layer_idx]
        # and:        _ffn_graphs[gpu_idx][layer_idx]
        self._attn_graphs: List[List[GraphSegment]] = []
        self._ffn_graphs: List[List[GraphSegment]] = []

        # Mutable node tracking for RoPE and decode-attention
        # _mutable_attn[gpu_idx][layer_idx] = list of (node, base_params) for mutable nodes
        self._mutable_attn: List[List[List]] = []

    @property
    def captured(self) -> bool:
        return self._captured

    def cleanup(self):
        """Destroy all captured graphs."""
        for gpu_attn_graphs in self._attn_graphs:
            for seg in gpu_attn_graphs:
                seg.cleanup()
        for gpu_ffn_graphs in self._ffn_graphs:
            for seg in gpu_ffn_graphs:
                seg.cleanup()
        self._attn_graphs.clear()
        self._ffn_graphs.clear()
        self._mutable_attn.clear()
        self._captured = False
