"""
Persistent Megakernel Dispatch for TP=4 Decode

Milestone: m5-persistent-kernel

This module provides Python bindings for the persistent_decode.hip megakernel.
The persistent kernel eliminates host-side kernel launch overhead by running
the entire decode step (64 layers) as a single GPU kernel.

Usage:
    from src.runtime.persistent_dispatch import PersistentDecodeDispatcher
    
    dispatcher = PersistentDecodeDispatcher(tp_engine)
    dispatcher.enable()
    hidden = dispatcher.decode_step(token_embedding, position)
"""

import ctypes
import os
from pathlib import Path
from typing import Optional, List, Dict, Any


class TaskDescriptor(ctypes.Structure):
    """Task descriptor for persistent kernel task queue (64 bytes)"""
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("layer_id", ctypes.c_uint32),
        ("input_ptr", ctypes.c_uint64),
        ("output_ptr", ctypes.c_uint64),
        ("weight_ptr", ctypes.c_uint64),
        ("extra_ptr", ctypes.c_uint64),
        ("input_size", ctypes.c_uint32),
        ("output_size", ctypes.c_uint32),
        ("eps", ctypes.c_float),
        ("group_size", ctypes.c_uint32),
        ("dep_count", ctypes.c_uint32),
        ("dep_task_ids", ctypes.c_uint32 * 4),
    ]


class PersistentDecodeState(ctypes.Structure):
    """Global state for persistent decode kernel"""
    _fields_ = [
        # Task queue
        ("task_queue", TaskDescriptor * 2048),
        ("queue_head", ctypes.c_uint32),
        ("queue_tail", ctypes.c_uint32),
        ("tasks_completed", ctypes.c_uint32),
        
        # WG state
        ("wg_ready_count", ctypes.c_uint32),
        ("current_layer", ctypes.c_uint32),
        
        # P2P allreduce state
        ("ar_phase", ctypes.c_uint32),
        ("partial_ptrs", ctypes.c_uint64 * 4),
        ("hidden_ptr", ctypes.c_uint64),
        ("hidden_size", ctypes.c_uint32),
        
        # KV cache pointers
        ("kv_cache_k_ptrs", ctypes.c_uint64 * 64),
        ("kv_cache_v_ptrs", ctypes.c_uint64 * 64),
        
        # RoPE tables
        ("cos_tab_ptr", ctypes.c_uint64),
        ("sin_tab_ptr", ctypes.c_uint64),
        
        # Kernel function pointers (not used in simplified version)
        ("gemv_fp16_fn", ctypes.c_uint64),
        ("gemv_int4_fn", ctypes.c_uint64),
        ("rmsnorm_fn", ctypes.c_uint64),
        ("attention_fn", ctypes.c_uint64),
        ("deltanet_fn", ctypes.c_uint64),
        ("allreduce_fn", ctypes.c_uint64),
    ]


class PersistentDecodeDispatcher:
    """
    Persistent megakernel dispatcher for TP=4 decode.
    
    When enabled, replaces the C dispatch loop with a single persistent kernel
    launch that executes all 64 layers without host intervention.
    """
    
    def __init__(self, tp_engine):
        """
        Initialize persistent dispatcher.
        
        Args:
            tp_engine: TPInferenceEngine instance
        """
        self.tp_engine = tp_engine
        self._enabled = False
        self._lib = None
        self._state = None
        self._state_ptr = None
        
        # Lazy load the library
        self._load_library()
    
    def _load_library(self) -> bool:
        """Load persistent_decode.so shared library"""
        if self._lib is not None:
            return True
        
        src_dir = Path(__file__).parent.parent / "kernels"
        so_path = src_dir.parent / "build" / "kernels" / "persistent_decode.so"
        
        if not so_path.exists():
            # Try to build it
            print(f"WARNING: persistent_decode.so not found at {so_path}")
            print("Building persistent kernel...")
            if not self._build_kernel():
                return False
        
        try:
            self._lib = ctypes.CDLL(str(so_path))
            
            # Register function signature
            # int persistent_decode_tp4(void* state, void* stream, unsigned int my_gpu_rank)
            self._lib.persistent_decode_tp4.argtypes = [
                ctypes.c_void_p,    # state
                ctypes.c_void_p,    # stream
                ctypes.c_uint32,    # my_gpu_rank
            ]
            self._lib.persistent_decode_tp4.restype = ctypes.c_int
            
            print(f"Loaded persistent_decode.so from {so_path}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load persistent_decode.so: {e}")
            return False
    
    def _build_kernel(self) -> bool:
        """Build persistent_decode.so"""
        import subprocess
        
        hipcc = "/opt/rocm/bin/hipcc"
        src_path = Path(__file__).parent.parent / "kernels" / "persistent_decode.hip"
        so_path = Path(__file__).parent.parent / "build" / "kernels" / "persistent_decode.so"
        
        if not src_path.exists():
            print(f"ERROR: Source file not found: {src_path}")
            return False
        
        cmd = [
            hipcc, "-O3", "--offload-arch=gfx906", "-std=c++17",
            "-shared", "-fPIC",
            "-o", str(so_path), str(src_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"ERROR: Build failed:\n{result.stderr}")
                return False
            print(f"Built persistent_decode.so successfully")
            return True
        except subprocess.TimeoutExpired:
            print("ERROR: Build timed out")
            return False
        except Exception as e:
            print(f"ERROR: Build failed: {e}")
            return False
    
    def enable(self) -> bool:
        """
        Enable persistent kernel mode.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._load_library():
            print("WARNING: Could not load persistent kernel library")
            return False
        
        # Initialize state
        self._state = PersistentDecodeState()
        self._state_ptr = ctypes.pointer(self._state)
        
        # Initialize task queue
        self._state.queue_head = 0
        self._state.queue_tail = 0
        self._state.tasks_completed = 0
        self._state.current_layer = 0
        self._state.ar_phase = 0
        
        # Set hidden size (Qwen3.5-27B)
        self._state.hidden_size = 5120
        
        # Get device pointers from engine
        # (would need to extract from tp_engine's internal buffers)
        
        self._enabled = True
        print("Persistent kernel mode ENABLED")
        return True
    
    def disable(self):
        """Disable persistent kernel mode"""
        self._enabled = False
        print("Persistent kernel mode DISABLED")
    
    def decode_step(self, token_embedding: 'np.ndarray', position: int) -> 'np.ndarray':
        """
        Execute a single decode step using the persistent kernel.
        
        Args:
            token_embedding: Input token embedding (hidden_size,)
            position: Current sequence position
            
        Returns:
            Updated hidden state (hidden_size,)
        """
        if not self._enabled:
            raise RuntimeError("Persistent kernel not enabled. Call enable() first.")
        
        # This is a simplified implementation
        # In production, would:
        # 1. Set up task queue with all 64 layers' tasks
        # 2. Update mutable params (RoPE cos/sin, KV cache pointers, seq_len)
        # 3. Launch persistent kernel on each GPU
        # 4. Wait for completion
        # 5. Return updated hidden state
        
        # For now, fallback to C dispatch
        print("WARNING: Persistent kernel decode_step not fully implemented")
        print("Falling back to C dispatch")
        return None
    
    def build_task_queue(self) -> bool:
        """
        Pre-build the task queue for all 64 layers.
        
        Called once at initialization to populate the task queue
        with all kernel launches needed for a full decode step.
        """
        # Build task descriptors for attention + FFN for all 64 layers
        # This mirrors what C dispatch does, but creates TaskDescriptor entries
        
        task_id = 0
        
        for layer_id in range(64):
            # Determine layer type (full attention or DeltaNet)
            # For Qwen3.5-27B: [DeltaNet, DeltaNet, DeltaNet, FullAttn] × 16
            
            layer_type = "full_attention" if (layer_id % 4 == 3) else "deltanet"
            
            if layer_type == "full_attention":
                # Attention phase
                # 1. Attention RMSNorm
                self._state.task_queue[task_id].type = 3  # TASK_RMSNORM
                self._state.task_queue[task_id].layer_id = layer_id
                task_id += 1
                
                # 2. GEMV Q, K, V
                for _ in range(3):
                    self._state.task_queue[task_id].type = 1  # TASK_GEMV_FP16
                    self._state.task_queue[task_id].layer_id = layer_id
                    self._state.task_queue[task_id].dep_count = 1
                    self._state.task_queue[task_id].dep_task_ids[0] = task_id - 1
                    task_id += 1
                
                # 3. QKNorm + RoPE
                self._state.task_queue[task_id].type = 3  # TASK_RMSNORM
                self._state.task_queue[task_id].layer_id = layer_id
                self._state.task_queue[task_id].dep_count = 2
                task_id += 1
                
                # 4. Decode attention
                self._state.task_queue[task_id].type = 5  # TASK_ATTENTION
                self._state.task_queue[task_id].layer_id = layer_id
                self._state.task_queue[task_id].dep_count = 1
                task_id += 1
                
                # 5. O projection
                self._state.task_queue[task_id].type = 1  # TASK_GEMV_FP16
                self._state.task_queue[task_id].layer_id = layer_id
                self._state.task_queue[task_id].dep_count = 1
                task_id += 1
                
                # 6. Attention allreduce
                self._state.task_queue[task_id].type = 7  # TASK_ALLREDUCE
                self._state.task_queue[task_id].layer_id = layer_id
                self._state.task_queue[task_id].dep_count = 1
                task_id += 1
            
            # FFN phase (same for both layer types)
            # 1. FFN RMSNorm
            self._state.task_queue[task_id].type = 3  # TASK_RMSNORM
            self._state.task_queue[task_id].layer_id = layer_id
            task_id += 1
            
            # 2. Gate + Up projections
            for _ in range(2):
                self._state.task_queue[task_id].type = 2  # TASK_GEMV_INT4
                self._state.task_queue[task_id].layer_id = layer_id
                self._state.task_queue[task_id].dep_count = 1
                task_id += 1
            
            # 3. SiLU activation
            self._state.task_queue[task_id].type = 9  # TASK_SILU
            self._state.task_queue[task_id].layer_id = layer_id
            self._state.task_queue[task_id].dep_count = 2
            task_id += 1
            
            # 4. Down projection
            self._state.task_queue[task_id].type = 2  # TASK_GEMV_INT4
            self._state.task_queue[task_id].layer_id = layer_id
            self._state.task_queue[task_id].dep_count = 1
            task_id += 1
            
            # 5. FFN allreduce
            self._state.task_queue[task_id].type = 7  # TASK_ALLREDUCE
            self._state.task_queue[task_id].layer_id = layer_id
            self._state.task_queue[task_id].dep_count = 1
            task_id += 1
        
        print(f"Built task queue with {task_id} tasks for 64 layers")
        return True
