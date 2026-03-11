"""
Minimal graph IR for transformer inference.

Represents a DAG of operations with typed tensor edges.
Designed for:
- Static shape inference (all shapes known at graph construction time)
- Fusion pass (merge compatible ops)
- Memory planning (compute buffer lifetimes, reuse allocations)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class DType(Enum):
    FP32 = auto()
    FP16 = auto()
    INT8 = auto()
    INT4 = auto()  # packed, 2 values per byte

    @property
    def itemsize(self) -> int:
        """Bytes per element (INT4 returns 1 for 2 elements)."""
        return {
            DType.FP32: 4,
            DType.FP16: 2,
            DType.INT8: 1,
            DType.INT4: 1,  # 2 elements packed
        }[self]


@dataclass
class TensorInfo:
    """Static tensor metadata."""
    shape: tuple
    dtype: DType
    name: str = ""

    @property
    def numel(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def nbytes(self) -> int:
        if self.dtype == DType.INT4:
            return (self.numel + 1) // 2
        return self.numel * self.dtype.itemsize


class OpType(Enum):
    # GEMM / linear
    GEMM = auto()           # C = A @ B
    GEMM_INT4 = auto()      # INT4 weight-only quantized GEMM
    GEMM_INT8 = auto()      # INT8 W8A8 GEMM
    GEMV = auto()            # Matrix-vector (decode path)

    # Attention
    FLASH_ATTN = auto()      # FlashAttention forward
    DECODE_ATTN = auto()     # Single-query decode attention

    # Elementwise / normalization
    RMSNORM = auto()
    SILU = auto()
    ROPE = auto()
    RESIDUAL_ADD = auto()
    MULTIPLY = auto()        # element-wise multiply (for gate)

    # Data movement
    CONCAT = auto()
    SPLIT = auto()
    TRANSPOSE = auto()
    COPY = auto()

    # Multi-GPU
    ALL_REDUCE = auto()

    # Sampling
    SOFTMAX = auto()
    TOP_K = auto()
    TOP_P = auto()


@dataclass
class Op:
    """A single operation in the graph."""
    op_type: OpType
    inputs: list  # list of TensorInfo or Op references
    outputs: list  # list of TensorInfo
    attrs: dict = field(default_factory=dict)  # op-specific attributes
    name: str = ""
    fused_with: Optional['Op'] = None  # if this op is fused into another


@dataclass
class Graph:
    """A DAG of operations representing a transformer layer or full model."""
    ops: list = field(default_factory=list)
    inputs: list = field(default_factory=list)   # graph-level inputs
    outputs: list = field(default_factory=list)  # graph-level outputs

    def add_op(self, op: Op) -> Op:
        self.ops.append(op)
        return op

    def toposort(self) -> list:
        """Topological sort of ops (placeholder for now)."""
        return list(self.ops)
