# nn2fpga/op_base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any, Iterable, Tuple
from nn2fpga.compiler.core.tensor_quant import TensorQuant
from qonnx.custom_op.base import CustomOp
from qonnx.core.modelwrapper import ModelWrapper
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class NodeInterface:
    in_stream_array: int
    out_stream_array: int
    in_word_array: int
    out_word_array: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "in_stream_array": self.in_stream_array,
            "out_stream_array": self.out_stream_array,
            "in_word_array": self.in_word_array,
            "out_word_array": self.out_word_array,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeInterface":
        return cls(
            in_stream_array=int(d["in_stream_array"]),
            out_stream_array=int(d["out_stream_array"]),
            in_word_array=int(d["in_word_array"]),
            out_word_array=int(d["out_word_array"]),
        )

class NN2FPGAOp(CustomOp, ABC):
    """Abstract base for nn2fpga operators. """
    
    @abstractmethod
    def lower_to_hls(self, model: ModelWrapper, hls_tag: int):
        """Lower this operator to the HLSKernel implementation."""

    @abstractmethod
    def has_linebuffer(self) -> bool:
        """Return whether the op needs a linebuffer (default False)."""
    
    @abstractmethod
    def get_latency(self, model: ModelWrapper) -> int:
        """Return latency [cycles] for 'point'. If point is None, use current node attrs."""

    @abstractmethod
    def get_brams(self, model: ModelWrapper) -> int:
        """Return BRAM usage for 'point'. If point is None, use current node attrs."""

    @abstractmethod
    def get_dsps(self, model: ModelWrapper) -> int:
        """Return DSP usage for 'point'. If point is None, use current node attrs."""
    
    @abstractmethod
    def accepted_input_layout(self) -> tuple | None:
        """Return the permutation tuple this op requires on input,
        or None if layout-transparent."""

    @abstractmethod
    def produced_output_layout(self, input_layout: tuple | None) -> tuple | None:
        """Return the permutation tuple this op produces on output.
        Receives the incoming layout so transparent ops can just return it."""

    def require_input_shape(self, model: ModelWrapper, input_index: int = 0) -> list[int]:
        """Helper to retrieve input shape, raising if not found."""
        shape = model.get_tensor_shape(self.onnx_node.input[input_index])
        if shape is None:
            raise ValueError(
                f"Tensor shape for input '{self.onnx_node.input[input_index]}' not found in model."
            )
        return shape + [1] * (4 - len(shape))  # Pad to 4D if needed.
    
    def require_output_shape(self, model: ModelWrapper, output_index: int = 0) -> list[int]:
        """Helper to retrieve output shape, raising if not found."""
        shape = model.get_tensor_shape(self.onnx_node.output[output_index])
        if shape is None:
            raise ValueError(
                f"Tensor shape for output '{self.onnx_node.output[output_index]}' not found in model."
            )
        return shape + [1] * (4 - len(shape))  # Pad to 4D if needed.

    def get_port_interface(self) -> NodeInterface:
        return NodeInterface.from_dict({
            "in_stream_array": self.get_nodeattr("in_stream_array"),
            "out_stream_array": self.get_nodeattr("out_stream_array"),
            "in_word_array": self.get_nodeattr("in_word_array"),
            "out_word_array": self.get_nodeattr("out_word_array"),
        })

    def can_inherit_interface(self) -> bool:
        return False

    def inherit_interface(self, model: ModelWrapper, upstream: NodeInterface) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support interface inheritance."
        )

class PointLike(Protocol):
    """Opaque per-operator DSE point.

    Each operator should define its own nested @dataclass (e.g., MyOp.DSEPoint)
    that implements this protocol. The global code never inspects fields.
    """
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PointLike": ...

class DSECapable(CustomOp, ABC):
    """Mixin for operators that support DSE."""
    
    @abstractmethod
    def get_dse_points(self, model: ModelWrapper) -> List[PointLike]:
        """Return ALL feasible DSE points for this operator.
        The list may be large; your operator decides feasibility.
        """
    
    @abstractmethod
    def apply_point(self, model: ModelWrapper, point: PointLike) -> None:
        """Write the chosen point into the operator's ONNX attributes."""

@dataclass(frozen=True)
class ParamDesc:
    input_index: int
    name: str
    shape: Tuple[int, ...]
    tensor_quant: TensorQuant
    in_channel_unroll: int
    out_channel_unroll: int
    width_unroll: int
    data_per_word: int
    times: int

class HasParameters(CustomOp, ABC):
    """Mixin: implement on ops that own parameters (Conv, MatMul, etc.)."""
    
    @abstractmethod
    def list_parameters(self, model: ModelWrapper) -> Iterable[ParamDesc]:
        """Describe every streamable parameter tensor the op currently uses."""
    
    def set_external_storage(self) -> None:
        self.set_nodeattr("param_storage", "EXTERNAL")