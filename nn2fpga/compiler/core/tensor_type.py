from qonnx.util import basic as qonnx_basic
from onnx import TensorAnnotation, StringStringEntryProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from onnx import TensorProto
from abc import ABC, abstractmethod
import numpy as np
import re

class TensorType(ABC):
    """Base class representing the data type of a tensor.
    Concrete subclasses: FloatTensorType, QuantizedTensorType.
    """

    @abstractmethod
    def get_canonical_name(self) -> str: 
        """Returns a canonical string representation of the tensor type."""
        ...

    @abstractmethod
    def get_tensorproto_dtype(self) -> int: 
        """Returns the corresponding ONNX TensorProto data type (e.g. TensorProto.FLOAT, TensorProto.INT8)."""
        ...

    @abstractmethod
    def get_numpy_dtype(self): 
        """Returns the corresponding NumPy dtype (e.g. np.float32, np.int8)."""
        ...

    @abstractmethod
    def get_hls_data_type(self) -> str:
        """Returns the HLS primitive type string (e.g. 'float', 'ap_int<8>')."""
        ...

    @abstractmethod
    def get_cpp_quant_type(self) -> str:
        """Returns the C++ scalar type string (e.g. 'float', 'unsigned char')."""
        ...
    
    @abstractmethod
    def get_onnxruntime_type(self) -> str:
        """Returns the ONNX Runtime data type string (e.g. 'ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT')."""
        ...

    @abstractmethod
    def get_spec_type(self) -> str:
        """Returns a string representing the type for use in code generation specifications."""
        ...

    @staticmethod
    def from_canonical_name(s: str) -> "TensorType":
        if s == "float32":
            return FloatTensorType()
        return QuantizedTensorType.from_canonical_name(s)
    
    @property
    @abstractmethod
    def bitwidth(self) -> int:
        """Returns the bitwidth of the tensor type."""
        ...

    def __repr__(self):
        return f"<{type(self).__name__} {self.get_canonical_name()}>"


class FloatTensorType(TensorType):
    """Represents a standard IEEE-754 float32 tensor (no quantization)."""

    def get_canonical_name(self) -> str:
        return "float32"

    def get_tensorproto_dtype(self) -> int:
        return TensorProto.FLOAT

    def get_numpy_dtype(self):
        return np.float32
    
    def get_hls_data_type(self) -> str:
        return "float"
    
    def get_cpp_quant_type(self) -> str:
        return "float"
    
    def get_onnxruntime_type(self) -> str:
        return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT"

    def get_spec_type(self) -> str:
        return "f32"

    @property
    def bitwidth(self) -> int:
        return 32

    def __eq__(self, other):
        return isinstance(other, FloatTensorType)

    def __hash__(self):
        return hash("float32")

class QuantizedTensorType(TensorType):
    """
    Represents quantization parameters for an activation tensor.
    Currently, it does not support per channel/per group quantization.

    The quantization parameters are stored in a canonical string format:
    Q[bitwidth,signed,scale,zeropt,narrow,rounding_mode]
    
    Where:
    - `bitwidth`: Number of bits used for quantization.
    - `signed`: Indicates if quantization is signed (1) or unsigned (0).
    - `scale`: Scale factor for quantization.
    - `zeropt`: Zero-point offset for quantization.
    - `narrow`: Indicates if narrow range quantization is used (1) or not (0).
    - `rounding_mode`: Rounding mode used during quantization.
    
    Methods:
        __init__(bitwidth, signed, scale, zeropt, narrow=False, rounding_mode="ROUND"):
            Initializes a QuantizedTensorType instance with the specified quantization parameters.
        from_quant_node(quant_node: NodeProto, model: ModelWrapper) -> QuantizedTensorType:
            Creates a QuantizedTensorType instance from a Quant node.
        __eq__(other) -> bool:
            Checks equality with another QuantizedTensorType instance.
        get_canonical_name() -> str:
            Returns a canonical string representation of the quantization parameters.
        from_canonical_name(s: str) -> QuantizedTensorType:
            Parses a canonical quantization string and returns a QuantizedTensorType instance.
        __repr__() -> str:
            Returns a string representation of the QuantizedTensorType instance.
    
    """

    def __init__(self, bitwidth, signed, scale, zeropt, narrow=False, rounding_mode="ROUND"):
        self._bitwidth = int(bitwidth)
        self.signed = int(signed)
        if scale is None:
            raise ValueError("Scale parameter cannot be None.")
        if hasattr(scale, "size"):
            if scale.size != 1:
                raise ValueError("Scale parameter must be a scalar or single-element array.")
            self.scale = float(scale.item())
        else:
            self.scale = float(scale)
        if zeropt is None:
            raise ValueError("Zero-point parameter cannot be None.")
        if hasattr(zeropt, "size"):
            if zeropt.size != 1:
                raise ValueError("Zero-point parameter must be a scalar or single-element array.")
            self.zeropt = int(zeropt.item())
        else:
            self.zeropt = int(zeropt)
        self.narrow = int(narrow)
        self.rounding_mode = str(rounding_mode)

    @classmethod
    def from_quant_node(cls, quant_node: NodeProto, model: ModelWrapper):
        params = Quant_to_QuantizedTensorType(quant_node, model)
        return cls(
            bitwidth=params["bitwidth"],
            signed=params["signed"],
            scale=params["scale"],
            zeropt=params["zeropt"],
            narrow=params["narrow"],
            rounding_mode=params["rounding_mode"],
        )

    def __eq__(self, other):
        if not isinstance(other, QuantizedTensorType):
            return False
        return (
            self._bitwidth == other.bitwidth and
            self.signed == other.signed and
            self.scale == other.scale and
            self.zeropt == other.zeropt and
            self.narrow == other.narrow and
            self.rounding_mode == other.rounding_mode
        )

    def __hash__(self):
        return hash(
            (
                self._bitwidth,
                self.signed,
                self.scale,
                self.zeropt,
                self.narrow,
                self.rounding_mode,
            )
        )

    def get_canonical_name(self):
        return f"Q[{self._bitwidth},{self.signed},{self.scale},{self.zeropt},{self.narrow},{self.rounding_mode}]"

    def get_tensorproto_dtype(self):
        """Returns the ONNX TensorProto data type corresponding to the quantization parameters."""
        bitwidth = self._bitwidth
        if self.signed:
            if bitwidth <= 8:
                return TensorProto.INT8
            elif bitwidth <= 16:
                return TensorProto.INT16
            elif bitwidth <= 32:
                return TensorProto.INT32
            else:
                raise ValueError(f"Unsupported signed bitwidth: {bitwidth}")
        else:
            if bitwidth <= 8:
                return TensorProto.UINT8
            elif bitwidth <= 16:
                return TensorProto.UINT16
            elif bitwidth <= 32:
                return TensorProto.UINT32
            else:
                raise ValueError(f"Unsupported unsigned bitwidth: {bitwidth}")

    def get_numpy_dtype(self):
        """Returns the NumPy data type corresponding to the quantization parameters."""
        if self.signed:
            if self._bitwidth <= 8:
                return np.int8
            elif self._bitwidth <= 16:
                return np.int16
            elif self._bitwidth <= 32:
                return np.int32
            else:
                raise ValueError(f"Unsupported signed bitwidth: {self._bitwidth}")
        else:
            if self._bitwidth <= 8:
                return np.uint8
            elif self._bitwidth <= 16:
                return np.uint16
            elif self._bitwidth <= 32:
                return np.uint32
            else:
                raise ValueError(f"Unsupported unsigned bitwidth: {self._bitwidth}")
        
    def get_hls_data_type(self) -> str:
        """Returns the HLS primitive type string (e.g. 'ap_int<8>')."""
        return f"ap_{'' if self.signed else 'u'}int<{self._bitwidth}>"
    
    def get_cpp_quant_type(self) -> str:
        """Returns the C++ scalar type string (e.g. 'unsigned char')."""
        cpp_type_string = "unsigned " if not self.signed else ""
        if self._bitwidth <= 8:
            return f"{cpp_type_string}char"
        elif self._bitwidth <= 16:
            return f"{cpp_type_string}short"
        elif self._bitwidth <= 32:
            return f"{cpp_type_string}int"
        else:
            raise ValueError(f"Unsupported bitwidth: {self._bitwidth}")
    
    def get_onnxruntime_type(self) -> str:
        """Returns the ONNX Runtime data type string (e.g. 'ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8')."""
        if self.signed:
            if self._bitwidth <= 8:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8"
            elif self._bitwidth <= 16:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16"
            elif self._bitwidth <= 32:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32"
            else:
                raise ValueError(f"Unsupported signed bitwidth: {self._bitwidth}")
        else:
            if self._bitwidth <= 8:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8"
            elif self._bitwidth <= 16:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16"
            elif self._bitwidth <= 32:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32"
            else:
                raise ValueError(f"Unsupported unsigned bitwidth: {self._bitwidth}")
    
    def get_spec_type(self) -> str:
        """Returns a string representing the type for use in code generation specifications."""
        if self.signed:
            if self._bitwidth <= 8:
                return "i8"
            elif self._bitwidth <= 16:
                return "i16"
            elif self._bitwidth <= 32:
                return "i32"
            else:
                raise ValueError(f"Unsupported signed bitwidth: {self._bitwidth}")
        else:
            if self._bitwidth <= 8:
                return "u8"
            elif self._bitwidth <= 16:
                return "u16"
            elif self._bitwidth <= 32:
                return "u32"
            else:
                raise ValueError(f"Unsupported unsigned bitwidth: {self._bitwidth}")

    @property
    def bitwidth(self) -> int:
        """Returns the bitwidth of the quantized tensor type."""
        return self._bitwidth

    @staticmethod
    def from_canonical_name(s):
        m = re.fullmatch(
            r"Q\[(\d+),(0|1),([0-9.eE+-]+),(-?\d+),(0|1),(\w+)\]",
            s
        )
        if not m:
            raise ValueError(f"Invalid quantization annotation string: {s}")
        return QuantizedTensorType(
            bitwidth=int(m.group(1)),
            signed=int(m.group(2)),
            scale=float(m.group(3)),
            zeropt=int(m.group(4)),
            narrow=int(m.group(5)),
            rounding_mode=str(m.group(6))
        )

# Helper functions

def set_custom_tensor_datatype(model: ModelWrapper, tensor_name: str, tensor_quant: TensorType):
    """Sets the TensorType of a tensor with the given name."""
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")

    if ret is not None:
        ret_dt = qonnx_basic.get_by_name(ret.quant_parameter_tensor_names, "tensor_quant", "key")
        if ret_dt is not None:
            if tensor_quant is None:
                ret_dt.Clear()
            else:
                ret_dt.value = tensor_quant.get_canonical_name()
        elif tensor_quant is not None:
            dt = StringStringEntryProto()
            dt.key = "tensor_quant"
            dt.value = tensor_quant.get_canonical_name()
            ret.quant_parameter_tensor_names.append(dt)
    elif tensor_quant is not None:
        qa = TensorAnnotation()
        qa.tensor_name = tensor_name
        dt = StringStringEntryProto()
        dt.key = "tensor_quant"
        dt.value = tensor_quant.get_canonical_name()
        qa.quant_parameter_tensor_names.append(dt)
        qnt_annotations.append(qa)

def get_custom_tensor_datatype(model: ModelWrapper, tensor_name: str) -> TensorType:
    """Gets the custom TensorType of a tensor with the given name.
    Returns None if not found.
    """
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")
    if ret is None:
        return None

    ret_dt = qonnx_basic.get_by_name(ret.quant_parameter_tensor_names, "tensor_quant", "key")
    if ret_dt is None:
        return None

    try:
        return TensorType.from_canonical_name(ret_dt.value)
    except Exception as e:
        raise ValueError(f"Invalid TensorType string for tensor {tensor_name}: {ret_dt.value}") from e

def require_tensor_type(model: ModelWrapper, tensor_name: str) -> TensorType:
    """Gets the custom TensorType of a tensor with the given name.
    Raises an error if not found.
    """
    quant = get_custom_tensor_datatype(model, tensor_name)
    if quant is None:
        raise ValueError(f"Tensor quantization for tensor '{tensor_name}' not found in model.")
    return quant

def Quant_to_QuantizedTensorType(node: NodeProto, model: ModelWrapper) -> dict:
    """
    Extracts quantization parameters from a Quant node and returns them as a dictionary.

    Parameters:
        node (NodeProto): The ONNX node representing the Quant operation.
        model (ModelWrapper): The model wrapper containing initializers and graph information.

    Returns:
        dict: A dictionary containing quantization parameters:
            - scale: The scale factor for quantization.
            - zeropt: The zero-point offset for quantization.
            - bitwidth: The bitwidth used for quantization.
            - signed: Indicates if quantization is signed.
            - narrow: Indicates if narrow range quantization is used.
            - rounding_mode: The rounding_mode used during quantization.
    """
    scale = zeropt = bitwidth = None

    if len(node.input) > 1:
        scale = model.get_initializer(node.input[1])
    if len(node.input) > 2:
        zeropt = model.get_initializer(node.input[2])
    if len(node.input) > 3:
        bitwidth = model.get_initializer(node.input[3])
        if bitwidth.size > 1:
            raise ValueError(
                f"Bitwidth for node {node.name} is not a scalar: {bitwidth}"
            )
        bitwidth = int(bitwidth.item())
    qonnx_node = getCustomOp(node)
    signed = qonnx_node.get_nodeattr("signed")
    narrow = qonnx_node.get_nodeattr("narrow")
    rounding_mode = qonnx_node.get_nodeattr("rounding_mode")

    return dict(
        scale=scale,
        zeropt=zeropt,
        bitwidth=bitwidth,
        signed=signed,
        narrow=narrow,
        rounding_mode=rounding_mode,
    )

def is_constant_input_node(model: ModelWrapper, node: NodeProto) -> bool:
    """Check if the node has only constant inputs.
    It is used to distinguish between Quant nodes on the activation and
    Quant nodes on the parameters (weights and biases).
    """
    init_names = [init.name for init in model.graph.initializer]
    return all(inp in init_names for inp in node.input)
