from qonnx.util import basic as qonnx_basic
from onnx import TensorAnnotation, StringStringEntryProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from onnx import TensorProto
import numpy as np
import re

def Quant_to_TensorQuant(node: NodeProto, model: ModelWrapper) -> dict:
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

class TensorQuant:
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
            Initializes a TensorQuant instance with the specified quantization parameters.
        from_quant_node(quant_node: NodeProto, model: ModelWrapper) -> TensorQuant:
            Creates a TensorQuant instance from a Quant node.
        __eq__(other) -> bool:
            Checks equality with another TensorQuant instance.
        get_canonical_name() -> str:
            Returns a canonical string representation of the quantization parameters.
        from_canonical_name(s: str) -> TensorQuant:
            Parses a canonical quantization string and returns a TensorQuant instance.
        __repr__() -> str:
            Returns a string representation of the TensorQuant instance.
    
    """

    def __init__(self, bitwidth, signed, scale, zeropt, narrow=False, rounding_mode="ROUND"):
        self.bitwidth = int(bitwidth)
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
        params = Quant_to_TensorQuant(quant_node, model)
        return cls(
            bitwidth=params["bitwidth"],
            signed=params["signed"],
            scale=params["scale"],
            zeropt=params["zeropt"],
            narrow=params["narrow"],
            rounding_mode=params["rounding_mode"],
        )
    
    def __eq__(self, other):
        if not isinstance(other, TensorQuant):
            return False
        return (
            self.bitwidth == other.bitwidth and
            self.signed == other.signed and
            self.scale == other.scale and
            self.zeropt == other.zeropt and
            self.narrow == other.narrow and
            self.rounding_mode == other.rounding_mode
        )

    def get_canonical_name(self):
        return f"Q[{self.bitwidth},{self.signed},{self.scale},{self.zeropt},{self.narrow},{self.rounding_mode}]"
    
    def get_tensorproto_dtype(self):
        """Returns the ONNX TensorProto data type corresponding to the quantization parameters."""
        bitwidth = self.bitwidth
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
            if self.bitwidth <= 8:
                return np.int8
            elif self.bitwidth <= 16:
                return np.int16
            elif self.bitwidth <= 32:
                return np.int32
            else:
                raise ValueError(f"Unsupported signed bitwidth: {self.bitwidth}")
        else:
            if self.bitwidth <= 8:
                return np.uint8
            elif self.bitwidth <= 16:
                return np.uint16
            elif self.bitwidth <= 32:
                return np.uint32
            else:
                raise ValueError(f"Unsupported unsigned bitwidth: {self.bitwidth}")

    @staticmethod
    def from_canonical_name(s):
        m = re.fullmatch(
            r"Q\[(\d+),(0|1),([0-9.eE+-]+),(-?\d+),(0|1),(\w+)\]",
            s
        )
        if not m:
            raise ValueError(f"Invalid quantization annotation string: {s}")
        return TensorQuant(
            bitwidth=int(m.group(1)),
            signed=int(m.group(2)),
            scale=float(m.group(3)),
            zeropt=int(m.group(4)),
            narrow=int(m.group(5)),
            rounding_mode=str(m.group(6))
        )

    def __repr__(self):
        return f"<TensorQuant {self.get_canonical_name()}>"

def set_custom_tensor_datatype(model: ModelWrapper, tensor_name: str, tensor_quant: TensorQuant):
    """Sets the TensorQuant of a tensor with the given name."""
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

def get_custom_tensor_datatype(model: ModelWrapper, tensor_name):
    """Gets the custom TensorQuant of a tensor with the given name.
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
        return TensorQuant.from_canonical_name(ret_dt.value)
    except Exception as e:
        raise ValueError(f"Invalid TensorQuant string for tensor {tensor_name}: {ret_dt.value}") from e
