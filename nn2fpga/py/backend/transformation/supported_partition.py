import time
import onnx
import numpy as np
from collections import deque, defaultdict
from onnx import helper, numpy_helper, TensorProto, AttributeProto
from qonnx.util.basic import get_by_name
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.create_generic_partitions import PartitionFromDict
from qonnx.transformation.general import SortGraph
from backend.core.tensor_quant import is_constant_input_node
from backend.transformation.convert_to_QCDQ import ConvertToQCDQ
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import logging
logger = logging.getLogger(__name__)

@dataclass
class Match:
    ok: bool
    pattern_name: str
    covered: Set[str]          # node names covered by this match
    reasons: List[str]         # failure reasons if ok=False

class Pattern:
    """Anchor-based pattern; match is evaluated around a specific anchor node."""
    name: str = "Pattern"
    anchor_op: str = ""
    debug_allowlist: Optional[Set[str]] = None  # optional debug filter
    
    def __init__(self, debug_allowlist: Optional[Set[str]] = None):
        self.debug_allowlist = set(debug_allowlist) if debug_allowlist is not None else None

    def match(self, model, anchor_node) -> Match:
        covered: Set[str] = set()

        # Debug filter: only consider certain node names (optional)
        if self.debug_allowlist is not None and anchor_node.name not in self.debug_allowlist:
            return Match(False, self.name, covered, [f"{anchor_node.name} not in debug allowlist"])

        # Anchor op-type check (common)
        if self.anchor_op and anchor_node.op_type != self.anchor_op:
            return Match(False, self.name, covered, [f"Not a {self.anchor_op} anchor"])

        # Delegate to subclass-specific logic
        return self._match_impl(model, anchor_node)

    def _match_impl(self, model, anchor_node) -> Match:
        raise NotImplementedError


# =========================== Helper Functions ===========================#

def _is_initializer(model, tensor_name: str) -> bool:
    return any(init.name == tensor_name for init in model.graph.initializer)

def _is_const_source(model, tensor_name: str) -> bool:
    """Initializer OR produced by Constant op."""
    if tensor_name == "" or tensor_name is None:
        return False
    if _is_initializer(model, tensor_name):
        return True
    p = model.find_producer(tensor_name)
    return (p is not None) and (p.op_type == "Constant")

def check_attribute(
    node: onnx.NodeProto, attr_name: str, expected_value, reasons: list, optional=False
) -> bool:
    """Check if the attribute is present and has the expected value.

    Args:
        node (onnx.NodeProto): The node to check.
        attr_name (str): The name of the attribute to check.
        expected_value: The expected value of the attribute.
        reasons (list): A list to append reasons for failure.
        optional (bool): If True, the attribute is optional and its absence is not an error.

    Returns:
        bool: True if the attribute is present and has the expected value, False otherwise.
    """
    attribute = get_by_name(node.attribute, attr_name)
    if attribute is None:
        if not optional:
            reasons.append(f"Attribute {attr_name} not found")
        return optional

    if attribute.type == AttributeProto.FLOAT:
        if not np.isclose(attribute.f, expected_value):
            reasons.append(
                f"Attribute {attribute.name} has unexpected value {attribute.f}, expected {expected_value}"
            )
            return False
    elif attribute.type == AttributeProto.INT:
        if attribute.i != expected_value:
            reasons.append(
                f"Attribute {attribute.name} has unexpected value {attribute.i}, expected {expected_value}"
            )
            return False
    elif attribute.type == AttributeProto.STRING:
        if attribute.s.decode() != expected_value:
            reasons.append(
                f"Attribute {attribute.name} has unexpected value {attribute.s.decode()}, expected {expected_value}"
            )
            return False
    elif attribute.type == AttributeProto.INTS:
        if not np.array_equal(list(attribute.ints), expected_value):
            reasons.append(
                f"Attribute {attribute.name} has unexpected value {list(attribute.ints)}, expected {expected_value}"
            )
            return False
    else:
        reasons.append(
            f"Attribute {attribute.name} has unsupported type {attribute.type}"
        )
        return False

    return True

def check_params_quant(model: ModelWrapper, node: onnx.NodeProto, reasons: list) -> bool: 
    """ Check params Quant node. Right now, it is only supported symmetric quantization, 
    with full range of values (narrow=0).
    Args:
        model (ModelWrapper): The ONNX model to check, wrapped in QONNX ModelWrapper.
        node (onnx.NodeProto): The node to check for activation quantization.
        reasons (list): A list to append reasons for failure.
    Returns:
        bool: True if the quantization is supported, False otherwise.
    """

    graph = model.graph

    if node is None or (node.op_type != "IntQuant" and node.op_type != "Quant"):
        reasons.append(f"Parameters Quant not found")
        return False

    # Check if node has only initializers. If not, it is an activation Quant node.
    if not is_constant_input_node(model, node):
        reasons.append(f"Parameters Quant must have initializers")
        return False

    # Get scale and zero_point initializers
    zeropt_name = node.input[2]
    zeropt = numpy_helper.to_array(get_by_name(graph.initializer, zeropt_name))

    # Check symmetric quantization
    if not np.allclose(zeropt, 0):
        reasons.append(f"Parameters Quant with unsupported asymmetric quantization")
        return False

    return True

def check_act_quant(model: ModelWrapper, node: onnx.NodeProto, reasons: list) -> bool: 
    """ Check activation Quant node. Right now, it is only supported per tensor quantization, 
    with full range of values (narrow=0).
    Args:
        model (ModelWrapper): The ONNX model to check, wrapped in QONNX ModelWrapper.
        node (onnx.NodeProto): The node to check for activation quantization.
        reasons (list): A list to append reasons for failure.
    Returns:
        bool: True if the quantization is supported, False otherwise.
    """

    graph = model.graph

    if node is None or (node.op_type != "IntQuant" and node.op_type != "Quant"):
        reasons.append(f"Activation Quant not found")
        return False

    # Check if node has initializers. If so, it isn't an activation Quant node.
    if is_constant_input_node(model, node):
        reasons.append(f"Activation Quant must not have initializers")
        return False

    # Check not narrow quantization
    if not check_attribute(node, "narrow", 0, reasons):
        return False
    if not check_attribute(node, "rounding_mode", "ROUND", reasons):
        return False

    # Get scale and zero_point initializers
    scale_name = node.input[1]
    zeropt_name = node.input[2]
    scale = numpy_helper.to_array(get_by_name(graph.initializer, scale_name))
    zeropt = numpy_helper.to_array(get_by_name(graph.initializer, zeropt_name))

    # Check if per-channel (length > 1)
    if scale.ndim > 1 or zeropt.ndim > 1:
        reasons.append(f"Activation Quant with unspported per-channel quantization")
        return False  # Per-channel quantization is not supported for activations.
    
    # Check that the zero point is close to integer values
    if float(zeropt.item()).is_integer() is False:
        reasons.append(f"Activation Quant with unsupported floating zero point")
        return False

    return True

def _same_quant_params(model: ModelWrapper, nodeA: onnx.NodeProto, nodeB: onnx.NodeProto) -> bool:
    """ Check if two Quant nodes are equal.
    Args:
        nodeA (onnx.NodeProto): The first Quant node.
        nodeB (onnx.NodeProto): The second Quant node.
    Returns:
        bool: True if the input quantizations are the same, False otherwise.
    """

    # Get the quantization parameters for each input
    scales = []
    zeropts = []

    # Get scale and zero_point initializers
    scale_name = nodeA.input[1]
    zeropt_name = nodeA.input[2]
    scale = numpy_helper.to_array(get_by_name(model.graph.initializer, scale_name))
    zeropt = numpy_helper.to_array(get_by_name(model.graph.initializer, zeropt_name))
    scales.append(scale)
    zeropts.append(zeropt)

    scale_name = nodeB.input[1]
    zeropt_name = nodeB.input[2]
    scale = numpy_helper.to_array(get_by_name(model.graph.initializer, scale_name))
    zeropt = numpy_helper.to_array(get_by_name(model.graph.initializer, zeropt_name))
    scales.append(scale)
    zeropts.append(zeropt)

    # Check if all scales and zero points are the same
    supported = True
    if any(not np.array_equal(s, scales[0]) for s in scales) or any(
        not np.array_equal(z, zeropts[0]) for z in zeropts
    ):
        supported = False

    return supported

# ============================= Patterns ============================#

class MulQuantized(Pattern):
    """ Pattern matching Mul with both inputs quantized. """
    name = "Mul(Quant(a), Quant(b))"
    anchor_op = "Mul"

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) != 2:
            return Match(False, self.name, covered, [f"Mul expects 2 inputs, got {len(anchor_node.input)}"])

        q0 = model.find_producer(anchor_node.input[0])
        q1 = model.find_producer(anchor_node.input[1])
        if q0 is None or q1 is None:
            reasons.append("Both Mul inputs must come from activation Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        if not check_act_quant(model, q0, reasons):
            return Match(False, self.name, covered, reasons)
        if not check_act_quant(model, q1, reasons):
            return Match(False, self.name, covered, reasons)

        covered.update({anchor_node.name, q0.name, q1.name})
        return Match(True, self.name, covered, reasons)

class MulHardSigmoidTimesConst(Pattern):
    """ Pattern matching Mul where one input is HardSigmoid(Quant(x)) and the other is a constant. """
    name = "Mul(HardSigmoid(Quant(x)), Const)"
    anchor_op = "Mul"

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) != 2:
            return Match(False, self.name, covered, [f"Mul expects 2 inputs, got {len(anchor_node.input)}"])

        a, b = anchor_node.input
        pa = model.find_producer(a)
        pb = model.find_producer(b)

        hs = None
        const_op = None

        if pa is not None and pa.op_type == "HardSigmoid" and _is_const_source(model, b):
            hs = pa
            const_op = pb
        elif pb is not None and pb.op_type == "HardSigmoid" and _is_const_source(model, a):
            hs = pb
            const_op = pa
        else:
            reasons.append("Expected HardSigmoid on one input and Const on the other")
            return Match(False, self.name, covered, reasons)

        # Require HardSigmoid input to be quantized (you can relax later if you want)
        hs_in = hs.input[0]
        hs_in_q = model.find_producer(hs_in)
        if hs_in_q is None:
            reasons.append("HardSigmoid input must come from activation Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        if not check_act_quant(model, hs_in_q, reasons):
            return Match(False, self.name, covered, reasons)

        covered.update({anchor_node.name, hs.name, hs_in_q.name})

        # Include const_op if it exists (it may be None if input is initializer)
        if const_op is not None:
            covered.add(const_op.name)

        return Match(True, self.name, covered, reasons)

class ConvQuantPattern(Pattern):
    """ Pattern matching Conv with quantized activation, weights, and optional bias. """
    name = "Conv(Quant(act), Quant(w), [Quant(b)]) + attr constraints"
    anchor_op = "Conv"

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) < 2:
            return Match(False, self.name, covered, ["Conv missing required inputs"])

        # activation quant
        act_q = model.find_producer(anchor_node.input[0])
        if act_q is None or not check_act_quant(model, act_q, reasons):
            reasons.append("Conv input must be produced by supported activation Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        # weight quant (params)
        w_q = model.find_producer(anchor_node.input[1])
        if w_q is None or not check_params_quant(model, w_q, reasons):
            reasons.append("Conv weights must be produced by supported params Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        covered.update({anchor_node.name, act_q.name, w_q.name})

        # optional bias quant
        if len(anchor_node.input) > 2 and anchor_node.input[2] != "":
            b_q = model.find_producer(anchor_node.input[2])
            if b_q is None or not check_params_quant(model, b_q, reasons):
                reasons.append("Conv bias must be produced by supported params Quant/IntQuant")
                return Match(False, self.name, covered, reasons)
            covered.add(b_q.name)

        # reuse your existing attribute checks (inline or call helpers)
        kernel_shape = get_by_name(anchor_node.attribute, "kernel_shape")
        if kernel_shape is None or len(kernel_shape.ints) != 2 or kernel_shape.ints[0] != kernel_shape.ints[1]:
            reasons.append("Kernel shape must be 2D and square")
            return Match(False, self.name, covered, reasons)

        group = get_by_name(anchor_node.attribute, "group")
        if group is None:
            reasons.append("Missing group attribute")
            return Match(False, self.name, covered, reasons)
        in_ch = model.get_tensor_shape(anchor_node.input[0])[1]
        if group.i != 1 and group.i != in_ch:
            reasons.append("Group must be 1 or equal to input channels")
            return Match(False, self.name, covered, reasons)

        dilations = get_by_name(anchor_node.attribute, "dilations")
        if dilations is not None and any(d != 1 for d in dilations.ints):
            reasons.append("Dilations must all be 1")
            return Match(False, self.name, covered, reasons)

        strides = get_by_name(anchor_node.attribute, "strides")
        if strides is not None:
            s = list(strides.ints)
            s = [1] * (3 - len(s)) + s
            if s[0] != 1:
                reasons.append("Stride over channels not supported")
                return Match(False, self.name, covered, reasons)
            if s[1] != s[2]:
                reasons.append("Strides must have equal H and W")
                return Match(False, self.name, covered, reasons)

        return Match(True, self.name, covered, reasons)

class GemmQuantPattern(Pattern):
    """Pattern matching Gemm with quantized activation, weights, and optional bias. Fully connected layer case."""
    name = "Gemm(Quant(act), Quant(w), [Quant(b)]) + attr/shape constraints"
    anchor_op = "Gemm"

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) < 2:
            return Match(False, self.name, covered, ["Gemm missing required inputs"])

        # ---- quant structure (pattern) ----

        # activation quant
        act_q = model.find_producer(anchor_node.input[0])
        if act_q is None or not check_act_quant(model, act_q, reasons):
            reasons.append("Gemm input must be produced by supported activation Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        # weight quant (params)
        w_q = model.find_producer(anchor_node.input[1])
        if w_q is None or not check_params_quant(model, w_q, reasons):
            reasons.append("Gemm weights must be produced by supported params Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        covered.update({anchor_node.name, act_q.name, w_q.name})

        # optional bias quant (params)
        if len(anchor_node.input) > 2 and anchor_node.input[2] != "":
            b_q = model.find_producer(anchor_node.input[2])
            if b_q is None or not check_params_quant(model, b_q, reasons):
                reasons.append("Gemm bias must be produced by supported params Quant/IntQuant")
                return Match(False, self.name, covered, reasons)
            covered.add(b_q.name)

        # ---- Gemm attribute constraints ----
        # alpha == 1.0, beta == 1.0, transB == 1, transA == 0 (optional)
        if not check_attribute(anchor_node, "alpha", 1.0, reasons):
            return Match(False, self.name, covered, reasons)
        if not check_attribute(anchor_node, "beta", 1.0, reasons):
            return Match(False, self.name, covered, reasons)
        if not check_attribute(anchor_node, "transA", 0, reasons, optional=True):
            return Match(False, self.name, covered, reasons)
        if not check_attribute(anchor_node, "transB", 1, reasons):
            return Match(False, self.name, covered, reasons)

        # ---- Input shape constraints ----
        # Supported: 2D tensor, or 4D tensor with spatial dims [1,1]
        in_shape = model.get_tensor_shape(anchor_node.input[0])
        if in_shape is None:
            reasons.append("Could not determine Gemm input shape")
            return Match(False, self.name, covered, reasons)

        if len(in_shape) == 2:
            pass  # OK
        elif len(in_shape) == 4 and in_shape[2:] == [1, 1]:
            pass  # OK (flattened FC case)
        else:
            reasons.append(f"Unsupported Gemm input shape {in_shape} (expected 2D or 4D with [1,1])")
            return Match(False, self.name, covered, reasons)

        return Match(True, self.name, covered, reasons)

class PoolQuantPattern(Pattern):
    """
    Matches MaxPool/AveragePool with quantized activation input + basic HW constraints:
      - input produced by activation Quant/IntQuant
      - no dilations (or all ones)
      - equal H/W strides (and no channel stride)
    """
    name = "Pool(Quant(act)) + attr constraints"
    anchor_op = ""  # set by subclasses

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) < 1:
            return Match(False, self.name, covered, ["Pool missing required input"])

        # ---- quant structure (pattern) ----
        act_q = model.find_producer(anchor_node.input[0])
        if act_q is None or not check_act_quant(model, act_q, reasons):
            reasons.append("Pool input must be produced by supported activation Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        covered.update({anchor_node.name, act_q.name})

        # ---- attribute constraints ----

        # Only supported Pool without dilations (or all ones)
        dilations = get_by_name(anchor_node.attribute, "dilations")
        if dilations is not None and any(d != 1 for d in dilations.ints):
            reasons.append("Dilations must have all values equal to 1")
            return Match(False, self.name, covered, reasons)

        # Only supported Pool with equal strides on H/W; and no stride over channels
        strides = get_by_name(anchor_node.attribute, "strides")
        if strides is not None:
            s = list(strides.ints)
            s = [1] * (3 - len(s)) + s  # ensure [C, H, W]
            if s[0] != 1:
                reasons.append("Strides over channels is not supported")
                return Match(False, self.name, covered, reasons)
            if s[1] != s[2]:
                reasons.append("Strides must have equal H and W values")
                return Match(False, self.name, covered, reasons)

        return Match(True, self.name, covered, reasons)

class MaxPoolQuantPattern(PoolQuantPattern):
    anchor_op = "MaxPool"
    name = "MaxPool(Quant(act)) + attr constraints"

class AveragePoolQuantPattern(PoolQuantPattern):
    anchor_op = "AveragePool"
    name = "AveragePool(Quant(act)) + attr constraints"

class GlobalPoolQuantPattern(Pattern):
    """
    Matches GlobalMaxPool/GlobalAveragePool with quantized activation input.
    """
    name = "GlobalPool(Quant(act))"
    anchor_op = ""  # set by subclasses

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) < 1:
            return Match(False, self.name, covered, ["GlobalPool missing required input"])

        act_q = model.find_producer(anchor_node.input[0])
        if act_q is None or not check_act_quant(model, act_q, reasons):
            reasons.append("GlobalPool input must be produced by supported activation Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        covered.update({anchor_node.name, act_q.name})
        return Match(True, self.name, covered, reasons)

class GlobalMaxPoolQuantPattern(GlobalPoolQuantPattern):
    anchor_op = "GlobalMaxPool"
    name = "GlobalMaxPool(Quant(act))"

class GlobalAveragePoolQuantPattern(GlobalPoolQuantPattern):
    anchor_op = "GlobalAveragePool"
    name = "GlobalAveragePool(Quant(act))"

class ResizeQuantUpsampleNearestAsymmetric(Pattern):
    """
    Matches Resize used as upsampling with quantized activation input and strict constraints:
      - input[0] produced by activation Quant/IntQuant
      - coordinate_transformation_mode == "asymmetric"
      - mode == "nearest"
      - roi input (input[1]) must be empty
      - scales input (input[2]) must be present and initializer with [1.0, 1.0, s, s]
    """
    name = "Resize(Quant(act)) upsample nearest/asymmetric + scales [1,1,s,s]"
    anchor_op = "Resize"

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) < 3:
            return Match(False, self.name, covered, [f"Resize expects at least 3 inputs, got {len(anchor_node.input)}"])

        # ---- quant structure (pattern) ----
        act_q = model.find_producer(anchor_node.input[0])
        if act_q is None or not check_act_quant(model, act_q, reasons):
            reasons.append("Resize input must be produced by supported activation Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        covered.update({anchor_node.name, act_q.name})

        # ---- attribute constraints ----
        if not check_attribute(anchor_node, "coordinate_transformation_mode", "asymmetric", reasons):
            return Match(False, self.name, covered, reasons)

        if not check_attribute(anchor_node, "mode", "nearest", reasons):
            return Match(False, self.name, covered, reasons)

        # ---- input constraints ----
        roi_name = anchor_node.input[1]
        scales_name = anchor_node.input[2]

        # Roi must be empty
        if roi_name != "":
            reasons.append("Resize roi input must be empty")
            return Match(False, self.name, covered, reasons)

        # Scales must exist and be an initializer
        if scales_name == "":
            reasons.append("Resize scales input must be present")
            return Match(False, self.name, covered, reasons)

        if not _is_initializer(model, scales_name):
            reasons.append("Resize scales must be provided as an initializer")
            return Match(False, self.name, covered, reasons)

        scales_init = next((i for i in model.graph.initializer if i.name == scales_name), None)
        if scales_init is None:
            reasons.append("Resize scales initializer not found")
            return Match(False, self.name, covered, reasons)

        scales = numpy_helper.to_array(scales_init)

        if scales.shape[0] != 4:
            reasons.append(f"Resize scales must have shape [4], got {list(scales.shape)}")
            return Match(False, self.name, covered, reasons)

        if not np.allclose(scales[0:2], [1.0, 1.0]):
            reasons.append("Resize scales first two dims must be [1.0, 1.0]")
            return Match(False, self.name, covered, reasons)

        if not np.isclose(scales[2], scales[3]):
            reasons.append("Resize scales H and W factors must be equal")
            return Match(False, self.name, covered, reasons)

        return Match(True, self.name, covered, reasons)

class AddQuant(Pattern):
    """
    Matches Add where both inputs come from activation Quant/IntQuant.
    """
    name = "Add(Quant(a), Quant(b))"
    anchor_op = "Add"

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()
        
        if len(anchor_node.input) != 2:
            return Match(False, self.name, covered, [f"Add expects 2 inputs, got {len(anchor_node.input)}"])

        q0 = model.find_producer(anchor_node.input[0])
        q1 = model.find_producer(anchor_node.input[1])
        if q0 is None or q1 is None:
            reasons.append("Both Add inputs must come from activation Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        if not check_act_quant(model, q0, reasons):
            return Match(False, self.name, covered, reasons)
        if not check_act_quant(model, q1, reasons):
            return Match(False, self.name, covered, reasons)

        # if not _same_quant_params(model, q0, q1):
        #     reasons.append("Add inputs quant params differ")
        #     return Match(False, self.name, covered, reasons)

        covered.update({anchor_node.name, q0.name, q1.name})
        return Match(True, self.name, covered, reasons)

class ConcatQuantSameParamsAxis1(Pattern):
    """
    Matches Concat where all inputs come from activation Quant/IntQuant
    AND have identical (scale, zeropt), AND axis==1.
    """
    name = "Concat(Quant(...)) same params + axis=1"
    anchor_op = "Concat"

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) < 2:
            return Match(False, self.name, covered, ["Concat expects >=2 inputs"])

        if (
            not check_attribute(anchor_node, "axis", 1, reasons)
            and not check_attribute(anchor_node, "axis", 2, reasons)
            and not check_attribute(anchor_node, "axis", 3, reasons)
        ):
            return Match(False, self.name, covered, reasons)

        qnodes = []
        for inp in anchor_node.input:
            q = model.find_producer(inp)
            if q is None:
                reasons.append("All Concat inputs must come from activation Quant/IntQuant")
                return Match(False, self.name, covered, reasons)
            if not check_act_quant(model, q, reasons):
                return Match(False, self.name, covered, reasons)

            qnodes.append(q)

        # all equal to first
        first = qnodes[0]
        if any(not _same_quant_params(model, first, q) for q in qnodes[1:]):
            reasons.append("Concat inputs quant params differ")
            return Match(False, self.name, covered, reasons)

        covered.add(anchor_node.name)
        covered.update(q.name for q in qnodes)
        return Match(True, self.name, covered, reasons)

class ReshapeFlattenFCOnly(Pattern):
    """
    Accept only reshapes whose shape input is a constant and matches one of:
      - [B, C, -1]  (preserves channels)
      - [B, -1]     (only allowed when input H=W=1)
    Everything else is rejected.
    """
    name = "Reshape/Flatten FC-only (strict shape-const)"
    anchor_op = ""  # set in subclasses

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        # must have input tensor and shape/second input
        if len(anchor_node.input) < 2 or len(anchor_node.output) < 1:
            return Match(False, self.name, covered, ["Missing input/shape/output"])

        in_shape = model.get_tensor_shape(anchor_node.input[0])
        if in_shape is None or len(in_shape) != 4:
            return Match(False, self.name, covered, [f"Input must be 4D, got {in_shape}"])

        # must have a constant shape vector as the second input
        shape_input = anchor_node.input[1]
        shape_const = model.get_initializer(shape_input)

        if shape_const is None:
            return Match(False, self.name, covered, ["Reshape shape is not a constant"])

        vec = [int(v) for v in list(shape_const)]
        B_in, C_in, H_in, W_in = in_shape

        # Pattern A: [B, C, -1]
        if len(vec) == 3 and vec[0] == B_in and vec[1] == C_in and vec[2] == -1:
            covered.add(anchor_node.name)
            return Match(True, self.name, covered, reasons)

        # Pattern B: [B, -1] allowed only when H=W=1
        if len(vec) == 2 and vec[0] == B_in and vec[1] == -1:
            if H_in == 1 and W_in == 1:
                covered.add(anchor_node.name)
                return Match(True, self.name, covered, reasons)
            else:
                return Match(False, self.name, covered, [f"[B,-1] only allowed when H=W=1, got H={H_in},W={W_in}"])

        # otherwise reject
        return Match(False, self.name, covered,
                     [f"Unsupported constant reshape shape {vec}; allowed: [B,C,-1] or [B,-1](if H=W=1)"])


class ReshapeFCOnly(ReshapeFlattenFCOnly):
    anchor_op = "Reshape"
    name = "Reshape FC-only shape constraints"

class FlattenFCOnly(ReshapeFlattenFCOnly):
    anchor_op = "Flatten"
    name = "Flatten FC-only shape constraints"

class QuantizedActivationPattern(Pattern):
    """
    Matches supported quantized activations
    with the constraint: input must come from activation Quant/IntQuant.
    """
    name = "QuantizedActivation(Quant(x))"
    anchor_op = ""  # set in subclasses

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) < 1:
            return Match(False, self.name, covered, ["Missing input"])

        act_q = model.find_producer(anchor_node.input[0])
        if act_q is None or not check_act_quant(model, act_q, reasons):
            reasons.append(f"{self.anchor_op} input must be produced by supported activation Quant/IntQuant")
            return Match(False, self.name, covered, reasons)

        covered.update({anchor_node.name, act_q.name})
        return Match(True, self.name, covered, reasons)

class HardSigmoidQuant(QuantizedActivationPattern):
    anchor_op = "HardSigmoid"
    name = "HardSigmoid(Quant(x))"

class LeakyReluQuant(QuantizedActivationPattern):
    anchor_op = "LeakyRelu"
    name = "LeakyRelu(Quant(x))"

class SigmoidQuant(QuantizedActivationPattern):
    anchor_op = "Sigmoid"
    name = "Sigmoid(Quant(x))"

class SwishQuant(QuantizedActivationPattern):
    anchor_op = "Swish"
    name = "Swish(Quant(x))"

class ReluQuantOrFusable(Pattern):
    """
    Mirrors your current Relu logic:
      - Either: Relu input is produced by activation Quant/IntQuant satisfying check_act_quant
      - Or: Relu input producer is in [Conv, Gemm, Add] (assumed fusable)
    """
    name = "Relu(Quant(x)) or fused into Conv/Gemm/Add"
    anchor_op = "Relu"

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()

        if len(anchor_node.input) < 1:
            return Match(False, self.name, covered, ["Missing input"])

        prod = model.find_producer(anchor_node.input[0])
        if prod is None:
            reasons.append("Relu input has no producer")
            return Match(False, self.name, covered, reasons)

        # Case 1: quantized input
        if prod.op_type in ("Quant", "IntQuant"):
            if not check_act_quant(model, prod, reasons):
                return Match(False, self.name, covered, reasons)
            covered.update({anchor_node.name, prod.name})
            return Match(True, self.name, covered, reasons)

        # Case 2: fusable into preceding op
        if prod.op_type in ("Conv", "Gemm", "Add"):
            covered.update({anchor_node.name})
            return Match(True, self.name, covered, reasons)

        reasons.append("Relu must be quantized or fusable into Conv/Gemm/Add")
        return Match(False, self.name, covered, reasons)

class ActivationQuantNodePattern(Pattern):
    """
    Matches standalone activation Quant/IntQuant nodes (non-constant inputs),
    to allow explicit support of quantizers that aren't only "covered" by other patterns.
    This mirrors your original special-case handling for Quant nodes.
    """
    name = "Activation Quant/IntQuant node"
    anchor_op = ""  # set in subclasses

    def _match_impl(self, model, anchor_node) -> Match:
        reasons: List[str] = []
        covered: Set[str] = set()
        
        # Only activation quantizers (not params quantizers)
        if is_constant_input_node(model, anchor_node):
            reasons.append("Not an activation quant node (has constant inputs)")
            return Match(False, self.name, covered, reasons)

        if not check_act_quant(model, anchor_node, reasons):
            return Match(False, self.name, covered, reasons)

        covered.add(anchor_node.name)
        return Match(True, self.name, covered, reasons)

class QuantNodePattern(ActivationQuantNodePattern):
    anchor_op = "Quant"
    name = "Activation Quant node"

class IntQuantNodePattern(ActivationQuantNodePattern):
    anchor_op = "IntQuant"
    name = "Activation IntQuant node"

class SliceSplitTreeFeasibleQuantized(Pattern):
    """
    Supported iff:
      - slice data input produced by supported activation Quant/IntQuant
      - all sibling Slice nodes (same input[0]) are static, single-axis, step=1
      - siblings form a full tiling [0..D) along that axis (no gaps/overlaps)
      - every Slice output is consumed by a Quant/IntQuant and all those quant nodes
        share the same quant params (checked via _same_quant_params)
      - Constant parameter nodes (starts/ends/axes/steps produced by Constant op)
        are included in `covered`.
    """

    name = "SliceSplitTreeFeasibleQuantized"
    anchor_op = "Slice"

    @dataclass
    class SimpleInterval:
        axis: int
        start: int
        end: int

    def __init__(self, allowed_axes: Set[int] = {1, 2, 3}, **kwargs):
        self.allowed_axes = set(allowed_axes)
        super().__init__(**kwargs)

    def _match_impl(self, model, anchor_node) -> Match:
        covered: Set[str] = set()
        reasons: List[str] = []

        # quick sanity
        if len(anchor_node.input) < 3:
            return Match(False, self.name, covered, ["Slice missing required inputs"])

        data_in = anchor_node.input[0]

        # (A) data input must be quantized
        prod = model.find_producer(data_in)
        if prod is None:
            return Match(False, self.name, covered, ["Slice data input has no producer"])
        if not check_act_quant(model, prod, reasons):
            # check_act_quant appends reasons; keep concise message
            return Match(False, self.name, covered, reasons)

        # (B) collect siblings (self-contained scan)
        siblings = self._collect_sibling_slices(model, data_in)
        if not siblings:
            return Match(False, self.name, covered, [f"No sibling Slice nodes found for {data_in}"])

        # (C) parse each sibling -> interval, track const parameter nodes and output quantizers
        data_shape = model.get_tensor_shape(data_in)
        if data_shape is None:
            return Match(False, self.name, covered, [f"Unknown shape for tensor {data_in}"])

        intervals: List[Tuple[SliceSplitTreeFeasibleQuantized.SimpleInterval, object]] = []
        const_param_nodes: Set[str] = set()
        output_quant_nodes: List[object] = []

        for s in siblings:
            ok, iv_or_reason, const_nodes, out_q = self._parse_single_slice(model, s, data_shape)
            if not ok:
                reasons.append(f"{s.name}: {iv_or_reason}")
            else:
                intervals.append((iv_or_reason, s))
                const_param_nodes.update(const_nodes)
                if out_q is None:
                    reasons.append(f"{s.name}: slice output has no consumer quantizer")
                else:
                    output_quant_nodes.append(out_q)

        if reasons:
            return Match(False, self.name, covered, reasons)

        # (D) single axis check
        axes = {iv.axis for (iv, _) in intervals}
        if len(axes) != 1:
            return Match(False, self.name, covered, [f"Mixed slice axes: {sorted(list(axes))}"])
        axis = axes.pop()
        if axis not in self.allowed_axes:
            return Match(False, self.name, covered, [f"Axis {axis} not allowed"])

        # (E) full tiling / contiguous coverage check
        simple_iv_list = sorted([(iv.start, iv.end) for (iv, _) in intervals], key=lambda x: x[0])
        if not self._is_full_tiling(simple_iv_list, int(data_shape[axis])):
            return Match(False, self.name, covered, ["Slices do not tile [0..D) contiguously (no gaps/overlaps)"])

        # (F) all output quantizers must agree on quant params
        if not output_quant_nodes:
            return Match(False, self.name, covered, ["No output quantizers found"])
        first_q = output_quant_nodes[0]
        for qn in output_quant_nodes[1:]:
            if not _same_quant_params(model, first_q, qn):
                return Match(False, self.name, covered, ["Slice output quant params differ among siblings"])

        # (G) success: mark covered nodes (anchor slice, data quant producer, param constants)
        covered.update({anchor_node.name, prod.name})
        covered.update(const_param_nodes)
        # optionally mark all siblings as covered:
        # covered.update({s.name for _, s in intervals})

        logger.info(f"{self.name}: tensor {data_in} can be split on axis {axis} into intervals {simple_iv_list}")
        return Match(True, self.name, covered, [])

    # -------------------- helpers --------------------

    def _collect_sibling_slices(self, model, data_in: str) -> List[object]:
        """Return Slice nodes that read the exact same data input."""
        return [n for n in model.graph.node if n.op_type == "Slice" and len(n.input) >= 3 and n.input[0] == data_in]

    def _get_const_array_and_constnode(self, model, tensor_name: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Return (np.array, const_node_name) for initializer or Constant node, else (None, None)."""
        if not tensor_name:
            return None, None
        init = get_by_name(model.graph.initializer, tensor_name)
        if init is not None:
            return numpy_helper.to_array(init), None
        prod = model.find_producer(tensor_name)
        if prod is None or prod.op_type != "Constant":
            return None, None
        v = get_by_name(prod.attribute, "value")
        if v is None:
            return None, prod.name
        return numpy_helper.to_array(v.t), prod.name

    def _first_consumer_node(self, model, tensor_name: str):
        """Return the first consumer node that lists tensor_name as an input (or None)."""
        for n in model.graph.node:
            if tensor_name in n.input:
                return n
        return None

    def _parse_single_slice(self, model, slice_node, data_shape: List[int]) -> Tuple[bool, object, Set[str], Optional[object]]:
        """
        Parse a single Slice node to SimpleInterval. Returns:
          (ok, interval_or_reason, used_const_nodes, first_output_quant_node_or_None)
        """
        used_const_nodes: Set[str] = set()

        # starts / ends required
        starts_arr, cn1 = self._get_const_array_and_constnode(model, slice_node.input[1])
        ends_arr, cn2 = self._get_const_array_and_constnode(model, slice_node.input[2])
        if cn1: used_const_nodes.add(cn1)
        if cn2: used_const_nodes.add(cn2)
        if starts_arr is None or ends_arr is None:
            return False, "starts/ends must be constant", used_const_nodes, None

        starts = [int(x) for x in np.array(starts_arr).flatten()]
        ends = [int(x) for x in np.array(ends_arr).flatten()]
        if len(starts) != len(ends):
            return False, "starts/ends length mismatch", used_const_nodes, None

        # axes (optional)
        if len(slice_node.input) >= 4 and slice_node.input[3] != "":
            axes_arr, cn3 = self._get_const_array_and_constnode(model, slice_node.input[3])
            if cn3: used_const_nodes.add(cn3)
            if axes_arr is None:
                return False, "axes must be constant if present", used_const_nodes, None
            axes = [int(x) for x in np.array(axes_arr).flatten()]
        else:
            axes = list(range(len(starts)))

        # steps (optional)
        if len(slice_node.input) >= 5 and slice_node.input[4] != "":
            steps_arr, cn4 = self._get_const_array_and_constnode(model, slice_node.input[4])
            if cn4: used_const_nodes.add(cn4)
            if steps_arr is None:
                return False, "steps must be constant if present", used_const_nodes, None
            steps = [int(x) for x in np.array(steps_arr).flatten()]
        else:
            steps = [1] * len(starts)

        if not (len(axes) == len(starts) == len(steps)):
            return False, "axes/starts/steps length mismatch", used_const_nodes, None
        if len(axes) != 1:
            return False, f"multi-axis Slice not supported (axes={axes})", used_const_nodes, None

        axis = int(axes[0])
        if axis < 0:
            axis += len(data_shape)
        if axis < 0 or axis >= len(data_shape):
            return False, f"axis out of range: {axes[0]}", used_const_nodes, None

        if int(steps[0]) != 1:
            return False, f"non-unit step not supported (step={steps[0]})", used_const_nodes, None

        dim = int(data_shape[axis])
        st = int(starts[0]); en = int(ends[0])
        if st < 0: st += dim
        if en < 0: en += dim
        st = max(0, min(dim, st))
        en = max(0, min(dim, en))
        if en < st:
            return False, f"invalid interval after normalize/clamp [{st},{en})", used_const_nodes, None

        # find first consumer of the slice output and require it to be an activation quant node
        out_name = slice_node.output[0] if slice_node.output else None
        out_cons = self._first_consumer_node(model, out_name) if out_name else None
        if out_cons is None or not check_act_quant(model, out_cons, []):
            # we require slice outputs to be consumed by a quantizer (as per your rules)
            return False, "slice output not consumed by supported activation Quant/IntQuant", used_const_nodes, None

        return True, self.SimpleInterval(axis=axis, start=st, end=en), used_const_nodes, out_cons

    def _is_full_tiling(self, intervals: List[Tuple[int, int]], dim: int) -> bool:
        """Intervals sorted by start -> check they tile [0..dim) exactly with no gaps/overlaps."""
        if not intervals:
            return False
        if intervals[0][0] != 0:
            return False
        prev_end = intervals[0][1]
        for st, en in intervals[1:]:
            if st != prev_end:
                return False
            prev_end = en
        return prev_end == dim

class MatMulQuantPattern(Pattern):

    
PATTERNS_BY_OP: Dict[str, List[Pattern]] = {
    "Add": [AddQuant()],
    "Concat": [ConcatQuantSameParamsAxis1()],
    "Flatten": [FlattenFCOnly()],
    "Reshape": [ReshapeFCOnly()],
    "HardSigmoid": [HardSigmoidQuant()],
    "LeakyRelu": [LeakyReluQuant()],
    "Relu": [ReluQuantOrFusable()],
    "Sigmoid": [SigmoidQuant()],
    "Swish": [SwishQuant()],
    "Slice": [SliceSplitTreeFeasibleQuantized(allowed_axes={1,2,3})],
    "IntQuant": [IntQuantNodePattern()],
    "Quant": [QuantNodePattern()],
    "AveragePool": [AveragePoolQuantPattern()],
    "Conv": [ConvQuantPattern()],  # debug allowlist by node name prefix
    "Gemm": [GemmQuantPattern()],
    "GlobalMaxPool": [GlobalMaxPoolQuantPattern()],
    "GlobalAveragePool": [GlobalAveragePoolQuantPattern()],
    "MaxPool": [MaxPoolQuantPattern()],
    "Mul": [MulHardSigmoidTimesConst(), MulQuantized()],  # order = priority
    "Resize": [ResizeQuantUpsampleNearestAsymmetric()],
    "MatMul": [MatMulQuantPattern()],
}

def match_supported_patterns(model, node) -> Match:
    patterns = PATTERNS_BY_OP.get(node.op_type, [])
    if not patterns:
        return Match(False, "NoPatternsForOp", set(), [f"No patterns registered for op {node.op_type}"])

    failures: List[str] = []
    for p in patterns:
        m = p.match(model, node)
        if m.ok:
            return m
        failures.append(f"{p.name}: " + ("; ".join(m.reasons) if m.reasons else "failed"))
    return Match(False, "NoPatternMatched", set(), failures)

class PreProcessPartitionModel(Transformation):
    """ Pre-process the model to ensure it is suitable for qonnx partitioning.
    This transformation modifies the model to handle Resize nodes with empty inputs.
    It adds dummy initializers and attributes to track which inputs were originally empty.
    """
    
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """ Apply the pre-processing transformation to the model. """
        graph = model.graph

        for node in [n for n in graph.node if n.op_type == "Resize"]:

            input_mask = []
            for i, inp in enumerate(node.input):
                if inp == "":
                    # Mark this index as originally empty
                    input_mask.append(1)
                    # Create dummy initializer
                    dummy_name = f"{node.name}_dummy_input_{i}"
                    dummy_tensor = helper.make_tensor(
                        name=dummy_name,
                        data_type=TensorProto.FLOAT,
                        dims=[1],  # Scalar or 1D dummy input
                        vals=[1.0],
                    )
                    graph.initializer.append(dummy_tensor)
                    node.input[i] = dummy_name
                else:
                    input_mask.append(0)

            # Save the input mask as an attribute
            mask_attr = helper.make_attribute("__resize_input_mask", input_mask)
            node.attribute.append(mask_attr)
        
        return (model, False)

class PostProcessPartitionModel(Transformation):
    """ Post-process the model to restore ONNX compliance after partitioning.
    """
    
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        dummy_inputs = []

        for node in [n for n in graph.node if n.op_type == "Resize"]:

            attr_map = {a.name: a for a in node.attribute}
            if "__resize_input_mask" not in attr_map:
                continue

            input_mask = attr_map["__resize_input_mask"].ints
            for i, mask_value in enumerate(input_mask):
                if mask_value == 1:
                    dummy_inputs.append(node.input[i])
                    node.input[i] = ""

            # Remove the temporary attribute
            node.attribute.remove(attr_map["__resize_input_mask"])

        # Filter out dummy initializers
        remaining_initializers = [
            init for init in graph.initializer if init.name not in dummy_inputs
        ]

        # Clear and replace initializers
        del graph.initializer[:]
        graph.initializer.extend(remaining_initializers)

        return (model, False)

class SupportedPartition(Transformation):
    """ Extracts from the ONNX a subgraph containing only operations supported by nn2FPGA.
    All the other operations are assigned to CPU.
    """

    def __init__(self, partition_directory: str = "partitions"):
        super().__init__()
        self.partition_directory = partition_directory

    from typing import Set, Tuple, List

    def __repair_convexity_remove_reentries(
        self,
        model: "ModelWrapper",
        fpga_nodes: Set[str],
    ) -> Set[str]:
        """
        Repair convexity by iteratively removing all 'dst' nodes that are reached
        via FPGA -> outside... -> FPGA paths (re-entries).

        Guaranteed to terminate and yield a set with no convexity violations.
        """
        fpga_nodes = set(fpga_nodes)

        while True:
            violations = self.__detect_convexity_violations(model, fpga_nodes)
            if not violations:
                return fpga_nodes

            # Remove every re-entry destination
            to_remove = {dst for (_src, dst, _path) in violations}
            logger.info(f"Detected {len(violations)} convexity violations.")
            logger.info(f"Removing {len(to_remove)} FPGA nodes to repair convexity violations")
            fpga_nodes.difference_update(to_remove)

    def __detect_convexity_violations(
        self,
        model: "ModelWrapper",
        fpga_nodes: Set[str],
    ) -> List[Tuple[str, str, List[str]]]:
        """
        Detect convexity violations in a candidate FPGA node set.

        A violation is a path u -> ... -> v where:
        - u and v are in fpga_nodes
        - all intermediate nodes are NOT in fpga_nodes (i.e., outside / CPU part)

        Returns a list of (src_fpga, dst_fpga, witness_path_node_names).
        witness_path includes src and dst.
        """
        graph = model.graph
        name_to_node: Dict[str, object] = {n.name: n for n in graph.node}

        # Helper: iterate direct successor names (safe if wrapper returns None)
        # build once at top of __detect_convexity_violations
        succ_cache: Dict[str, List[str]] = {}

        def succ_names(node_name: str) -> List[str]:
            try:
                return succ_cache[node_name]
            except KeyError:
                node = name_to_node[node_name]
                succs = model.find_direct_successors(node) or []
                out = [s.name for s in succs if s is not None]
                succ_cache[node_name] = out
                return out

        violations: List[Tuple[str, str, List[str]]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        # For each FPGA node u, BFS through OUTSIDE nodes only; if we reach FPGA node v => violation u->v
        for u in fpga_nodes:
            # Start frontier: successors of u that are outside
            q = deque()
            parent: Dict[str, Optional[str]] = {}  # parent for path reconstruction (only for visited outside nodes)

            for s in succ_names(u):
                if s in fpga_nodes:
                    # Direct FPGA->FPGA edge is NOT a convexity violation; it's fine.
                    continue
                if s not in parent:
                    parent[s] = None  # root (outside neighbor of u)
                    q.append(s)

            visited_outside: Set[str] = set(parent.keys())

            while q:
                x = q.popleft()

                for y in succ_names(x):
                    if y in fpga_nodes:
                        # Found re-entry: u -> (outside...) -> y
                        pair = (u, y)
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)

                        # Reconstruct witness path: u -> ... -> x -> y
                        path_mid = []
                        cur = x
                        while cur is not None:
                            path_mid.append(cur)
                            cur = parent[cur]
                        path_mid.reverse()  # from first outside node after u to x
                        witness = [u] + path_mid + [y]
                        violations.append((u, y, witness))
                        continue

                    # Continue BFS only through outside nodes
                    if y in visited_outside:
                        continue
                    visited_outside.add(y)
                    parent[y] = x
                    q.append(y)

        return violations

    def __find_largest_connected_component(self, model: ModelWrapper) -> list:
        
        graph = model.graph

        fpga_nodes = set()
        node_names = {n.name for n in graph.node}

        for node in graph.node:
            patterns = PATTERNS_BY_OP.get(node.op_type, [])
            if not patterns:
                continue

            m = match_supported_patterns(model, node)
            if not m.ok:
                logger.info(f"Node {node.name} ({node.op_type}) not supported for FPGA: {m.reasons}")
                continue
            else:
                logger.info(f"Node {node.name} ({node.op_type}) supported for FPGA via pattern '{m.pattern_name}'")

            # Add all covered nodes that exist in the graph
            fpga_nodes.update(n for n in m.covered if n in node_names)
        
        fpga_nodes = self.__repair_convexity_remove_reentries(model, fpga_nodes)

        # Build adjacency list between FPGA nodes
        adj = defaultdict(list)
        for node in graph.node:
            if node.name not in fpga_nodes:
                continue
            for succ in model.find_direct_successors(node) or []:
                if succ.name in fpga_nodes:
                    adj[node.name].append(succ.name)
                    adj[succ.name].append(node.name)  # undirected edge

        # Find connected components using BFS
        visited = set()
        components = []

        for node_name in fpga_nodes:
            if node_name in visited:
                continue
            component = set()
            queue = deque([node_name])
            visited.add(node_name)

            while queue:
                curr = queue.popleft()
                component.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

        # Select the largest component
        largest_component = max(components, key=len, default=set())
        logger.info(f"Found {len(components)} connected components among FPGA-supported nodes.")
        logger.info(f"Largest component has {len(largest_component)} nodes.")
        return largest_component

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        logger.info("Partitioning model into FPGA and CPU partitions.")
        graph = model.graph

        # Find the largest connected component of FPGA-supported nodes
        largest_component = self.__find_largest_connected_component(model)

        # Create a partition dictionary
        node_list = [node.name for node in graph.node]
        partition_dict = {
            "FPGA": [node_list.index(node) for node in largest_component]
        }

        if len(partition_dict["FPGA"]) == 0:
            logger.warning("No FPGA-supported nodes found in the model. Returning original model.")
            return (model, False)
        
        # Since there is something to be run on FPGA, we import the opset of nn2fpga
        model.model.opset_import.append(
            helper.make_opsetid("backend.custom_op", 1)
        )

        # Pre-process the model to ensure it is suitable for partitioning
        model = model.transform(PreProcessPartitionModel())

        # Create a partition from the dictionary
        parent_model = model.transform(PartitionFromDict(partition_dict, self.partition_directory))

        # Post-process the partitioned model to restore ONNX compliance
        parent_model = parent_model.transform(PostProcessPartitionModel())
        parent_model.save(self.partition_directory + "/wrapper_model.onnx")
        logger.info(f"Saved partitioned model wrapper to {self.partition_directory}/wrapper_model.onnx")

        # Load the FPGA partition model
        FPGA_model = ModelWrapper(self.partition_directory + "/partition_FPGA.onnx")
        FPGA_model = FPGA_model.transform(PostProcessPartitionModel())

        # Assign as a metadata attribute of the partitioned model the name of the node which it corresponds to.
        partition_nodes = parent_model.get_nodes_by_op_type("nn2fpgaPartition")
        if len(partition_nodes) != 1:
            raise ValueError("Extracting more than one HW partition is not supported.")

        logger.info(f"Out of {len(graph.node)} nodes, {len(partition_dict['FPGA'])} nodes are assigned to FPGA partition.")
        return (FPGA_model, False)
