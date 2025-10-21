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
import logging
logger = logging.getLogger(__name__)

FPGA_SUPPORTED_QUANTIZED_ACTIVATIONS = {
    "LeakyRelu",
    "Sigmoid",
    "Swish",
}

FPGA_SUPPORTED_OPS = {
    "Add",
    # "AveragePool",
    # "Concat",
    "Conv",
    "Flatten",
    "Gemm",
    "GlobalAveragePool",
    # "GlobalMaxPool",
    # "IntQuant",
    # "MaxPool",
    "Quant",
    "Relu",
    "Reshape",
    # "Resize",
}

FPGA_SUPPORTED_OPS.update(FPGA_SUPPORTED_QUANTIZED_ACTIVATIONS)

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

    # Check not narrow quantization
    if not check_attribute(node, "narrow", 0, reasons):
        return False
    if not check_attribute(node, "rounding_mode", "ROUND", reasons):
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

def check_input_same_quant(model: ModelWrapper, node: onnx.NodeProto, reasons: list) -> bool:
    """ Check if the input quantizations of multi-input operations are the same.
    Args:
        model (ModelWrapper): The ONNX model to check, wrapped in QONNX ModelWrapper.
        node (onnx.NodeProto): The node to check for input quantization.
        reasons (list): A list to append reasons for failure.
    Returns:
        bool: True if the input quantizations are the same, False otherwise.
    """

    graph = model.graph
    input_names = node.input
    supported = True

    # Get the quantization parameters for each input
    scales = []
    zeropts = []
    for input_name in input_names:
        
        # Retrieve Quant node
        quant_node = model.find_producer(input_name)

        if not check_act_quant(model, quant_node, reasons):
            supported = False
            continue
        
        # Get scale and zero_point initializers
        scale_name = quant_node.input[1]
        zeropt_name = quant_node.input[2]
        scale = numpy_helper.to_array(get_by_name(graph.initializer, scale_name))
        zeropt = numpy_helper.to_array(get_by_name(graph.initializer, zeropt_name))
        scales.append(scale)
        zeropts.append(zeropt)

    # Check if all scales and zero points are the same
    if any(not np.array_equal(s, scales[0]) for s in scales) or any(not np.array_equal(z, zeropts[0]) for z in zeropts):
        reasons.append(f"Not all inputs have the same quantization parameters")
        supported = False

    return supported

def is_fpga_supported_op(model: ModelWrapper, node: onnx.NodeProto) -> bool:
    """ Check if the operation is supported by NN2FPGA. """

    reasons = []
    is_supported = True
    if node.op_type not in FPGA_SUPPORTED_OPS:
        reasons.append(f"Not supported operation")
        is_supported = False
    
    # Per operation checks
    elif node.op_type == "Conv":
        # if node.name not in [f"Conv_{i}" for i in range(30)]:
        #     is_supported = False

        # Check Conv activation quantization
        act_quant = model.find_producer(node.input[0])
        is_supported = is_supported and check_act_quant(model, act_quant, reasons)

        # Check Conv weight quantization
        weight_quant = model.find_producer(node.input[1])
        is_supported = is_supported and check_params_quant(model, weight_quant, reasons)

        # Check Conv bias quantization
        if len(node.input) > 2:
            bias_quant = model.find_producer(node.input[2])
            is_supported = is_supported and check_params_quant(
                model, bias_quant, reasons
            )

        # Check Conv attributes
        # Only supported 2D convolutions with square kernels
        kernel_shape = get_by_name(node.attribute, "kernel_shape")
        if (
            kernel_shape is None
            or len(kernel_shape.ints) != 2
            or not (kernel_shape.ints[0] == kernel_shape.ints[1])
        ):
            reasons.append(f"Kernel shape must be a 2D tensor with equal values")
            is_supported = False

        # Only supported depthwise convolutions with group size equal to input channels
        group = get_by_name(node.attribute, "group")
        if group is None or (group.i != 1 and group.i != model.get_tensor_shape(node.input[0])[1]):
            reasons.append(f"Group must be 1 or equal to input channels")
            is_supported = False

        # Only supported Conv without dilations
        dilations = get_by_name(node.attribute, "dilations")
        if dilations is not None and any(d != 1 for d in dilations.ints):
            reasons.append(f"Dilations must have all values equal to 1")
            is_supported = False

        # Only supported Conv with equal strides on both dimensions
        strides = get_by_name(node.attribute, "strides")
        if strides is not None:
            strides = list(strides.ints)
            strides = [1] * (3 - len(strides)) + strides  # Ensure strides has 3 elements
            if strides[1] != strides[2]:
                reasons.append(f"Strides must have equal H, W values.")
                is_supported = False
            if strides[0] != 1:
                reasons.append(f"Strides over channels is not supported.")
                is_supported = False

    elif node.op_type == "Gemm":
        # Right now, Gemm is only supported as a fully connected layer with quantized inputs and weights.

        # Check Gemm activation quantization
        act_quant = model.find_producer(node.input[0])
        is_supported = is_supported and check_act_quant(model, act_quant, reasons)

        # Check Gemm weight quantization
        weight_quant = model.find_producer(node.input[1])
        is_supported = is_supported and check_params_quant(model, weight_quant, reasons)

        # Check Gemm bias quantization
        if len(node.input) > 2:
            bias_quant = model.find_producer(node.input[2])
            is_supported = is_supported and check_params_quant(
                model, bias_quant, reasons
            )

        # Check Gemm attributes
        if not check_attribute(node, "alpha", 1.0, reasons):
            is_supported = False
        if not check_attribute(node, "beta", 1.0, reasons):
            is_supported = False
        if not check_attribute(node, "transA", 0, reasons, optional=True):
            is_supported = False
        if not check_attribute(node, "transB", 1, reasons):
            is_supported = False

        # Check input tensor shape to be a 2D tensor
        input_tensor = model.get_tensor_shape(node.input[0])
        if len(input_tensor) != 2 and not (
            len(input_tensor) == 4 and input_tensor[2:] == [1, 1]
        ):
            reasons.append(f"Unsupported input tensor shape")
            is_supported = False

    elif node.op_type == "Resize":
        # Right now, Resize is only supported as an upsampling operation.

        # Check Resize activation quantization
        act_quant = model.find_producer(node.input[0])
        is_supported = is_supported and check_act_quant(model, act_quant, reasons)

        # Check Resize attributes
        if not check_attribute(
            node, "coordinate_transformation_mode", "asymmetric", reasons
        ):
            is_supported = False
        if not check_attribute(node, "mode", "nearest", reasons):
            is_supported = False

        # Roi is not supported in Resize
        roi, scales = node.input[1:3]
        if roi != "":
            is_supported = False

        # Scales must be present and of form [1.0, 1.0, s, s]
        if scales == "":
            is_supported = False
        else:
            scales_init = numpy_helper.to_array(
                get_by_name(model.graph.initializer, scales)
            )

            if (
                scales_init.shape[0] != 4
                or not np.allclose(scales_init[0:2], [1.0, 1.0])
                or not np.isclose(scales_init[2], scales_init[3])
            ):
                is_supported = False

    elif node.op_type in [
        "MaxPool",
        "AveragePool",
    ]:

        # Check Pool activation quantization
        act_quant = model.find_producer(node.input[0])
        is_supported = is_supported and check_act_quant(model, act_quant, reasons)

        # Only supported Pool without dilations
        dilations = get_by_name(node.attribute, "dilations")
        if dilations is not None and any(d != 1 for d in dilations.ints):
            reasons.append(f"Dilations must have all values equal to 1")
            is_supported = False
        
        # Only supported Pool with equal strides on both dimensions
        strides = get_by_name(node.attribute, "strides")
        if strides is not None:
            strides = list(strides.ints)
            strides = [1] * (3 - len(strides)) + strides  # Ensure strides has 3 elements
            if strides[1] != strides[2]:
                reasons.append(f"Strides must have equal H, W values.")
                is_supported = False
            if strides[0] != 1:
                reasons.append(f"Strides over channels is not supported.")
                is_supported = False
    
    elif node.op_type in [
        "GlobalMaxPool",
        "GlobalAveragePool",
    ]:

        # Check Pool activation quantization
        act_quant = model.find_producer(node.input[0])
        is_supported = is_supported and check_act_quant(model, act_quant, reasons)

    elif node.op_type in "Add":

        # Check if all inputs have the same quantization
        if not check_input_same_quant(model, node, reasons):
            is_supported = False

    elif node.op_type == "Concat":
        # Check if all inputs have the same quantization
        if not check_input_same_quant(model, node, reasons):
            is_supported = False

        # Check Concat attributes
        if not check_attribute(node, "axis", 1, reasons):
            is_supported = False

    elif node.op_type in ["Reshape", "Flatten"]:

        input_shape = model.get_tensor_shape(node.input[0])
        if input_shape is None or len(input_shape) != 4:
            reasons.append(f"Input must be a 4D tensor")
            is_supported = False

        # Check last two dimensions of the input tensor
        if input_shape[-2] != 1 or input_shape[-1] != 1:
            reasons.append(f"Reshape/Flatten only supported with last two dimensions as 1")
            is_supported = False

        # Check output shape to be a identical to input shape
        output_shape = model.get_tensor_shape(node.output[0])
        if output_shape[0:2] != input_shape[0:2]:
            reasons.append(f"Output shape must be identical to input shape")
            is_supported = False

    elif node.op_type in FPGA_SUPPORTED_QUANTIZED_ACTIVATIONS:

        # Check activation quantization
        act_quant = model.find_producer(node.input[0])
        is_supported = is_supported and check_act_quant(model, act_quant, reasons)
    
    elif node.op_type == "Relu":
        # Check activation quantization
        act_quant = model.find_producer(node.input[0])

        # Either is quantized or can be fused in an operator.
        if not check_act_quant(model, act_quant, reasons):
            producer = act_quant
            if producer.op_type not in ["Conv", "Gemm"]:
                reasons.append(f"Relu activation must be quantized or fused into a preceding operator like Conv or Gemm")
                is_supported = False

    elif node.op_type in ["IntQuant", "Quant"]:
        # In case of multiple Quant nodes, only checking the Quant nodes attached to some
        # other operation is not sufficient. We need to check all activation Quant nodes. 
        if not is_constant_input_node(model, node):
            is_supported = is_supported and check_act_quant(model, node, reasons)
    
    if not is_supported:
        logger.info(
            f"Node {node.name} of type {node.op_type} is not supported for FPGA execution. Reasons: {', '.join(reasons)}"
        )
    return is_supported

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
    
    def __find_largest_connected_component(self, model: ModelWrapper) -> list:
        graph = model.graph

        # Label FPGA-supported nodes
        fpga_nodes = set()
        for node in graph.node:
            if is_fpga_supported_op(model, node):
                fpga_nodes.add(node.name)

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
        return largest_component

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        logger.info("\nPartitioning model into FPGA and CPU partitions.")
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

        # Load the FPGA partition model
        FPGA_model = ModelWrapper(self.partition_directory + "/partition_FPGA.onnx")
        FPGA_model = FPGA_model.transform(PostProcessPartitionModel())

        # Assign as a metadata attribute of the partitioned model the name of the node which it corresponds to.
        partition_nodes = parent_model.get_nodes_by_op_type("nn2fpgaPartition")
        if len(partition_nodes) != 1:
            raise ValueError("Extracting more than one HW partition is not supported.")

        logger.info(f"Out of {len(graph.node)} nodes, {len(partition_dict['FPGA'])} nodes are assigned to FPGA partition.")
        return (FPGA_model, False)
