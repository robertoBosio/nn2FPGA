from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
from onnx import helper, NodeProto
import backend.transformation as transformation
from backend.core.tensor_quant import TensorQuant
import copy
import logging
logger = logging.getLogger(__name__)

QUANT_INVARIANT_NODES = [
    "BandwidthAdjustDecreaseChannels",  # nn2FPGA
    "BandwidthAdjustDecreaseStreams",  # nn2FPGA
    "BandwidthAdjustIncreaseChannels",  # nn2FPGA
    "BandwidthAdjustIncreaseStreams",  # nn2FPGA
    "Concat",
    "Flatten",
    "GlobalMaxPool",
    "Identity",
    "MaxPool",
    "NHWCToStream",  # nn2FPGA
    "Pad",
    "Reshape",
    "Split",
    "StreamingLineBuffer",  # nn2FPGA
    "StreamToNHWC",  # nn2FPGA
    "TensorDuplicator",  # nn2FPGA
    "Transpose",
]

def get_non_constant_inputs(node: NodeProto, model: ModelWrapper) -> list[str]:
    """Get non-constant inputs of a node."""
    init_dict = [init.name for init in model.model.graph.initializer]
    not_constant_inputs = []
    for inp in node.input:
        if inp in init_dict:
            continue
        node = model.find_producer(inp)
        if node is not None and node.op_type == "Constant":
            continue
        not_constant_inputs.append(inp)
    return not_constant_inputs

def get_non_constant_outputs(node: NodeProto, model: ModelWrapper) -> list[str]:
    """Get non-constant outputs of a node."""
    init_dict = [init.name for init in model.model.graph.initializer]
    not_constant_outputs = []
    for out in node.output:
        if out in init_dict:
            continue
        node = model.find_producer(out)
        if node is not None and node.op_type == "Constant":
            continue
        not_constant_outputs.append(out)
    return not_constant_outputs

def forward_propagate_quantization(
    producers: list[NodeProto],
    consumers: list[NodeProto],
    node: NodeProto,
    model: ModelWrapper,
) -> list[NodeProto] | None:
    """
    Try to propagate quantization forward from producer nodes to consumer nodes.
    This function checks if all producer nodes are quantized with the same parameters and if none of the consumer nodes are quantized.
    If these conditions are met, it inserts new quantization nodes after the given node, rewires the consumers,
    and returns the list of newly created quantization nodes.
    Args:
        producers (list[NodeProto]): List of producer nodes that provide inputs to the current node.
        consumers (list[NodeProto]): List of consumer nodes that use outputs from the current node.
        node (NodeProto): The current node in the graph where quantization propagation is considered.
        model (ModelWrapper): The model wrapper containing the graph and related utilities.
    Returns:
        list[NodeProto] | None: List of newly created quantization nodes if propagation is possible, otherwise None.
    """

    # The behavior when there is a producer which is None (i.e. an input) should
    # be defined in the future, but for now we avoid it.
    if len(producers) == 0 or any(p is None for p in producers):
        return None

    # To propagate forward a quantization, all the inputs must be quantized
    # and with the same quantization parameters, while the consumers must not be quantized
    if not all(
        p.op_type in ["IntQuant", "Quant"] for p in producers
    ):
        # If not all producers are quantized, we cannot propagate quantization
        return None

    quant_node = producers[0]
    reference_quant = TensorQuant.from_quant_node(quant_node, model)
    if not all(
        TensorQuant.from_quant_node(p, model) == reference_quant
        for p in producers
    ):
        # If producers are not quantized with the same parameters, we cannot propagate quantization
        return None

    if any(
        c.op_type in ["IntQuant", "Quant"] for c in consumers if c is not None
    ):
        # If any consumers are quantized, we cannot propagate quantization
        return None

    # If the producer is quantized and consumers are not, propagate quantization
    added_quant_nodes = []
    for out in get_non_constant_outputs(node, model):
        logger.info(
            f"Propagating forward quantization of {quant_node.name} to tensor {out}"
        )

        # Insert new quant node after current node
        new_output = out + "_quant_forward_propagated"
        new_quant_node = helper.make_node(
            quant_node.op_type,
            inputs=[
                new_output,
                quant_node.input[1],
                quant_node.input[2],
                quant_node.input[3],
            ],  # Use the same quantization parameters
            outputs=[out],
            name=quant_node.name + "_forward_propagated",
            domain=quant_node.domain,
        )
        for attr in quant_node.attribute:
            new_quant_node.attribute.append(copy.deepcopy(attr))

        # Rewire the consumer to use the new quantized output
        for j, node_output in enumerate(node.output):
            if node_output == out:
                node.output[j] = new_output

        # Append the new quant node created
        added_quant_nodes.append(new_quant_node)

    return added_quant_nodes

def backward_propagate_quantization(
    consumers: list[NodeProto],
    producers: list[NodeProto],
    node: NodeProto,
    model: ModelWrapper,
) -> list[NodeProto] | None:
    """
    Try to propagate quantization backward from consumer nodes to producer nodes.
    This function checks if all consumer nodes are quantized with the same parameters and if none of the producer nodes are quantized.
    If these conditions are met, it inserts new quantization nodes before the given node, rewires the producers,
    and returns the list of newly created quantization nodes.
    Args:
        consumers (list[NodeProto]): List of consumer nodes that use outputs from the current node.
        producers (list[NodeProto]): List of producer nodes that provide inputs to the current node.
        node (NodeProto): The current node in the graph where quantization propagation is considered.
        model (ModelWrapper): The model wrapper containing the graph and related utilities.
    Returns:
        list[NodeProto] | None: List of newly created quantization nodes if propagation is possible, otherwise None.
    """

    if len(consumers) == 0 or any(c is None for c in consumers):
        return None

    # To propagate backward a quantization, all the outputs must be quantized
    # and with the same quantization parameters, while the producers must not be quantized
    if not all(
        c.op_type in ["IntQuant", "Quant"] for c in consumers
    ):
        # If not all consumers are quantized, we cannot propagate quantization
        return None

    quant_node = consumers[0]
    reference_quant = TensorQuant.from_quant_node(quant_node, model)
    if not all(
        TensorQuant.from_quant_node(c, model) == reference_quant
        for c in consumers
    ):
        # If consumers are not quantized with the same parameters, we cannot propagate quantization
        return None

    if any(
        p.op_type in ["IntQuant", "Quant"] for p in producers if p is not None
    ):
        # If any producers are quantized, we cannot propagate quantization
        return None

    # If the consumers are quantized and producers are not, propagate quantization
    added_quant_nodes = []
    for inp in get_non_constant_inputs(node, model):
        logger.info(
            f"Propagating backward quantization of {quant_node.name} to tensor {inp}"
        )

        # Insert new quant node before current node
        new_quant_output = inp + "_quant_backward_propagated"
        new_quant_node = helper.make_node(
            quant_node.op_type,
            inputs=[
                inp,
                quant_node.input[1],
                quant_node.input[2],
                quant_node.input[3],
            ],  # Use the same quantization parameters
            outputs=[new_quant_output],
            name=quant_node.name + "_backward_propagated",
            domain=quant_node.domain,
        )
        for attr in quant_node.attribute:
            new_quant_node.attribute.append(copy.deepcopy(attr))

        # Rewire the node to use the new quantized inputs
        for j, node_input in enumerate(node.input):
            if node_input == inp:
                node.input[j] = new_quant_output

        # Append the new quant node created
        added_quant_nodes.append(new_quant_node)

    return added_quant_nodes

class PropagateQuant(Transformation):
    """Propagate quantization parameters through quantization invariant nodes.

    This transformation analyzes nodes that are quantization invariant and propagates quantization
    parameters either forward (from producers to consumers) or backward (from consumers to producers)
    through the graph. It inserts new Quant nodes where appropriate, inferring the quantization parameters.

    Forward propagation occurs when all input tensors to a quantization invariant node are quantized with identical parameters,
    and none of the output tensors are quantized. In this case, quantization is propagated to the outputs.

    Backward propagation occurs when all output tensors from a node are quantized with identical parameters,
    and none of the input tensors are quantized. In this case, quantization is propagated to the inputs.

    After any changes, the graph is re-sorted and additional transformations are applied to maintain
    graph integrity, infer shapes, and ensure unique and readable tensor and node names.

    Args:
        model (ModelWrapper): The model wrapper containing the computational graph.

    Returns:
        tuple[ModelWrapper, bool]: The transformed model and a boolean indicating if the transformation should be run again.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        graph = model.graph
        is_changed = False
        new_quant_nodes = []

        # Iterate through all quantization invariant nodes
        for node in list(filter(lambda n: n.op_type in QUANT_INVARIANT_NODES, graph.node)):

            # Retrieve all the producers of the inputs of the current node
            # removing Quant nodes that have all constat inputs (i.e., weights)
            producers = [
                model.find_producer(inp) for inp in get_non_constant_inputs(node, model)
            ]
            
            # Retrieve all the consumers of the outputs of the current node
            # removing Quant nodes that have all constant inputs (i.e., weights)
            # Consumers of a tensor can be multiple nodes, so we collect them in a list
            consumers = list()
            for out in get_non_constant_outputs(node, model):
                consumers.extend(model.find_consumers(out))

            quant_forward_nodes = forward_propagate_quantization(
                producers, consumers, node, model
            )
            if quant_forward_nodes is not None and len(quant_forward_nodes) > 0:
                new_quant_nodes.extend(quant_forward_nodes)
            
            quant_backward_nodes = backward_propagate_quantization(
                consumers, producers, node, model
            )
            if quant_backward_nodes is not None and len(quant_backward_nodes) > 0:
                new_quant_nodes.extend(quant_backward_nodes)


        if new_quant_nodes:
            # Insert all new quant nodes into the graph
            graph.node.extend(new_quant_nodes)
            is_changed = True

            # If any changes were made, sort the graph to maintain a valid topological order
            model = model.transform(SortGraph())
        else:
            # If no changes were made, just ensure the model is sorted and has unique names
            model = model.transform(SortGraph())
            model = model.transform(transformation.CustomInferShapes())
            model = model.transform(GiveUniqueParameterTensors())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())

        return (model, is_changed)
