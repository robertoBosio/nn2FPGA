from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.transforms.custom_infershape import CustomInferShapes
from qonnx.util.basic import remove_by_name
from onnx import helper, NodeProto
import logging
logger = logging.getLogger(__name__)


def remove_flatten_reshape(model: ModelWrapper, node: NodeProto) -> None:
    """
    Removes Flatten or Reshape nodes from the model, as they are not needed
    after converting fully connected layers to convolutional layers.

    Args:
        model (ModelWrapper): The ONNX model wrapper.
        node (onnx.NodeProto): The Flatten or Reshape node to be removed.
    """
    graph = model.graph

    # If the input is not produced by any node, it's a model input.
    consumers = model.find_consumers(node.output[0])
    for consumer in consumers:
        # Replace the input with the Flatten/Reshape node's input
        for i, consumer_input in enumerate(consumer.input):
            if consumer_input == node.output[0]:
                consumer.input[i] = node.input[0]

    # Remove the Flatten/Reshape node from the graph
    graph.node.remove(node)


def remove_all_shape_info(model: ModelWrapper) -> None:
    """
    Removes all inferrable shape/type information from the model's
    graph.output and and graph.value_info to enable re-inference.

    Args:
        model (ModelWrapper): The ONNX model wrapper.
    """
    for tensor in model.get_all_tensor_names():
        if tensor not in [init.name for init in model.graph.initializer]:
            remove_by_name(model.graph.value_info, tensor)

    new_outputs = []
    for out in model.graph.output:
        if not out.type.HasField("tensor_type"):
            new_outputs.append(out)
            continue

        tensor_type = out.type.tensor_type
        elem_type = tensor_type.elem_type

        # Create output without shape
        new_vi = helper.make_tensor_value_info(
            name=out.name,
            elem_type=elem_type,
            shape=None  # No shape -> makes it completely unspecified
        )
        new_outputs.append(new_vi)

    # Replace graph outputs with shape-less versions
    del model.graph.output[:]
    model.graph.output.extend(new_outputs)

class RemoveSqueeze(Transformation):
    """Remove all the operations that act as a squeeze operation in the model, 
    since in HW, when configured in streaming mode, it does not change anything."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        nodes_to_remove = []
        for node in model.graph.node:
            if node.op_type == "Flatten" or node.op_type == "Squeeze" or node.op_type == "Reshape":
                input_shape = model.get_tensor_shape(node.input[0])
                output_shape = model.get_tensor_shape(node.output[0])
                if input_shape is not None and output_shape is not None and len(input_shape) > len(output_shape):
                    
                    # Check that all the missing dimensions are 1
                    for i in range(len(output_shape), len(input_shape)):
                        if input_shape[i] != 1:
                            break
                    else:
                        nodes_to_remove.append(node)

        for node in nodes_to_remove:
            remove_flatten_reshape(model, node)
            logger.info(f"Removed node {node.name} of type {node.op_type} as it acts as a squeeze operation.")

        # Remove all shape information to enable re-inference
        # remove_all_shape_info(model)

        # Re-infer shapes after removing squeeze operations
        # model = model.transform(CustomInferShapes())

        return model, False
