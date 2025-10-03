from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from onnx import helper
from backend.util.par_utils import get_par_attributes
from backend.core.tensor_quant import (
    get_custom_tensor_datatype,
    set_custom_tensor_datatype,
)
import backend.transformation as transformation
import numpy as np
import logging

logger = logging.getLogger(__name__)

LAYERS_WITH_KERNELS = [
    "StreamingConv",
    "MaxPool",
    "AveragePool",
    "ConvTranspose",
    "StreamingDepthwiseConv",
]

def has_streaming_linebuffer(op_type: str, kernel: int, w_par: int) -> bool:
    """
    Check if the node needs a StreamingLineBuffer.
    This is determined by a kernel bigger than 1x1 or
    a parallelization over the width dimension.
    """
    if op_type not in LAYERS_WITH_KERNELS:
        return False

    # Check if the kernel size is greater than 1x1
    if kernel > 1:
        return True

    # Check if there is parallelization over the width dimension
    if w_par > 1:
        return True

    return False

def has_streaming_linebuffer_wrap(model: ModelWrapper, node: helper.NodeProto) -> bool:
    """
    Extracts the kernel shape and width parallelization from the node
    and checks if a StreamingLineBuffer is needed.
    """
    if node.op_type not in LAYERS_WITH_KERNELS:
        return False
    
    kernel_shape = get_by_name(node.attribute, "kernel_shape").ints
    kernel = int(np.prod(kernel_shape))

    w_par = 1  # Default value for width parallelization
    node_par = get_par_attributes(node)
    if node_par["in_w_par"] is not None and node_par["in_w_par"] > 1:
        w_par = node_par["in_w_par"]

    return has_streaming_linebuffer(node.op_type, kernel, w_par)

class InsertStreamingLineBuffer(Transformation):
    """
    Inserts a StreamingLineBuffer node to create the windows in input to compute intensive nodes.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        new_nodes = []
        for node in model.graph.node:
            if node.op_type not in LAYERS_WITH_KERNELS:
                continue

            if not has_streaming_linebuffer_wrap(model, node):
                continue

            # Retrieve the necessary attributes from the node
            pads = get_by_name(node.attribute, "pads")
            if pads is None:
                pads = [0, 0, 0, 0]
            else:
                pads = pads.ints

            kernel_shape = get_by_name(node.attribute, "kernel_shape")
            if kernel_shape is None:
                input_shape = model.get_tensor_shape(node.input[0])
                input_shape = [1] * (4 - len(input_shape)) + input_shape
                kernel_shape = input_shape[2:4]
            else:
                kernel_shape = kernel_shape.ints

            par = get_par_attributes(node)

            dilation = get_by_name(node.attribute, "dilations")
            if dilation is None:
                dilation = [1, 1]
            else:
                dilation = dilation.ints

            stride = get_by_name(node.attribute, "strides")
            if stride is None:
                stride = [1, 1]
            else:
                stride = stride.ints

            # Create the StreamingLineBuffer node
            streaming_line_buffer_node = helper.make_node(
                op_type="StreamingLineBuffer",
                domain="backend.custom_op",
                inputs=[node.input[0]],
                outputs=[f"{node.name}_window"],
                pads=pads,
                kernel_shape=kernel_shape,
                dilation=dilation,
                strides=stride,
                in_w_par=par["in_w_par"],
                in_ch_par=par["in_ch_par"],
                out_w_par=par["in_w_par"],
                out_ch_par=par["in_ch_par"],
                name=f"{node.name}_streaming_linebuffer",
            )

            # Replace the node's input with the output of the StreamingLineBuffer
            node.input[0] = streaming_line_buffer_node.output[0]

            # Add the StreamingLineBuffer node to the model
            new_nodes.append(streaming_line_buffer_node)

        if len(new_nodes) > 0:
            for new_node in new_nodes:
                model.graph.node.append(new_node)
            model = model.transform(SortGraph())
            model = model.transform(transformation.CustomInferShapes())

        return (model, False)
