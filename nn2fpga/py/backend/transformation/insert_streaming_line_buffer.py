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
from qonnx.custom_op.registry import getCustomOp

logger = logging.getLogger(__name__)

class InsertStreamingLineBuffer(Transformation):
    """
    Inserts a StreamingLineBuffer node to create the windows in input to compute intensive nodes.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        new_nodes = []
        for node in model.graph.node:
            nn2fpga_node = getCustomOp(node)
            if not nn2fpga_node.has_linebuffer():
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

            iface = nn2fpga_node.get_port_interface()

            pad_value = 0
            if node.op_type == "StreamingMaxPool":
                pad_value = float('-inf')

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
                in_word_array=iface.in_word_array,
                in_stream_array=iface.in_stream_array,
                out_word_array=iface.in_word_array,
                out_stream_array=iface.in_stream_array,
                channel_unroll=iface.in_word_array,
                width_unroll=iface.in_stream_array,
                pad_value=pad_value,
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
