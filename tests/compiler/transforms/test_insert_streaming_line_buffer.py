from onnx import TensorProto, helper
from qonnx.util.basic import get_by_name
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, set_custom_tensor_datatype
from nn2fpga.compiler.transforms.insert_streaming_line_buffer import InsertStreamingLineBuffer


def _build_streaming_conv_model(*, asym_folding: int, zeropt: int) -> ModelWrapper:
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 4, 4])

    conv_node = helper.make_node(
        "StreamingConv",
        inputs=[
            "input",
            "weights",
            "weight_scale",
            "weight_zero_point",
            "weight_bitwidth",
            "bias",
            "bias_scale",
            "bias_zero_point",
            "bias_bitwidth",
        ],
        outputs=["output"],
        name="conv0",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dilations=[1, 1],
        group=1,
        asym_folding=asym_folding,
        w_signed=1,
        b_signed=1,
    )

    graph = helper.make_graph(
        [conv_node],
        "insert_streaming_line_buffer_graph",
        [graph_input],
        [graph_output],
        initializer=[
            helper.make_tensor("weights", TensorProto.FLOAT, [4, 3, 3, 3], [1.0] * (4 * 3 * 3 * 3)),
            helper.make_tensor("weight_scale", TensorProto.FLOAT, [1], [0.25]),
            helper.make_tensor("weight_zero_point", TensorProto.INT8, [1], [0]),
            helper.make_tensor("weight_bitwidth", TensorProto.INT32, [1], [8]),
            helper.make_tensor("bias", TensorProto.FLOAT, [4], [0.0, 0.0, 0.0, 0.0]),
            helper.make_tensor("bias_scale", TensorProto.FLOAT, [1], [0.125]),
            helper.make_tensor("bias_zero_point", TensorProto.INT32, [1], [0]),
            helper.make_tensor("bias_bitwidth", TensorProto.INT32, [1], [32]),
        ],
    )

    model = ModelWrapper(qonnx_make_model(graph, producer_name="test_insert_streaming_line_buffer"))
    set_custom_tensor_datatype(
        model,
        "input",
        QuantizedTensorType(
            bitwidth=8,
            signed=1,
            scale=0.5,
            zeropt=zeropt,
            narrow=0,
            rounding_mode="ROUND",
        ),
    )
    return model


def test_insert_streaming_line_buffer_uses_folded_input_zero_point_as_padding():
    model = _build_streaming_conv_model(asym_folding=1, zeropt=3)

    transformed_model, changed = InsertStreamingLineBuffer().apply(model)

    assert changed is False
    linebuffer_node = transformed_model.get_nodes_by_op_type("StreamingCircularLineBuffer")[0]
    assert linebuffer_node.name == "conv0_streaming_linebuffer"
    assert get_by_name(linebuffer_node.attribute, "pad_value").f == 3.0


def test_insert_streaming_line_buffer_keeps_zero_padding_without_folding():
    model = _build_streaming_conv_model(asym_folding=0, zeropt=3)

    transformed_model, changed = InsertStreamingLineBuffer().apply(model)

    assert changed is False
    linebuffer_node = transformed_model.get_nodes_by_op_type("StreamingCircularLineBuffer")[0]
    assert linebuffer_node.name == "conv0_streaming_linebuffer"
    assert get_by_name(linebuffer_node.attribute, "pad_value").f == 0.0
