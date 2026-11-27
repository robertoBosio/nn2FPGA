from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.core.tensor_layout import TensorLayout, set_custom_tensor_layout
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, set_custom_tensor_datatype
from nn2fpga.compiler.utils.codegen_utils import HLSCodegen


def _build_model() -> ModelWrapper:
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 1, 1])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 1, 1])
    conv = helper.make_node(
        "StreamingConv",
        inputs=[
            "input",
            "weights",
            "weight_scale",
            "weight_zero_point",
            "weight_bitwidth",
            "biases",
            "bias_scale",
            "bias_zero_point",
            "bias_bitwidth",
        ],
        outputs=["output"],
        name="conv0",
        domain="nn2fpga.compiler.custom_op",
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
        dilations=[1, 1],
        group=1,
        w_signed=1,
        w_narrow=0,
        w_rounding_mode="ROUND",
        b_signed=1,
        b_narrow=0,
        b_rounding_mode="ROUND",
        param_storage="INTERNAL",
    )
    graph = helper.make_graph(
        [conv],
        "streaming_conv_internal_params",
        [graph_input],
        [graph_output],
        initializer=[
            helper.make_tensor(
                "weights",
                TensorProto.FLOAT,
                [2, 2, 1, 1],
                [0.5, -0.25, 0.0, 0.75],
            ),
            helper.make_tensor("weight_scale", TensorProto.FLOAT, [], [0.25]),
            helper.make_tensor("weight_zero_point", TensorProto.INT8, [], [0]),
            helper.make_tensor("weight_bitwidth", TensorProto.INT32, [], [8]),
            helper.make_tensor("biases", TensorProto.FLOAT, [2], [1.0, -1.5]),
            helper.make_tensor("bias_scale", TensorProto.FLOAT, [], [0.5]),
            helper.make_tensor("bias_zero_point", TensorProto.INT32, [], [0]),
            helper.make_tensor("bias_bitwidth", TensorProto.INT32, [], [32]),
        ],
    )
    model = ModelWrapper(
        qonnx_make_model(graph, producer_name="test_streamingconv_internal_params")
    )
    set_custom_tensor_layout(model, "input", TensorLayout((0, 2, 3, 1)))
    set_custom_tensor_layout(model, "output", TensorLayout((0, 2, 3, 1)))
    tensor_type = QuantizedTensorType(
        scale=1.0, zeropt=0, bitwidth=8, signed=True, narrow=False,
        rounding_mode="ROUND",
    )
    set_custom_tensor_datatype(model, "input", tensor_type)
    set_custom_tensor_datatype(model, "output", tensor_type)
    return model


def test_internal_parameters_are_quantized_and_initialized():
    model = _build_model()
    op = getCustomOp(model.graph.node[0])

    declaration = op._StreamingConv__get_variable_declaration(model)

    assert "ap_int<8> conv0_weights[4][1][1] =" in declaration
    assert "{2}" in declaration
    assert "{-1}" in declaration
    assert "{0}" in declaration
    assert "{3}" in declaration
    assert "ap_int<32> conv0_biases[2][1][1] =" in declaration
    assert "{-3}" in declaration


def test_catapult_external_parameters_use_ac_types_and_no_step_calls():
    model = _build_model()
    op = getCustomOp(model.graph.node[0])
    op.set_nodeattr("param_storage", "EXTERNAL")
    codegen = HLSCodegen("catapult")

    declaration = op._StreamingConv__get_object_declaration(model, codegen)
    variables = op._StreamingConv__get_variable_declaration(model, codegen)

    assert "ac_int<" in declaration
    assert "hls::" not in declaration
    assert variables == ""
    assert op._StreamingConv__get_step_call(codegen) == ""

    model.set_metadata_prop("hls_backend", "catapult")
    hls_nodes, _, fifo, _ = op.lower_to_hls(model, 0)
    hls_op = getCustomOp(hls_nodes[0])
    assert "hls::" not in hls_op.get_nodeattr("hls_object_declaration")
    assert all("ac_int<" in metadata.hls_type for metadata in fifo.values())


def test_catapult_external_parameters_support_fused_relu():
    model = _build_model()
    op = getCustomOp(model.graph.node[0])
    op.set_nodeattr("param_storage", "EXTERNAL")
    op.set_nodeattr("activation", "ReLU")

    declaration = op._StreamingConv__get_object_declaration(
        model, HLSCodegen("catapult")
    )

    assert "ReLUQuantPo2<" in declaration
