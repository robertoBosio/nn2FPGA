from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.analysis.check_quantization import check_input_same_quant
from nn2fpga.compiler.core.tensor_layout import TensorLayout, set_custom_tensor_layout
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, set_custom_tensor_datatype


def _make_streamingadd_model(quant_a, quant_b, quant_out):
    input_a = helper.make_tensor_value_info("inputA", TensorProto.FLOAT, [1, 4, 4, 4])
    input_b = helper.make_tensor_value_info("inputB", TensorProto.FLOAT, [1, 4, 4, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 4, 4])

    node = helper.make_node(
        "StreamingAdd",
        inputs=["inputA", "inputB"],
        outputs=["output"],
        domain="nn2fpga.compiler.custom_op",
        name="add_node",
        dim2_unroll=1,
        dim1_unroll=1,
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=1,
        out_word_array=1,
    )

    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([node], "streamingadd_graph", [input_a, input_b], [output]),
            producer_name="test",
        )
    )

    set_custom_tensor_datatype(model, "inputA", quant_a)
    set_custom_tensor_datatype(model, "inputB", quant_b)
    set_custom_tensor_datatype(model, "output", quant_out)

    identity = TensorLayout.identity(4)
    set_custom_tensor_layout(model, "inputA", identity)
    set_custom_tensor_layout(model, "inputB", identity)
    set_custom_tensor_layout(model, "output", identity)
    return model


def _get_add_op(model):
    return getCustomOp(model.get_nodes_by_op_type("StreamingAdd")[0])


def _build_quantized_add_model(scale_a, zp_a, scale_b, zp_b):
    x0 = helper.make_tensor_value_info("x0", TensorProto.FLOAT, [1])
    x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])

    initializers = [
        helper.make_tensor("scale_a", TensorProto.FLOAT, [], [scale_a]),
        helper.make_tensor("zp_a", TensorProto.INT8, [], [zp_a]),
        helper.make_tensor("bw_a", TensorProto.INT32, [], [8]),
        helper.make_tensor("scale_b", TensorProto.FLOAT, [], [scale_b]),
        helper.make_tensor("zp_b", TensorProto.INT8, [], [zp_b]),
        helper.make_tensor("bw_b", TensorProto.INT32, [], [8]),
    ]
    nodes = [
        helper.make_node("Quant", inputs=["x0", "scale_a", "zp_a", "bw_a"], outputs=["qa"], name="qa"),
        helper.make_node("Quant", inputs=["x1", "scale_b", "zp_b", "bw_b"], outputs=["qb"], name="qb"),
        helper.make_node("Add", inputs=["qa", "qb"], outputs=["y"], name="add"),
    ]
    return ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "quant_add_graph", [x0, x1], [y], initializer=initializers),
            producer_name="test",
        )
    )


def test_object_declaration_supports_po2_asymmetric_add():
    model = _make_streamingadd_model(
        QuantizedTensorType(bitwidth=8, signed=0, scale=2**-4, zeropt=23),
        QuantizedTensorType(bitwidth=8, signed=0, scale=2**-6, zeropt=101),
        QuantizedTensorType(bitwidth=8, signed=0, scale=2**-5, zeropt=77),
    )

    declaration = _get_add_op(model)._StreamingAdd__get_object_declaration(model)

    assert "ap_int<12>" in declaration
    assert "DequantAlignPo2<-2, 23, ap_uint<8>, ap_int<12>>" in declaration
    assert "DequantAlignEqual<101, ap_uint<8>, ap_int<12>>" in declaration
    assert "DequantQuantPo2<1, ap_int<12>, ap_uint<8>, 77>" in declaration


def test_object_declaration_supports_float_asymmetric_add():
    model = _make_streamingadd_model(
        QuantizedTensorType(bitwidth=8, signed=0, scale=0.375, zeropt=91),
        QuantizedTensorType(bitwidth=8, signed=1, scale=0.2, zeropt=-17),
        QuantizedTensorType(bitwidth=8, signed=0, scale=0.3, zeropt=109),
    )

    declaration = _get_add_op(model)._StreamingAdd__get_object_declaration(model)

    assert "ap_int<10>" in declaration
    assert "DequantAlignFloat<15, 8, 91, ap_uint<8>, ap_int<10>>" in declaration
    assert "DequantAlignEqual<-17, ap_int<8>, ap_int<10>>" in declaration
    assert "DequantQuantFloatShift<341, 9, ap_int<10>, ap_uint<8>, 109>" in declaration


def test_object_declaration_supports_float_relu_add():
    model = _make_streamingadd_model(
        QuantizedTensorType(bitwidth=8, signed=0, scale=0.375, zeropt=91),
        QuantizedTensorType(bitwidth=8, signed=1, scale=0.2, zeropt=-17),
        QuantizedTensorType(bitwidth=8, signed=0, scale=0.3, zeropt=109),
    )
    op = _get_add_op(model)
    op.set_nodeattr("activation", "ReLU")

    declaration = op._StreamingAdd__get_object_declaration(model)

    assert "ReLUQuantFloat<2, 3, ap_int<10>, ap_uint<8>, 109>" in declaration


def test_check_input_same_quant_allows_add_with_different_quant_params():
    model = _build_quantized_add_model(0.1, 0, 0.2, 3)
    assert check_input_same_quant(model) is True
