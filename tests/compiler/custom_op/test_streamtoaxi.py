import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.core.tensor_layout import TensorLayout, set_custom_tensor_layout
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, set_custom_tensor_datatype


def _make_streamtoaxi_model(interface_type="s_axis"):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 4])
    node = helper.make_node(
        "StreamToAXI",
        ["input"],
        ["output"],
        domain="nn2fpga.compiler.custom_op",
        name="stream_to_axi",
        interface_type=interface_type,
    )
    model = ModelWrapper(
        qonnx_make_model(helper.make_graph([node], "streamtoaxi_graph", [input_info], [output_info]))
    )
    tensor_type = QuantizedTensorType(8, False, 1.0, 0, False, "ROUND")
    set_custom_tensor_datatype(model, "input", tensor_type)
    set_custom_tensor_datatype(model, "output", tensor_type)
    layout = TensorLayout.identity(4)
    set_custom_tensor_layout(model, "input", layout)
    set_custom_tensor_layout(model, "output", layout)
    return model


def test_catapult_lowering_uses_axis_and_ac_types_without_step_call():
    model = _make_streamtoaxi_model()
    model.set_metadata_prop("hls_backend", "catapult")

    op = getCustomOp(model.get_nodes_by_op_type("StreamToAXI")[0])
    nodes, _, _, _ = op.lower_to_hls(model, 0)
    hls_kernel = getCustomOp(nodes[0])

    declaration = hls_kernel.get_nodeattr("hls_object_declaration")
    assert "StreamToAXIS" in declaration
    assert "AxisWord<128>" in declaration
    assert "std::array<ac_int<8, false>, 1>" in declaration
    assert "ac_int<8, false>,  // TInput" in declaration
    assert hls_kernel.get_nodeattr("hls_step_call") == ""


def test_catapult_rejects_maxi_during_lowering_and_object_declaration():
    model = _make_streamtoaxi_model(interface_type="m_axi")
    model.set_metadata_prop("hls_backend", "catapult")
    op = getCustomOp(model.get_nodes_by_op_type("StreamToAXI")[0])

    with pytest.raises(ValueError, match="only supports interface_type='s_axis'"):
        op.lower_to_hls(model, 0)
    with pytest.raises(ValueError, match="only supports interface_type='s_axis'"):
        op._StreamToAXI__get_object_declaration(model)
