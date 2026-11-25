from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.core.tensor_layout import TensorLayout, set_custom_tensor_layout
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, set_custom_tensor_datatype


def _make_axitostream_model(only_once):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 4])
    node = helper.make_node(
        "AXIToStream",
        ["input"],
        ["output"],
        domain="nn2fpga.compiler.custom_op",
        name="axi_to_stream",
        interface_type="m_axi",
        only_once=only_once,
    )
    model = ModelWrapper(
        qonnx_make_model(helper.make_graph([node], "axitostream_graph", [input_info], [output_info]))
    )
    tensor_type = QuantizedTensorType(1.0, 0, 8, False, False, "ROUND")
    set_custom_tensor_datatype(model, "input", tensor_type)
    set_custom_tensor_datatype(model, "output", tensor_type)
    layout = TensorLayout.identity(4)
    set_custom_tensor_layout(model, "input", layout)
    set_custom_tensor_layout(model, "output", layout)
    return model


def test_object_declaration_passes_only_once_template_parameter():
    model = _make_axitostream_model(1)
    op = getCustomOp(model.get_nodes_by_op_type("AXIToStream")[0])

    declaration = op._AXIToStream__get_object_declaration(model)

    assert "true  // ONLY_ONCE" in declaration


def test_object_declaration_defaults_to_repeated_execution():
    model = _make_axitostream_model(0)
    op = getCustomOp(model.get_nodes_by_op_type("AXIToStream")[0])

    declaration = op._AXIToStream__get_object_declaration(model)

    assert "false  // ONLY_ONCE" in declaration
