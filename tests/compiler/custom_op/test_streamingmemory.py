from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.core.tensor_layout import TensorLayout, set_custom_tensor_layout
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, set_custom_tensor_datatype


def _get_streaming_memory(storage_impl: str, data_to_shift: int):
    node = helper.make_node(
        "StreamingMemory",
        ["input"],
        ["output", "shift_output"] if data_to_shift else ["output"],
        domain="nn2fpga.compiler.custom_op",
        name="streaming_memory",
        storage_impl=storage_impl,
        data_to_shift=data_to_shift,
        data_per_word=1,
        times=1,
    )
    graph = helper.make_graph(
        [node],
        "streaming_memory_graph",
        [helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 1, 1, 64])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 1, 1, 64])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    tensor_type = QuantizedTensorType(1.0, 0, 8, False, False, "ROUND")
    for tensor_name in ("input", "output"):
        set_custom_tensor_datatype(model, tensor_name, tensor_type)
        set_custom_tensor_layout(model, tensor_name, TensorLayout.identity(4))
    return model, getCustomOp(model.graph.node[0])


def test_run_call_keeps_the_same_entry_point_when_uram_is_selected():
    _, op = _get_streaming_memory("uram", data_to_shift=0)

    run_call = op._StreamingMemory__get_run_call(0)
    assert "streaming_memory.run<" in run_call
    assert "run_uram" not in run_call


def test_run_call_uses_the_same_entry_point_when_not_selected():
    _, op = _get_streaming_memory("auto", data_to_shift=1)

    run_call = op._StreamingMemory__get_run_call(0)
    assert "streaming_memory.run<" in run_call
    assert "run_uram" not in run_call


def test_object_declaration_passes_uram_template_argument():
    model, op = _get_streaming_memory("uram", data_to_shift=0)

    declaration = op._StreamingMemory__get_object_declaration(model)

    assert "true  // USE_URAM" in declaration
