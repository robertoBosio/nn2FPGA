from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.custom_op.op_base import NodeInterface


def test_inherit_interface_updates_window_to_channels_packing():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 12, 4, 4])
    node = helper.make_node(
        "StreamingWindowToChannels",
        inputs=["input"],
        outputs=["output"],
        domain="nn2fpga.compiler.custom_op",
        name="focus",
        kernel_shape=[2, 2],
        pads=[0, 0],
        strides=[2, 2],
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([node], "focus_graph", [input_tensor], [output_tensor]),
            producer_name="test",
        )
    )

    op = getCustomOp(node)
    op.inherit_interface(
        model,
        NodeInterface(
            in_stream_array=1,
            out_stream_array=2,
            in_word_array=1,
            out_word_array=3,
        ),
    )

    assert op.can_inherit_interface()
    assert op.get_nodeattr("in_stream_array") == 2
    assert op.get_nodeattr("out_stream_array") == 2
    assert op.get_nodeattr("in_word_array") == 3
    assert op.get_nodeattr("out_word_array") == 3
    assert op.get_nodeattr("dim1_unroll") == 2
    assert op.get_nodeattr("dim2_unroll") == 3
