from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.transforms.fuse_elementwise_op import FuseElementwiseOps


def _build_fusable_model(producer_op_type: str, activation_op_type: str):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])

    producer_node = helper.make_node(
        producer_op_type,
        inputs=["input"],
        outputs=["producer_output"],
        name="producer_node",
        domain="nn2fpga.compiler.custom_op",
    )

    activation_kwargs = {}
    if activation_op_type == "StreamingLeakyReLU":
        activation_kwargs["alpha"] = 0.1015625

    activation_node = helper.make_node(
        activation_op_type,
        inputs=["producer_output"],
        outputs=["output"],
        name="activation_node",
        domain="nn2fpga.compiler.custom_op",
        **activation_kwargs,
    )

    graph = helper.make_graph(
        [producer_node, activation_node],
        name="fuse_elementwise_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def test_fuse_streaming_leakyrelu_into_streamingconv():
    model = _build_fusable_model("StreamingConv", "StreamingLeakyReLU")

    model, changed = FuseElementwiseOps().apply(model)

    assert changed is True
    assert len(model.graph.node) == 1
    producer = model.graph.node[0]
    producer_op = getCustomOp(producer)
    assert producer.output[0] == "output"
    assert producer_op.get_nodeattr("activation") == "LeakyReLU"
    assert producer_op.get_nodeattr("activation_alpha_num") == 13
    assert producer_op.get_nodeattr("activation_alpha_den") == 128


def test_fuse_streaming_leakyrelu_into_streamingdepthwiseconv():
    model = _build_fusable_model("StreamingDepthwiseConv", "StreamingLeakyReLU")

    model, changed = FuseElementwiseOps().apply(model)

    assert changed is True
    producer = model.graph.node[0]
    producer_op = getCustomOp(producer)
    assert producer.output[0] == "output"
    assert producer_op.get_nodeattr("activation") == "LeakyReLU"
    assert producer_op.get_nodeattr("activation_alpha_num") == 13
    assert producer_op.get_nodeattr("activation_alpha_den") == 128


def test_fuse_streaming_leakyrelu_into_streamingadd():
    model = _build_fusable_model("StreamingAdd", "StreamingLeakyReLU")

    model, changed = FuseElementwiseOps().apply(model)

    assert changed is True
    producer = model.graph.node[0]
    producer_op = getCustomOp(producer)
    assert producer.output[0] == "output"
    assert producer_op.get_nodeattr("activation") == "LeakyReLU"
    assert producer_op.get_nodeattr("activation_alpha_num") == 13
    assert producer_op.get_nodeattr("activation_alpha_den") == 128


def test_fuse_streaming_relu_still_sets_relu_activation():
    model = _build_fusable_model("StreamingConv", "StreamingReLU")

    model, changed = FuseElementwiseOps().apply(model)

    assert changed is True
    producer = model.graph.node[0]
    assert producer.output[0] == "output"
    assert getCustomOp(producer).get_nodeattr("activation") == "ReLU"
