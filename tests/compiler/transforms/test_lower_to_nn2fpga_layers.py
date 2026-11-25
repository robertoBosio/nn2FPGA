from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.transforms.lower_to_nn2fpga_layers import LowerToNN2FPGALayers


def _build_sigmoid_model() -> ModelWrapper:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])

    sigmoid = helper.make_node(
        "Sigmoid",
        inputs=["input"],
        outputs=["output"],
        name="sigmoid_node",
    )

    graph = helper.make_graph(
        [sigmoid],
        name="sigmoid_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def _build_hardsigmoid_mul_model(alpha: float = 0.1666666716337204, b_value: float = 1.0001220703125) -> ModelWrapper:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])

    B = helper.make_tensor("B", TensorProto.FLOAT, [], [b_value])

    hardsigmoid = helper.make_node(
        "HardSigmoid",
        inputs=["input"],
        outputs=["hardsigmoid_out"],
        name="hardsigmoid_node",
        alpha=alpha,
    )
    mul = helper.make_node(
        "Mul",
        inputs=["hardsigmoid_out", "B"],
        outputs=["output"],
        name="mul_node",
    )

    graph = helper.make_graph(
        [hardsigmoid, mul],
        name="hardsigmoid_mul_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[B],
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def _build_focus_model() -> ModelWrapper:
    input_tensor = helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, 3, 8, 8])
    output_tensor = helper.make_tensor_value_info("focus_out", TensorProto.FLOAT, [1, 12, 4, 4])

    initializers = []
    for name, value in (
        ("even", 0),
        ("odd", 1),
        ("end", 9223372036854775807),
        ("height_axis", 2),
        ("width_axis", 3),
        ("step", 2),
        ("scale", 1.0),
        ("zeropt", -128),
        ("bitwidth", 8.0),
    ):
        dtype = TensorProto.FLOAT if name in {"scale", "bitwidth"} else TensorProto.INT64
        initializers.append(helper.make_tensor(name, dtype, [1], [value]))

    def slice_node(name: str, data: str, start: str, axis: str) -> object:
        return helper.make_node(
            "Slice", [data, start, "end", axis, "step"], [name], name=f"slice_{name}"
        )

    def quant_node(name: str, data: str) -> object:
        return helper.make_node(
            "Quant",
            [data, "scale", "zeropt", "bitwidth"],
            [name],
            name=f"quant_{name}",
            domain="qonnx.custom_op.general",
            signed=1,
            narrow=0,
            rounding_mode="ROUND",
        )

    nodes = [
        slice_node("h_even", "source", "even", "height_axis"),
        slice_node("h_odd", "source", "odd", "height_axis"),
        quant_node("q_h_even", "h_even"),
        quant_node("q_h_odd", "h_odd"),
        slice_node("ee", "q_h_even", "even", "width_axis"),
        slice_node("eo", "q_h_even", "odd", "width_axis"),
        slice_node("oe", "q_h_odd", "even", "width_axis"),
        slice_node("oo", "q_h_odd", "odd", "width_axis"),
        quant_node("q_ee", "ee"),
        quant_node("q_eo", "eo"),
        quant_node("q_oe", "oe"),
        quant_node("q_oo", "oo"),
        helper.make_node("Concat", ["q_ee", "q_oe"], ["concat01"], axis=1),
        quant_node("q_concat01", "concat01"),
        helper.make_node("Concat", ["q_concat01", "q_eo"], ["concat012"], axis=1),
        quant_node("q_concat012", "concat012"),
        helper.make_node("Concat", ["q_concat012", "q_oo"], ["focus_out"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        name="focus_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def test_lower_real_sigmoid_to_streamingsigmoid():
    model = _build_sigmoid_model()

    model, changed = LowerToNN2FPGALayers().apply(model)

    assert changed is False
    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "StreamingSigmoid"
    assert node.domain == "nn2fpga.compiler.custom_op"
    assert getCustomOp(node).get_nodeattr("impl") == "sigmoid"


def test_lower_hardsigmoid_mul_to_streamingsigmoid_legacy_impl():
    alpha = 0.1666666716337204
    b_value = 1.0001220703125
    model = _build_hardsigmoid_mul_model(alpha=alpha, b_value=b_value)

    model, changed = LowerToNN2FPGALayers().apply(model)

    assert changed is False
    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    op = getCustomOp(node)
    assert node.op_type == "StreamingSigmoid"
    assert node.domain == "nn2fpga.compiler.custom_op"
    assert op.get_nodeattr("impl") == "hardsigmoid_mul"
    assert op.get_nodeattr("alpha") == alpha
    assert op.get_nodeattr("B") == b_value


def test_lower_focus_to_streamingwindowtochannels():
    model = _build_focus_model()

    model, changed = LowerToNN2FPGALayers().apply(model)

    assert changed is False
    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "StreamingWindowToChannels"
    assert node.domain == "nn2fpga.compiler.custom_op"
    assert list(node.input) == ["source"]
    assert list(node.output) == ["focus_out"]
    op = getCustomOp(node)
    assert op.get_nodeattr("kernel_shape") == [2, 2]
    assert op.get_nodeattr("strides") == [2, 2]
    assert op.get_nodeattr("pads") == [0, 0]
