import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.transforms.supported_partition import (
    _same_quant_params,
    check_act_quant,
    check_params_quant,
    match_supported_patterns,
)


def _make_quant_node(
    input_name: str,
    output_name: str,
    node_name: str,
    scale_name: str,
    zeropt_name: str,
    bitwidth_name: str,
    rounding_mode: str = "ROUND",
    narrow: int = 0,
):
    return helper.make_node(
        "Quant",
        inputs=[input_name, scale_name, zeropt_name, bitwidth_name],
        outputs=[output_name],
        domain="qonnx.custom_op.general",
        name=node_name,
        signed=1,
        narrow=narrow,
        rounding_mode=rounding_mode,
    )


def _make_quant_initializers(
    scale_name: str,
    zeropt_name: str,
    bitwidth_name: str,
    scale_vals,
    zeropt_vals,
    zeropt_dtype=TensorProto.INT8,
):
    return [
        helper.make_tensor(scale_name, TensorProto.FLOAT, [len(scale_vals)], scale_vals),
        helper.make_tensor(zeropt_name, zeropt_dtype, [len(zeropt_vals)], zeropt_vals),
        helper.make_tensor(bitwidth_name, TensorProto.INT32, [1], [8]),
    ]


def _wrap_model(nodes, inputs, outputs, initializers=None):
    graph = helper.make_graph(
        nodes=nodes,
        name="test_graph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers or [],
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def _build_activation_quant_model(
    *,
    narrow=0,
    rounding_mode="ROUND",
    scale_vals=(0.1,),
    zeropt_vals=(0,),
    zeropt_dtype=TensorProto.INT8,
):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    quant_node = _make_quant_node(
        "input",
        "output",
        "quant_node",
        "scale",
        "zero_point",
        "bitwidth",
        rounding_mode=rounding_mode,
        narrow=narrow,
    )
    initializers = _make_quant_initializers(
        "scale",
        "zero_point",
        "bitwidth",
        scale_vals,
        zeropt_vals,
        zeropt_dtype=zeropt_dtype,
    )
    model = _wrap_model([quant_node], [input_tensor], [output_tensor], initializers)
    return model, quant_node


def _build_params_quant_model(*, zeropt_vals=(0,)):
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    quant_node = _make_quant_node(
        "weights",
        "output",
        "quant_node",
        "scale",
        "zero_point",
        "bitwidth",
    )
    initializers = [
        helper.make_tensor("weights", TensorProto.FLOAT, [1, 4], [1.0, 2.0, 3.0, 4.0]),
        *_make_quant_initializers(
            "scale",
            "zero_point",
            "bitwidth",
            (0.1,),
            zeropt_vals,
        ),
    ]
    model = _wrap_model([quant_node], [], [output_tensor], initializers)
    return model, quant_node


def test_check_act_quant_accepts_supported_quant():
    model, quant_node = _build_activation_quant_model()
    reasons = []

    assert check_act_quant(model, quant_node, reasons) is True
    assert reasons == []


@pytest.mark.parametrize(
    ("kwargs", "expected_reason"),
    [
        ({"narrow": 1}, "unexpected value 1, expected 0"),
        ({"rounding_mode": "FLOOR"}, "unexpected value FLOOR, expected ROUND"),
        ({"scale_vals": (0.1, 0.2)}, "per-channel quantization"),
        ({"zeropt_vals": (0.5,), "zeropt_dtype": TensorProto.FLOAT}, "floating zero point"),
    ],
)
def test_check_act_quant_rejects_unsupported_quant(kwargs, expected_reason):
    model, quant_node = _build_activation_quant_model(**kwargs)
    reasons = []

    assert check_act_quant(model, quant_node, reasons) is False
    assert any(expected_reason in reason for reason in reasons)


def test_check_act_quant_rejects_const_only_quant():
    model, quant_node = _build_params_quant_model()
    reasons = []

    assert check_act_quant(model, quant_node, reasons) is False
    assert any("must not have initializers" in reason for reason in reasons)


def test_check_params_quant_accepts_supported_quant():
    model, quant_node = _build_params_quant_model()
    reasons = []

    assert check_params_quant(model, quant_node, reasons) is True
    assert reasons == []


def test_check_params_quant_rejects_non_constant_source():
    model, quant_node = _build_activation_quant_model()
    reasons = []

    assert check_params_quant(model, quant_node, reasons) is False
    assert any("must have initializers" in reason for reason in reasons)


def test_check_params_quant_rejects_asymmetric_quant():
    model, quant_node = _build_params_quant_model(zeropt_vals=(1,))
    reasons = []

    assert check_params_quant(model, quant_node, reasons) is False
    assert any("unsupported asymmetric quantization" in reason for reason in reasons)


def test_same_quant_params_returns_true_for_equal_params():
    input_tensor_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, [1, 4])
    input_tensor_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])
    quant_node_a = _make_quant_node(
        "input_a", "quant_a", "quant_a", "scale_a", "zero_a", "bitwidth_a"
    )
    quant_node_b = _make_quant_node(
        "input_b", "quant_b", "quant_b", "scale_b", "zero_b", "bitwidth_b"
    )
    concat_node = helper.make_node(
        "Concat",
        inputs=["quant_a", "quant_b"],
        outputs=["output"],
        axis=1,
        name="concat_node",
    )
    initializers = [
        *_make_quant_initializers("scale_a", "zero_a", "bitwidth_a", (0.1,), (0,)),
        *_make_quant_initializers("scale_b", "zero_b", "bitwidth_b", (0.1,), (0,)),
    ]
    model = _wrap_model(
        [quant_node_a, quant_node_b, concat_node],
        [input_tensor_a, input_tensor_b],
        [output_tensor],
        initializers,
    )

    assert _same_quant_params(model, quant_node_a, quant_node_b) is True


def test_same_quant_params_returns_false_for_mismatched_params():
    input_tensor_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, [1, 4])
    input_tensor_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])
    quant_node_a = _make_quant_node(
        "input_a", "quant_a", "quant_a", "scale_a", "zero_a", "bitwidth_a"
    )
    quant_node_b = _make_quant_node(
        "input_b", "quant_b", "quant_b", "scale_b", "zero_b", "bitwidth_b"
    )
    concat_node = helper.make_node(
        "Concat",
        inputs=["quant_a", "quant_b"],
        outputs=["output"],
        axis=1,
        name="concat_node",
    )
    initializers = [
        *_make_quant_initializers("scale_a", "zero_a", "bitwidth_a", (0.1,), (0,)),
        *_make_quant_initializers("scale_b", "zero_b", "bitwidth_b", (0.2,), (0,)),
    ]
    model = _wrap_model(
        [quant_node_a, quant_node_b, concat_node],
        [input_tensor_a, input_tensor_b],
        [output_tensor],
        initializers,
    )

    assert _same_quant_params(model, quant_node_a, quant_node_b) is False


def test_match_supported_patterns_reports_missing_pattern_registration():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    identity_node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["output"],
        name="identity_node",
    )
    model = _wrap_model([identity_node], [input_tensor], [output_tensor])

    match = match_supported_patterns(model, identity_node)

    assert match.ok is False
    assert match.pattern_name == "NoPatternsForOp"
    assert any("No patterns registered for op Identity" in reason for reason in match.reasons)


def test_match_supported_patterns_reports_failed_known_pattern():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    relu_node = helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"],
        name="relu_node",
    )
    model = _wrap_model([relu_node], [input_tensor], [output_tensor])

    match = match_supported_patterns(model, relu_node)

    assert match.ok is False
    assert match.pattern_name == "NoPatternMatched"
    assert match.reasons
