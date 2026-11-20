import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.transforms.normalize_asymmetric_weight_quant import (
    NormalizeAsymmetricWeightQuant,
)
from nn2fpga.compiler.utils.quantization import quant_array


def _build_model(*, zeropt, raw_weights, scale=1.0):
    weights = helper.make_node(
        "Quant",
        inputs=["weight_values", "weight_scale", "weight_zero_point", "weight_bitwidth"],
        outputs=["quantized_weights"],
        domain="qonnx.custom_op.general",
        name="weight_quant",
        signed=0,
        narrow=0,
        rounding_mode="ROUND",
    )
    conv = helper.make_node(
        "Conv",
        inputs=["input", "quantized_weights"],
        outputs=["output"],
        name="conv",
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph(
        [weights, conv],
        "normalize_asymmetric_weight_quant",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1, 1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 1, 1])],
        initializer=[
            helper.make_tensor("weight_values", TensorProto.FLOAT, [2, 1, 1, 1], raw_weights),
            helper.make_tensor("weight_scale", TensorProto.FLOAT, [], [scale]),
            helper.make_tensor("weight_zero_point", TensorProto.INT32, [], [zeropt]),
            helper.make_tensor("weight_bitwidth", TensorProto.INT32, [], [8]),
        ],
    )
    return ModelWrapper(qonnx_make_model(graph))


def _weight_quantizer(model):
    return next(node for node in model.graph.node if node.name == "weight_quant")


def test_normalizes_uint8_weights_to_signed_int8_without_changing_values():
    model = _build_model(zeropt=128, raw_weights=[-128.0, 127.0])

    transformed, changed = NormalizeAsymmetricWeightQuant().apply(model)
    quant = _weight_quantizer(transformed)

    assert changed is False
    assert int(transformed.get_initializer("weight_zero_point").reshape(())) == 0
    assert int(transformed.get_initializer("weight_bitwidth").reshape(())) == 8
    assert next(attr.i for attr in quant.attribute if attr.name == "signed") == 1
    np.testing.assert_array_equal(
        quant_array(
            transformed.get_initializer("weight_values"),
            scale=1.0,
            zeropt=0,
            bitwidth=8,
            signed=True,
            narrow=False,
            rounding_mode="ROUND",
        ),
        np.array([[[[-128]]], [[[127]]]], dtype=np.int32),
    )


def test_widens_centered_weights_to_nine_bits_when_needed():
    model = _build_model(zeropt=1, raw_weights=[-1.0, 254.0])

    transformed, _ = NormalizeAsymmetricWeightQuant().apply(model)

    assert int(transformed.get_initializer("weight_zero_point").reshape(())) == 0
    assert int(transformed.get_initializer("weight_bitwidth").reshape(())) == 9
    np.testing.assert_array_equal(
        quant_array(
            transformed.get_initializer("weight_values"),
            scale=1.0,
            zeropt=0,
            bitwidth=9,
            signed=True,
            narrow=False,
            rounding_mode="ROUND",
        ),
        np.array([[[[-1]]], [[[254]]]], dtype=np.int32),
    )


def test_keeps_non_negative_centered_weights_unsigned_at_eight_bits():
    scale = 0.00007470748823834583
    model = _build_model(zeropt=121, raw_weights=[0.0, 255 * scale], scale=scale)

    transformed, _ = NormalizeAsymmetricWeightQuant().apply(model)
    quant = _weight_quantizer(transformed)

    assert int(transformed.get_initializer("weight_zero_point").reshape(())) == 0
    assert int(transformed.get_initializer("weight_bitwidth").reshape(())) == 8
    assert next(attr.i for attr in quant.attribute if attr.name == "signed") == 0
    np.testing.assert_array_equal(
        quant_array(
            transformed.get_initializer("weight_values"),
            scale=scale,
            zeropt=0,
            bitwidth=8,
            signed=False,
            narrow=False,
            rounding_mode="ROUND",
        ),
        np.array([[[[0]]], [[[134]]]], dtype=np.int32),
    )


def test_preserves_codes_at_zero_point_rounding_boundary():
    scale = 0.00007470748823834583
    model = _build_model(
        zeropt=121,
        raw_weights=[29 * scale, 59 * scale],
        scale=scale,
    )

    original_codes = quant_array(
        model.get_initializer("weight_values"),
        scale=scale,
        zeropt=121,
        bitwidth=8,
        signed=False,
        narrow=False,
        rounding_mode="ROUND",
    )
    transformed, _ = NormalizeAsymmetricWeightQuant().apply(model)

    np.testing.assert_array_equal(
        original_codes,
        np.array([[[[150]]], [[[180]]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        quant_array(
            transformed.get_initializer("weight_values"),
            scale=scale,
            zeropt=0,
            bitwidth=8,
            signed=False,
            narrow=False,
            rounding_mode="ROUND",
        ),
        np.array([[[[29]]], [[[59]]]], dtype=np.int32),
    )


def test_rejects_per_channel_weight_quantization():
    model = _build_model(zeropt=1, raw_weights=[-1.0, 254.0])
    model.set_initializer("weight_scale", np.array([1.0, 1.0], dtype=np.float32))

    with pytest.raises(ValueError, match="scalar or single-element array"):
        NormalizeAsymmetricWeightQuant().apply(model)
