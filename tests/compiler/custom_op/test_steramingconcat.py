import pytest
import numpy as np
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper
from nn2fpga.compiler.custom_op.streamingconcat import StreamingConcat
from nn2fpga.compiler.analysis.check_quantization import check_input_same_quant
from nn2fpga.compiler.core.tensor_type import (
    QuantizedTensorType,
    get_custom_tensor_datatype,
    set_custom_tensor_datatype,
)
from nn2fpga.compiler.core.tensor_layout import TensorLayout, set_custom_tensor_layout

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_concat_model(
    shapeA,
    shapeB,
    axis,
    layoutA=None,
    layoutB=None,
    layout_out=None,
    quantA=None,
    quantB=None,
    quant_out=None,
):
    """Build a minimal model with a single StreamingConcat node."""
    rank = len(shapeA)
    shapeOut = list(shapeA)
    shapeOut[axis] += shapeB[axis]

    inA = helper.make_tensor_value_info("inputA", TensorProto.FLOAT, shapeA)
    inB = helper.make_tensor_value_info("inputB", TensorProto.FLOAT, shapeB)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shapeOut)

    node = helper.make_node(
        "StreamingConcat",
        inputs=["inputA", "inputB"],
        outputs=["output"],
        domain="nn2fpga.compiler.custom_op",
        name="concat_node",
        axis=axis,
        dim2_unroll=1,
        dim1_unroll=1,
        in_stream_array=1,
        out_stream_array=1,
        in_word_array=1,
        out_word_array=1,
    )

    graph = helper.make_graph([node], "test_graph", [inA, inB], [out])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="test"))

    # Default quant: 8-bit signed
    default_quant = QuantizedTensorType(bitwidth=8, signed=1, scale=0.5, zeropt=0)
    set_custom_tensor_datatype(model, "inputA", quantA or default_quant)
    set_custom_tensor_datatype(model, "inputB", quantB or default_quant)
    set_custom_tensor_datatype(model, "output", quant_out or default_quant)

    # Default layout: identity
    identity = TensorLayout.identity(rank)
    set_custom_tensor_layout(model, "inputA", layoutA or identity)
    set_custom_tensor_layout(model, "inputB", layoutB or identity)
    set_custom_tensor_layout(model, "output", layout_out or identity)

    return model


def _get_concat_op(model) -> StreamingConcat:
    from qonnx.custom_op.registry import getCustomOp
    node = model.get_nodes_by_op_type("StreamingConcat")[0]
    return getCustomOp(node)


def _build_quantized_concat_model(scale_a, zp_a, scale_b, zp_b):
    x0 = helper.make_tensor_value_info("x0", TensorProto.FLOAT, [1])
    x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])

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
        helper.make_node("Concat", inputs=["qa", "qb"], outputs=["y"], name="concat", axis=0),
    ]
    return ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "quant_concat_graph", [x0, x1], [y], initializer=initializers),
            producer_name="test",
        )
    )


# ── accepted_input_layout / produced_output_layout ────────────────────────────


def test_accepted_input_layout_is_none():
    op = _get_concat_op(_make_concat_model([1, 3, 8, 8], [1, 3, 8, 8], axis=1))
    assert op.accepted_input_layout() is None


def test_produced_output_layout_passthrough():
    op = _get_concat_op(_make_concat_model([1, 3, 8, 8], [1, 3, 8, 8], axis=1))
    layout = TensorLayout((0, 2, 3, 1))
    assert op.produced_output_layout(layout) == layout


def test_produced_output_layout_none_passthrough():
    op = _get_concat_op(_make_concat_model([1, 3, 8, 8], [1, 3, 8, 8], axis=1))
    assert op.produced_output_layout(None) is None


# ── axis permutation logic ─────────────────────────────────────────────────────


def test_axis_permuted_selects_dim2_class():
    # NHWC layout: perm=(0,2,3,1). ONNX axis=1 (C) maps to physical axis 3 → StreamingConcatDim2
    layout = TensorLayout((0, 2, 3, 1))
    model = _make_concat_model(
        [1, 3, 8, 8],
        [1, 5, 8, 8],
        axis=1,
        layoutA=layout,
        layoutB=layout,
        layout_out=layout,
    )
    op = _get_concat_op(model)
    declaration = op._StreamingConcat__get_object_declaration(model)
    assert "StreamingConcatDim2" in declaration


def test_axis_permuted_selects_dim0_class():
    # NHWC layout: perm=(0,2,3,1). ONNX axis=3 (W) maps to physical axis 2 → StreamingConcatDim1
    layout = TensorLayout((0, 2, 3, 1))
    model = _make_concat_model(
        [1, 3, 8, 8],
        [1, 3, 4, 8],
        axis=3,
        layoutA=layout,
        layoutB=layout,
        layout_out=layout,
    )
    op = _get_concat_op(model)
    declaration = op._StreamingConcat__get_object_declaration(model)
    assert "StreamingConcatDim1" in declaration


def test_mismatched_layouts_raises():
    model = _make_concat_model(
        [1, 3, 8, 8],
        [1, 3, 8, 8],
        axis=1,
        layoutA=TensorLayout((0, 2, 3, 1)),
        layoutB=TensorLayout((0, 3, 2, 1)),
    )
    op = _get_concat_op(model)
    with pytest.raises(ValueError, match="layouts.*must be the same"):
        op._StreamingConcat__get_object_declaration(model)


# ── template argument tests ────────────────────────────────────────────────────


def test_template_args_contain_correct_shapes():
    # NHWC: perm=(0,2,3,1). Concat on ONNX axis=1 (C, physical axis 3).
    # inputA shape NCHW=[1,3,8,8] → permuted NHWC=[1,8,8,3]
    # inputB shape NCHW=[1,5,8,8] → permuted NHWC=[1,8,8,5]
    layout = TensorLayout((0, 2, 3, 1))
    model = _make_concat_model(
        [1, 3, 8, 8],
        [1, 5, 8, 8],
        axis=1,
        layoutA=layout,
        layoutB=layout,
        layout_out=layout,
    )
    op = _get_concat_op(model)
    declaration = op._StreamingConcat__get_object_declaration(model)
    assert "3" in declaration  # IN_DIM2_A
    assert "5" in declaration  # IN_DIM2_B
    assert "8" in declaration  # IN_DIM0 / IN_DIM1


def test_template_args_contain_unroll_factors():
    layout = TensorLayout((0, 2, 3, 1))
    model = _make_concat_model(
        [1, 4, 8, 8],
        [1, 4, 8, 8],
        axis=1,
        layoutA=layout,
        layoutB=layout,
        layout_out=layout,
    )
    # Manually set unroll factors
    op = _get_concat_op(model)
    op.set_nodeattr("dim1_unroll", 2)
    op.set_nodeattr("dim2_unroll", 4)
    declaration = op._StreamingConcat__get_object_declaration(model)
    assert "2" in declaration  # DIM1_UNROLL
    assert "4" in declaration  # DIM2_UNROLL


# ── DSE points ─────────────────────────────────────────────────────────────────


def test_dse_points_nonempty():
    layout = TensorLayout((0, 2, 3, 1))
    model = _make_concat_model(
        [1, 4, 8, 8],
        [1, 4, 8, 8],
        axis=1,
        layoutA=layout,
        layoutB=layout,
        layout_out=layout,
    )
    op = _get_concat_op(model)
    points = op.get_dse_points(model)
    assert len(points) > 0


def test_dse_points_all_divide_shapes():
    layout = TensorLayout((0, 2, 3, 1))
    model = _make_concat_model(
        [1, 4, 8, 8],
        [1, 4, 8, 8],
        axis=1,
        layoutA=layout,
        layoutB=layout,
        layout_out=layout,
    )
    op = _get_concat_op(model)
    points = op.get_dse_points(model)
    for p in points:
        assert 8 % p.dim1_unroll == 0
        assert 4 % p.dim2_unroll == 0


def test_dse_points_respect_stream_width_limit():
    layout = TensorLayout((0, 2, 3, 1))
    quant = QuantizedTensorType(bitwidth=8, signed=1, scale=0.5, zeropt=0)
    model = _make_concat_model(
        [1, 512, 8, 8],
        [1, 512, 8, 8],
        axis=1,
        layoutA=layout,
        layoutB=layout,
        layout_out=layout,
        quantA=quant,
        quantB=quant,
        quant_out=quant,
    )
    op = _get_concat_op(model)
    points = op.get_dse_points(model)
    for p in points:
        assert quant.bitwidth * p.dim2_unroll <= 4096


def test_dse_mismatched_layouts_raises():
    model = _make_concat_model(
        [1, 3, 8, 8],
        [1, 3, 8, 8],
        axis=1,
        layoutA=TensorLayout((0, 2, 3, 1)),
        layoutB=TensorLayout((0, 3, 2, 1)),
    )
    op = _get_concat_op(model)
    with pytest.raises(ValueError):
        op.get_dse_points(model)


# ── apply_point ────────────────────────────────────────────────────────────────


def test_apply_point_sets_attributes():
    model = _make_concat_model([1, 4, 8, 8], [1, 4, 8, 8], axis=1)
    op = _get_concat_op(model)
    point = StreamingConcat.DSEPoint(dim2_unroll=4, dim1_unroll=2)
    op.apply_point(model, point)
    assert op.get_nodeattr("dim2_unroll") == 4
    assert op.get_nodeattr("dim1_unroll") == 2
    assert op.get_nodeattr("in_stream_array") == 2
    assert op.get_nodeattr("out_stream_array") == 2
    assert op.get_nodeattr("in_word_array") == 4
    assert op.get_nodeattr("out_word_array") == 4


def test_apply_point_roundtrip():
    model = _make_concat_model([1, 4, 8, 8], [1, 4, 8, 8], axis=1)
    op = _get_concat_op(model)
    points = op.get_dse_points(model)
    for p in points:
        op.apply_point(model, p)
        assert op.get_nodeattr("dim2_unroll") == p.dim2_unroll
        assert op.get_nodeattr("dim1_unroll") == p.dim1_unroll


def test_object_declaration_supports_po2_asymmetric_per_input_concat():
    model = _make_concat_model(
        [1, 4, 8, 8],
        [1, 4, 8, 8],
        axis=1,
        quantA=QuantizedTensorType(bitwidth=8, signed=0, scale=2**-4, zeropt=23),
        quantB=QuantizedTensorType(bitwidth=8, signed=0, scale=2**-6, zeropt=101),
        quant_out=QuantizedTensorType(bitwidth=8, signed=0, scale=2**-5, zeropt=77),
    )

    declaration = _get_concat_op(model)._StreamingConcat__get_object_declaration(model)

    assert "TInputWordA" in declaration
    assert "TInputWordB" in declaration
    assert "RequantizePo2<-1, 23, 77, ap_uint<8>, ap_uint<8>>" in declaration
    assert "RequantizePo2<1, 101, 77, ap_uint<8>, ap_uint<8>>" in declaration


def test_object_declaration_supports_float_asymmetric_per_input_concat():
    model = _make_concat_model(
        [1, 4, 8, 8],
        [1, 4, 8, 8],
        axis=1,
        quantA=QuantizedTensorType(bitwidth=8, signed=0, scale=0.375, zeropt=91),
        quantB=QuantizedTensorType(bitwidth=8, signed=1, scale=0.2, zeropt=-17),
        quant_out=QuantizedTensorType(bitwidth=8, signed=0, scale=0.3, zeropt=109),
    )

    declaration = _get_concat_op(model)._StreamingConcat__get_object_declaration(model)

    assert "RequantizeFloat<5, 4, 91, 109, ap_uint<8>, ap_uint<8>>" in declaration
    assert "RequantizeFloat<2, 3, -17, 109, ap_int<8>, ap_uint<8>>" in declaration


def test_infer_node_datatype_keeps_output_quant_type():
    quant_in = QuantizedTensorType(bitwidth=8, signed=1, scale=0.5, zeropt=0)
    quant_out = QuantizedTensorType(bitwidth=8, signed=0, scale=0.25, zeropt=13)
    model = _make_concat_model(
        [1, 4, 8, 8],
        [1, 4, 8, 8],
        axis=1,
        quantA=quant_in,
        quantB=quant_in,
        quant_out=quant_out,
    )

    op = _get_concat_op(model)
    op.infer_node_datatype(model)

    assert get_custom_tensor_datatype(model, "output") == quant_out


def test_check_input_same_quant_allows_concat_with_different_quant_params():
    model = _build_quantized_concat_model(0.1, 0, 0.2, 3)
    assert check_input_same_quant(model) is True
