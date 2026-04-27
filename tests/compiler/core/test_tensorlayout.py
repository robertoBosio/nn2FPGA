import pytest
from nn2fpga.compiler.core.tensor_layout import (
    TensorLayout,
    set_custom_tensor_layout,
    get_custom_tensor_layout,
)
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper


# ── TensorLayout unit tests ────────────────────────────────────────────────────

def test_identity_layout():
    layout = TensorLayout.identity(4)
    assert layout.perm == (0, 1, 2, 3)
    assert layout.is_identity()

def test_non_identity_layout():
    layout = TensorLayout((0, 2, 3, 1))
    assert not layout.is_identity()

def test_invalid_permutation_raises():
    with pytest.raises(ValueError):
        TensorLayout((0, 1, 1, 3))  # duplicate axis

def test_empty_permutation_raises():
    with pytest.raises(ValueError):
        TensorLayout(())

def test_canonical_name_roundtrip():
    layout = TensorLayout((0, 2, 3, 1))
    assert TensorLayout.from_canonical_name(layout.get_canonical_name()) == layout

def test_canonical_name_invalid_raises():
    with pytest.raises(ValueError):
        TensorLayout.from_canonical_name("NHWC")  # wrong format

def test_equality():
    assert TensorLayout((0, 2, 3, 1)) == TensorLayout((0, 2, 3, 1))
    assert TensorLayout((0, 2, 3, 1)) != TensorLayout((0, 3, 2, 1))
    assert TensorLayout((0, 2, 3, 1)) != "L[0,2,3,1]"

def test_inverse_of_identity():
    layout = TensorLayout.identity(4)
    assert layout.inverse() == layout

def test_inverse_roundtrip():
    layout = TensorLayout((0, 2, 3, 1))  # NHWC
    assert layout.inverse().compose(layout) == TensorLayout.identity(4)
    assert layout.compose(layout.inverse()) == TensorLayout.identity(4)

def test_compose_nhwc_to_nchw():
    # NHWC -> NCHW: applying NHWC perm then its inverse gives identity
    nhwc = TensorLayout((0, 2, 3, 1))
    nchw = TensorLayout.identity(4)
    composed = nhwc.inverse().compose(nchw)
    assert composed == nhwc.inverse()

def test_compose_different_ranks_raises():
    with pytest.raises(ValueError):
        TensorLayout((0, 1, 2)).compose(TensorLayout((0, 1, 2, 3)))

def test_repr():
    layout = TensorLayout((0, 2, 3, 1))
    assert "L[0,2,3,1]" in repr(layout)

def test_3d_layout():
    # Transformer-style 3D tensor
    layout = TensorLayout((0, 2, 1))  # BSF -> BFS
    assert layout.get_canonical_name() == "L[0,2,1]"
    assert layout.inverse() == layout  # self-inverse for this perm


# ── Annotation tests ───────────────────────────────────────────────────────────

def _make_simple_model():
    """Helper: minimal single-node model with one tensor."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])
    identity_node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([identity_node], "test_graph", [input_tensor], [output_tensor])
    model = qonnx_make_model(graph, producer_name="test_producer")
    return ModelWrapper(model)

def test_set_and_get_layout():
    model = _make_simple_model()
    layout = TensorLayout((0, 2, 3, 1))
    set_custom_tensor_layout(model, "input", layout)
    assert get_custom_tensor_layout(model, "input") == layout

def test_get_layout_not_set_returns_none():
    model = _make_simple_model()
    assert get_custom_tensor_layout(model, "input") is None

def test_overwrite_layout():
    model = _make_simple_model()
    set_custom_tensor_layout(model, "input", TensorLayout((0, 2, 3, 1)))
    set_custom_tensor_layout(model, "input", TensorLayout((0, 3, 2, 1)))
    assert get_custom_tensor_layout(model, "input") == TensorLayout((0, 3, 2, 1))

def test_clear_layout():
    model = _make_simple_model()
    set_custom_tensor_layout(model, "input", TensorLayout((0, 2, 3, 1)))
    set_custom_tensor_layout(model, "input", None)
    assert get_custom_tensor_layout(model, "input") is None

def test_layout_and_quant_coexist():
    # Verify that setting a layout annotation does not disturb an existing
    # quant annotation on the same tensor and vice versa.
    from nn2fpga.compiler.core.tensor_quant import (
        TensorQuant,
        set_custom_tensor_datatype,
        get_custom_tensor_datatype,
    )
    model = _make_simple_model()
    quant = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0, narrow=0, rounding_mode="ROUND")
    layout = TensorLayout((0, 2, 3, 1))

    set_custom_tensor_datatype(model, "input", quant)
    set_custom_tensor_layout(model, "input", layout)

    assert get_custom_tensor_datatype(model, "input") == quant
    assert get_custom_tensor_layout(model, "input") == layout

def test_layout_on_multiple_tensors():
    model = _make_simple_model()
    layout_in = TensorLayout((0, 2, 3, 1))
    layout_out = TensorLayout((0, 3, 2, 1))
    set_custom_tensor_layout(model, "input", layout_in)
    set_custom_tensor_layout(model, "output", layout_out)
    assert get_custom_tensor_layout(model, "input") == layout_in
    assert get_custom_tensor_layout(model, "output") == layout_out