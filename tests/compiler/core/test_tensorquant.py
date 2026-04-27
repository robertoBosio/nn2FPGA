import pytest
import numpy as np
from nn2fpga.compiler.core.tensor_quant import (
    TensorQuant,
    set_custom_tensor_datatype,
    get_custom_tensor_datatype,
)
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper


# ── TensorQuant unit tests ─────────────────────────────────────────────────────

def test_basic_construction():
    q = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert q.bitwidth == 8
    assert q.signed == 1
    assert q.scale == 0.1
    assert q.zeropt == 0
    assert q.narrow == 0
    assert q.rounding_mode == "ROUND"

def test_scale_none_raises():
    with pytest.raises(ValueError):
        TensorQuant(bitwidth=8, signed=1, scale=None, zeropt=0)

def test_zeropt_none_raises():
    with pytest.raises(ValueError):
        TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=None)

def test_scale_from_numpy_scalar():
    scale = np.array([0.1], dtype=np.float32)
    q = TensorQuant(bitwidth=8, signed=1, scale=scale, zeropt=0)
    assert isinstance(q.scale, float)
    assert abs(q.scale - 0.1) < 1e-6

def test_scale_multielement_array_raises():
    scale = np.array([0.1, 0.2], dtype=np.float32)
    with pytest.raises(ValueError):
        TensorQuant(bitwidth=8, signed=1, scale=scale, zeropt=0)

def test_zeropt_from_numpy_scalar():
    zeropt = np.array([0], dtype=np.int8)
    q = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=zeropt)
    assert isinstance(q.zeropt, int)
    assert q.zeropt == 0

def test_zeropt_multielement_array_raises():
    zeropt = np.array([0, 1], dtype=np.int8)
    with pytest.raises(ValueError):
        TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=zeropt)

def test_equality():
    q1 = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    q2 = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert q1 == q2

def test_inequality_bitwidth():
    q1 = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    q2 = TensorQuant(bitwidth=16, signed=1, scale=0.1, zeropt=0)
    assert q1 != q2

def test_inequality_signed():
    q1 = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    q2 = TensorQuant(bitwidth=8, signed=0, scale=0.1, zeropt=0)
    assert q1 != q2

def test_inequality_scale():
    q1 = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    q2 = TensorQuant(bitwidth=8, signed=1, scale=0.2, zeropt=0)
    assert q1 != q2

def test_inequality_zeropt():
    q1 = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    q2 = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=1)
    assert q1 != q2

def test_inequality_wrong_type():
    q = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert q != "Q[8,1,0.1,0,0,ROUND]"

def test_canonical_name_roundtrip():
    q = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0, narrow=0, rounding_mode="ROUND")
    assert TensorQuant.from_canonical_name(q.get_canonical_name()) == q

def test_canonical_name_unsigned():
    q = TensorQuant(bitwidth=4, signed=0, scale=0.5, zeropt=0, narrow=1, rounding_mode="CEIL")
    assert TensorQuant.from_canonical_name(q.get_canonical_name()) == q

def test_canonical_name_invalid_raises():
    with pytest.raises(ValueError):
        TensorQuant.from_canonical_name("invalid_string")

def test_canonical_name_missing_fields_raises():
    with pytest.raises(ValueError):
        TensorQuant.from_canonical_name("Q[8,1,0.1,0]")  # missing narrow and rounding_mode

def test_repr():
    q = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert "Q[8,1," in repr(q)


# ── TensorProto dtype tests ────────────────────────────────────────────────────

def test_tensorproto_dtype_signed_8():
    q = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert q.get_tensorproto_dtype() == TensorProto.INT8

def test_tensorproto_dtype_signed_16():
    q = TensorQuant(bitwidth=16, signed=1, scale=0.1, zeropt=0)
    assert q.get_tensorproto_dtype() == TensorProto.INT16

def test_tensorproto_dtype_signed_32():
    q = TensorQuant(bitwidth=32, signed=1, scale=0.1, zeropt=0)
    assert q.get_tensorproto_dtype() == TensorProto.INT32

def test_tensorproto_dtype_unsigned_8():
    q = TensorQuant(bitwidth=8, signed=0, scale=0.1, zeropt=0)
    assert q.get_tensorproto_dtype() == TensorProto.UINT8

def test_tensorproto_dtype_unsigned_16():
    q = TensorQuant(bitwidth=16, signed=0, scale=0.1, zeropt=0)
    assert q.get_tensorproto_dtype() == TensorProto.UINT16

def test_tensorproto_dtype_unsigned_32():
    q = TensorQuant(bitwidth=32, signed=0, scale=0.1, zeropt=0)
    assert q.get_tensorproto_dtype() == TensorProto.UINT32

def test_tensorproto_dtype_signed_overflow_raises():
    q = TensorQuant(bitwidth=64, signed=1, scale=0.1, zeropt=0)
    with pytest.raises(ValueError):
        q.get_tensorproto_dtype()

def test_tensorproto_dtype_unsigned_overflow_raises():
    q = TensorQuant(bitwidth=64, signed=0, scale=0.1, zeropt=0)
    with pytest.raises(ValueError):
        q.get_tensorproto_dtype()


# ── NumPy dtype tests ──────────────────────────────────────────────────────────

def test_numpy_dtype_signed_8():
    q = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert q.get_numpy_dtype() == np.int8

def test_numpy_dtype_signed_16():
    q = TensorQuant(bitwidth=16, signed=1, scale=0.1, zeropt=0)
    assert q.get_numpy_dtype() == np.int16

def test_numpy_dtype_signed_32():
    q = TensorQuant(bitwidth=32, signed=1, scale=0.1, zeropt=0)
    assert q.get_numpy_dtype() == np.int32

def test_numpy_dtype_unsigned_8():
    q = TensorQuant(bitwidth=8, signed=0, scale=0.1, zeropt=0)
    assert q.get_numpy_dtype() == np.uint8

def test_numpy_dtype_unsigned_16():
    q = TensorQuant(bitwidth=16, signed=0, scale=0.1, zeropt=0)
    assert q.get_numpy_dtype() == np.uint16

def test_numpy_dtype_unsigned_32():
    q = TensorQuant(bitwidth=32, signed=0, scale=0.1, zeropt=0)
    assert q.get_numpy_dtype() == np.uint32

def test_numpy_dtype_overflow_raises():
    q = TensorQuant(bitwidth=64, signed=1, scale=0.1, zeropt=0)
    with pytest.raises(ValueError):
        q.get_numpy_dtype()


# ── Annotation tests ───────────────────────────────────────────────────────────

def _make_simple_model():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])
    identity_node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([identity_node], "test_graph", [input_tensor], [output_tensor])
    model = qonnx_make_model(graph, producer_name="test_producer")
    return ModelWrapper(model)

def test_set_and_get_quant():
    model = _make_simple_model()
    q = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    set_custom_tensor_datatype(model, "input", q)
    assert get_custom_tensor_datatype(model, "input") == q

def test_get_quant_not_set_returns_none():
    model = _make_simple_model()
    assert get_custom_tensor_datatype(model, "input") is None

def test_overwrite_quant():
    model = _make_simple_model()
    set_custom_tensor_datatype(model, "input", TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0))
    set_custom_tensor_datatype(model, "input", TensorQuant(bitwidth=16, signed=1, scale=0.2, zeropt=1))
    result = get_custom_tensor_datatype(model, "input")
    assert result == TensorQuant(bitwidth=16, signed=1, scale=0.2, zeropt=1)

def test_clear_quant():
    model = _make_simple_model()
    set_custom_tensor_datatype(model, "input", TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0))
    set_custom_tensor_datatype(model, "input", None)
    assert get_custom_tensor_datatype(model, "input") is None

def test_quant_on_multiple_tensors():
    model = _make_simple_model()
    q_in = TensorQuant(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    q_out = TensorQuant(bitwidth=16, signed=0, scale=0.5, zeropt=2)
    set_custom_tensor_datatype(model, "input", q_in)
    set_custom_tensor_datatype(model, "output", q_out)
    assert get_custom_tensor_datatype(model, "input") == q_in
    assert get_custom_tensor_datatype(model, "output") == q_out