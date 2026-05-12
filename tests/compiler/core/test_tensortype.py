import pytest
import numpy as np
from nn2fpga.compiler.core.tensor_type import (
    QuantizedTensorType,
    FloatTensorType,
    TensorType,
    set_custom_tensor_datatype,
    get_custom_tensor_datatype,
)
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_simple_model():
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    identity_node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph(
        [identity_node], "test_graph", [input_tensor], [output_tensor]
    )
    model = qonnx_make_model(graph, producer_name="test_producer")
    return ModelWrapper(model)

# ── QuantizedTensorType unit tests ────────────────────────────────────────────

def test_basic_construction():
    q = QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert q.bitwidth == 8
    assert q.signed == 1
    assert q.scale == 0.1
    assert q.zeropt == 0
    assert q.narrow == 0
    assert q.rounding_mode == "ROUND"


def test_scale_none_raises():
    with pytest.raises(ValueError):
        QuantizedTensorType(bitwidth=8, signed=1, scale=None, zeropt=0)


def test_zeropt_none_raises():
    with pytest.raises(ValueError):
        QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=None)


def test_scale_from_numpy_scalar():
    scale = np.array([0.1], dtype=np.float32)
    q = QuantizedTensorType(bitwidth=8, signed=1, scale=scale, zeropt=0)
    assert isinstance(q.scale, float)
    assert abs(q.scale - 0.1) < 1e-6


def test_scale_multielement_array_raises():
    scale = np.array([0.1, 0.2], dtype=np.float32)
    with pytest.raises(ValueError):
        QuantizedTensorType(bitwidth=8, signed=1, scale=scale, zeropt=0)


def test_zeropt_from_numpy_scalar():
    zeropt = np.array([0], dtype=np.int8)
    q = QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=zeropt)
    assert isinstance(q.zeropt, int)
    assert q.zeropt == 0


def test_zeropt_multielement_array_raises():
    zeropt = np.array([0, 1], dtype=np.int8)
    with pytest.raises(ValueError):
        QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=zeropt)


def test_equality():
    q1 = QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    q2 = QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert q1 == q2


def test_inequality_bitwidth():
    assert QuantizedTensorType(
        bitwidth=8, signed=1, scale=0.1, zeropt=0
    ) != QuantizedTensorType(bitwidth=16, signed=1, scale=0.1, zeropt=0)


def test_inequality_signed():
    assert QuantizedTensorType(
        bitwidth=8, signed=1, scale=0.1, zeropt=0
    ) != QuantizedTensorType(bitwidth=8, signed=0, scale=0.1, zeropt=0)


def test_inequality_scale():
    assert QuantizedTensorType(
        bitwidth=8, signed=1, scale=0.1, zeropt=0
    ) != QuantizedTensorType(bitwidth=8, signed=1, scale=0.2, zeropt=0)


def test_inequality_zeropt():
    assert QuantizedTensorType(
        bitwidth=8, signed=1, scale=0.1, zeropt=0
    ) != QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=1)


def test_inequality_wrong_type():
    q = QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert q != "Q[8,1,0.1,0,0,ROUND]"


def test_inequality_quantized_vs_float():
    assert (
        QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
        != FloatTensorType()
    )


def test_canonical_name_roundtrip():
    q = QuantizedTensorType(
        bitwidth=8, signed=1, scale=0.1, zeropt=0, narrow=0, rounding_mode="ROUND"
    )
    assert QuantizedTensorType.from_canonical_name(q.get_canonical_name()) == q


def test_canonical_name_unsigned():
    q = QuantizedTensorType(
        bitwidth=4, signed=0, scale=0.5, zeropt=0, narrow=1, rounding_mode="CEIL"
    )
    assert QuantizedTensorType.from_canonical_name(q.get_canonical_name()) == q


def test_canonical_name_invalid_raises():
    with pytest.raises(ValueError):
        QuantizedTensorType.from_canonical_name("invalid_string")


def test_canonical_name_missing_fields_raises():
    with pytest.raises(ValueError):
        QuantizedTensorType.from_canonical_name("Q[8,1,0.1,0]")


def test_repr_quantized():
    q = QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    assert "QuantizedTensorType" in repr(q)
    assert "Q[8,1," in repr(q)

def test_quantized_bitwidth():
    q = QuantizedTensorType(bitwidth=16, signed=1, scale=0.1, zeropt=0)
    assert q.bitwidth == 16

# ── TensorProto dtype tests — QuantizedTensorType ─────────────────────────────

@pytest.mark.parametrize(
    "bitwidth,signed,expected",
    [
        (8, 1, TensorProto.INT8),
        (16, 1, TensorProto.INT16),
        (32, 1, TensorProto.INT32),
        (8, 0, TensorProto.UINT8),
        (16, 0, TensorProto.UINT16),
        (32, 0, TensorProto.UINT32),
    ],
)
def test_tensorproto_dtype(bitwidth, signed, expected):
    q = QuantizedTensorType(bitwidth=bitwidth, signed=signed, scale=0.1, zeropt=0)
    assert q.get_tensorproto_dtype() == expected


def test_tensorproto_dtype_overflow_signed_raises():
    with pytest.raises(ValueError):
        QuantizedTensorType(
            bitwidth=64, signed=1, scale=0.1, zeropt=0
        ).get_tensorproto_dtype()


def test_tensorproto_dtype_overflow_unsigned_raises():
    with pytest.raises(ValueError):
        QuantizedTensorType(
            bitwidth=64, signed=0, scale=0.1, zeropt=0
        ).get_tensorproto_dtype()


# ── NumPy dtype tests — QuantizedTensorType ───────────────────────────────────

@pytest.mark.parametrize(
    "bitwidth,signed,expected",
    [
        (8, 1, np.int8),
        (16, 1, np.int16),
        (32, 1, np.int32),
        (8, 0, np.uint8),
        (16, 0, np.uint16),
        (32, 0, np.uint32),
    ],
)
def test_numpy_dtype(bitwidth, signed, expected):
    q = QuantizedTensorType(bitwidth=bitwidth, signed=signed, scale=0.1, zeropt=0)
    assert q.get_numpy_dtype() == expected


def test_numpy_dtype_overflow_raises():
    with pytest.raises(ValueError):
        QuantizedTensorType(
            bitwidth=64, signed=1, scale=0.1, zeropt=0
        ).get_numpy_dtype()

# ── HLS data type tests — QuantizedTensorType ───────────────────────────────────

@pytest.mark.parametrize("bitwidth,signed,expected", [
    (8,  1, "ap_int<8>"),
    (8,  0, "ap_uint<8>"),
    (16, 1, "ap_int<16>"),
    (16, 0, "ap_uint<16>"),
    (4,  1, "ap_int<4>"),
    (4,  0, "ap_uint<4>"),
])
def test_hls_data_type_quantized(bitwidth, signed, expected):
    q = QuantizedTensorType(bitwidth=bitwidth, signed=signed, scale=0.1, zeropt=0)
    assert q.get_hls_data_type() == expected

# ── C++ quant type tests — QuantizedTensorType ───────────────────────────────────

@pytest.mark.parametrize("bitwidth,signed,expected", [
    (8,  1, "char"),
    (8,  0, "unsigned char"),
    (16, 1, "short"),
    (16, 0, "unsigned short"),
    (32, 1, "int"),
    (32, 0, "unsigned int"),
])
def test_cpp_quant_type_quantized(bitwidth, signed, expected):
    q = QuantizedTensorType(bitwidth=bitwidth, signed=signed, scale=0.1, zeropt=0)
    assert q.get_cpp_quant_type() == expected

def test_cpp_quant_type_overflow_raises():
    q = QuantizedTensorType(bitwidth=64, signed=1, scale=0.1, zeropt=0)
    with pytest.raises(ValueError):
        q.get_cpp_quant_type()


# ── FloatTensorType unit tests ────────────────────────────────────────────────

def test_float_canonical_name():
    assert FloatTensorType().get_canonical_name() == "float32"


def test_float_tensorproto_dtype():
    assert FloatTensorType().get_tensorproto_dtype() == TensorProto.FLOAT


def test_float_numpy_dtype():
    assert FloatTensorType().get_numpy_dtype() == np.float32


def test_float_equality():
    assert FloatTensorType() == FloatTensorType()


def test_float_inequality_wrong_type():
    assert FloatTensorType() != "float32"


def test_float_repr():
    assert "FloatTensorType" in repr(FloatTensorType())
    assert "float32" in repr(FloatTensorType())


def test_float_hashable():
    # FloatTensorType must be usable as a dict key / in a set
    s = {FloatTensorType(), FloatTensorType()}
    assert len(s) == 1

def test_hls_data_type_float():
    assert FloatTensorType().get_hls_data_type() == "float"

def test_cpp_quant_type_float():
    assert FloatTensorType().get_cpp_quant_type() == "float"

def test_float_bitwidth():
    assert FloatTensorType().bitwidth == 32

# ── TensorType.from_canonical_name dispatch ───────────────────────────────────

def test_dispatch_float():
    t = TensorType.from_canonical_name("float32")
    assert isinstance(t, FloatTensorType)


def test_dispatch_quantized():
    t = TensorType.from_canonical_name("Q[8,1,0.1,0,0,ROUND]")
    assert isinstance(t, QuantizedTensorType)
    assert t == QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)


def test_dispatch_invalid_raises():
    with pytest.raises(ValueError):
        TensorType.from_canonical_name("unknown_type")


# ── Annotation tests ───────────────────────────────────────────────────────────


def test_set_and_get_quantized():
    model = _make_simple_model()
    q = QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    set_custom_tensor_datatype(model, "input", q)
    assert get_custom_tensor_datatype(model, "input") == q


def test_set_and_get_float():
    model = _make_simple_model()
    set_custom_tensor_datatype(model, "input", FloatTensorType())
    result = get_custom_tensor_datatype(model, "input")
    assert isinstance(result, FloatTensorType)
    assert result == FloatTensorType()


def test_get_quant_not_set_returns_none():
    model = _make_simple_model()
    assert get_custom_tensor_datatype(model, "input") is None


def test_overwrite_quantized_with_float():
    model = _make_simple_model()
    set_custom_tensor_datatype(
        model, "input", QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    )
    set_custom_tensor_datatype(model, "input", FloatTensorType())
    assert get_custom_tensor_datatype(model, "input") == FloatTensorType()


def test_overwrite_float_with_quantized():
    model = _make_simple_model()
    q = QuantizedTensorType(bitwidth=16, signed=1, scale=0.2, zeropt=1)
    set_custom_tensor_datatype(model, "input", FloatTensorType())
    set_custom_tensor_datatype(model, "input", q)
    assert get_custom_tensor_datatype(model, "input") == q


def test_overwrite_quantized():
    model = _make_simple_model()
    set_custom_tensor_datatype(
        model, "input", QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    )
    set_custom_tensor_datatype(
        model, "input", QuantizedTensorType(bitwidth=16, signed=1, scale=0.2, zeropt=1)
    )
    assert get_custom_tensor_datatype(model, "input") == QuantizedTensorType(
        bitwidth=16, signed=1, scale=0.2, zeropt=1
    )


def test_clear_annotation():
    model = _make_simple_model()
    set_custom_tensor_datatype(
        model, "input", QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    )
    set_custom_tensor_datatype(model, "input", None)
    assert get_custom_tensor_datatype(model, "input") is None


def test_annotation_on_multiple_tensors():
    model = _make_simple_model()
    q_in = QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    q_out = FloatTensorType()
    set_custom_tensor_datatype(model, "input", q_in)
    set_custom_tensor_datatype(model, "output", q_out)
    assert get_custom_tensor_datatype(model, "input") == q_in
    assert get_custom_tensor_datatype(model, "output") == q_out
