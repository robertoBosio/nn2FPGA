from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.core.tensor_type import (
    QuantizedTensorType,
    get_custom_tensor_datatype,
    set_custom_tensor_datatype,
)
from nn2fpga.compiler.transforms.infer_quant import InferQuant


def _make_concat_model():
    input_a = helper.make_tensor_value_info("qa", TensorProto.FLOAT, [1, 4, 1, 1])
    input_b = helper.make_tensor_value_info("qb", TensorProto.FLOAT, [1, 4, 1, 1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 1, 1])

    nodes = [
        helper.make_node(
            "Concat",
            inputs=["qa", "qb"],
            outputs=["output"],
            axis=1,
            name="concat_node",
        ),
    ]
    return ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "concat_quant_graph", [input_a, input_b], [output]),
            producer_name="test",
        )
    )


def test_infer_quant_propagates_concat_output_when_input_quants_match():
    model = _make_concat_model()
    quant = QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0)
    set_custom_tensor_datatype(model, "qa", quant)
    set_custom_tensor_datatype(model, "qb", quant)
    model = model.transform(InferQuant())

    assert get_custom_tensor_datatype(model, "output") == quant


def test_infer_quant_skips_concat_output_when_input_quants_differ():
    model = _make_concat_model()
    set_custom_tensor_datatype(model, "qa", QuantizedTensorType(bitwidth=8, signed=1, scale=0.1, zeropt=0))
    set_custom_tensor_datatype(model, "qb", QuantizedTensorType(bitwidth=8, signed=1, scale=0.2, zeropt=3))
    model = model.transform(InferQuant())

    assert get_custom_tensor_datatype(model, "output") is None
