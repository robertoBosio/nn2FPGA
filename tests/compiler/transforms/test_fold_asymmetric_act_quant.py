import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

import nn2fpga.compiler.custom_op  # noqa: F401
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, get_custom_tensor_datatype, set_custom_tensor_datatype
from nn2fpga.compiler.transforms.fold_asymmetric_act_quant import FoldAsymmetricActQuant


def _build_model(input_name: str) -> ModelWrapper:
    graph_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 2, 1, 1])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 1, 1])

    conv_node = helper.make_node(
        "StreamingConv",
        inputs=[
            input_name,
            "weights",
            "weight_scale",
            "weight_zero_point",
            "weight_bitwidth",
            "bias",
            "bias_scale",
            "bias_zero_point",
            "bias_bitwidth",
        ],
        outputs=["output"],
        name="conv0",
        domain="nn2fpga.compiler.custom_op",
        w_signed=1,
        w_narrow=0,
        w_rounding_mode="ROUND",
        b_signed=1,
        b_narrow=0,
    )

    graph = helper.make_graph(
        [conv_node],
        "fold_asymmetric_act_quant_graph",
        [graph_input],
        [graph_output],
        initializer=[
            helper.make_tensor("weights", TensorProto.FLOAT, [2, 2, 1, 1], [1.0, 2.0, 3.0, 4.0]),
            helper.make_tensor("weight_scale", TensorProto.FLOAT, [2], [0.25, 0.25]),
            helper.make_tensor("weight_zero_point", TensorProto.INT8, [2], [0, 0]),
            helper.make_tensor("weight_bitwidth", TensorProto.INT32, [], [8]),
            helper.make_tensor("bias", TensorProto.FLOAT, [2], [10.0, 20.0]),
            helper.make_tensor("bias_scale", TensorProto.FLOAT, [2], [0.125, 0.125]),
            helper.make_tensor("bias_zero_point", TensorProto.INT32, [2], [0, 0]),
            helper.make_tensor("bias_bitwidth", TensorProto.INT32, [], [32]),
        ],
    )

    model = ModelWrapper(qonnx_make_model(graph, producer_name="test_fold_asymmetric_act_quant"))
    set_custom_tensor_datatype(
        model,
        input_name,
        QuantizedTensorType(
            bitwidth=8,
            signed=1,
            scale=0.5,
            zeropt=3,
            narrow=0,
            rounding_mode="ROUND",
        ),
    )
    return model


def _build_model_with_bias_bitwidth(input_name: str, bias_bitwidth: int) -> ModelWrapper:
    graph_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 2, 1, 1])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 1, 1])

    conv_node = helper.make_node(
        "StreamingConv",
        inputs=[
            input_name,
            "weights",
            "weight_scale",
            "weight_zero_point",
            "weight_bitwidth",
            "bias",
            "bias_scale",
            "bias_zero_point",
            "bias_bitwidth",
        ],
        outputs=["output"],
        name="conv0",
        domain="nn2fpga.compiler.custom_op",
        w_signed=1,
        w_narrow=0,
        w_rounding_mode="ROUND",
        b_signed=1,
        b_narrow=0,
    )

    graph = helper.make_graph(
        [conv_node],
        "fold_asymmetric_act_quant_graph_with_bias_bitwidth",
        [graph_input],
        [graph_output],
        initializer=[
            helper.make_tensor("weights", TensorProto.FLOAT, [2, 2, 1, 1], [1.0, 2.0, 3.0, 4.0]),
            helper.make_tensor("weight_scale", TensorProto.FLOAT, [2], [1.0, 1.0]),
            helper.make_tensor("weight_zero_point", TensorProto.INT8, [1], [0]),
            helper.make_tensor("weight_bitwidth", TensorProto.INT32, [1], [8]),
            helper.make_tensor("bias", TensorProto.FLOAT, [2], [0.0, 0.0]),
            helper.make_tensor("bias_scale", TensorProto.FLOAT, [2], [1.0, 1.0]),
            helper.make_tensor("bias_zero_point", TensorProto.INT32, [2], [0, 0]),
            helper.make_tensor("bias_bitwidth", TensorProto.INT32, [], [bias_bitwidth]),
        ],
    )

    model = ModelWrapper(
        qonnx_make_model(graph, producer_name="test_fold_asymmetric_act_quant")
    )
    set_custom_tensor_datatype(
        model,
        input_name,
        QuantizedTensorType(
            bitwidth=8,
            signed=1,
            scale=1.0,
            zeropt=7,
            narrow=0,
            rounding_mode="ROUND",
        ),
    )
    return model


def test_fold_asymmetric_act_quant_folds_bias_without_mutating_input_quant():
    model = _build_model("input")

    transformed_model, changed = FoldAsymmetricActQuant().apply(model)

    assert changed is False
    assert get_custom_tensor_datatype(transformed_model, "input").zeropt == 3
    assert getCustomOp(transformed_model.graph.node[0]).get_nodeattr("asym_folding") == 1
    np.testing.assert_allclose(
        transformed_model.get_initializer("bias"),
        np.array([5.5, 9.5], dtype=np.float32),
    )


def test_fold_asymmetric_act_quant_leaves_attr_unset_when_input_is_already_symmetric():
    model = _build_model("internal_input")
    set_custom_tensor_datatype(
        model,
        "internal_input",
        QuantizedTensorType(
            bitwidth=8,
            signed=1,
            scale=0.5,
            zeropt=0,
            narrow=0,
            rounding_mode="ROUND",
        ),
    )
    original_bias = model.get_initializer("bias").copy()

    transformed_model, changed = FoldAsymmetricActQuant().apply(model)

    assert changed is False
    assert get_custom_tensor_datatype(transformed_model, "internal_input").zeropt == 0
    assert getCustomOp(transformed_model.graph.node[0]).get_nodeattr("asym_folding") == 0
    np.testing.assert_allclose(transformed_model.get_initializer("bias"), original_bias)


def test_fold_asymmetric_act_quant_increases_bias_bitwidth_when_needed():
    model = _build_model_with_bias_bitwidth("input", bias_bitwidth=4)

    transformed_model, changed = FoldAsymmetricActQuant().apply(model)

    assert changed is False
    assert getCustomOp(transformed_model.graph.node[0]).get_nodeattr("asym_folding") == 1
    np.testing.assert_allclose(
        transformed_model.get_initializer("bias"),
        np.array([-21.0, -49.0], dtype=np.float32),
    )
    assert int(np.asarray(transformed_model.get_initializer("bias_bitwidth")).reshape(()).item()) == 7


def test_fold_asymmetric_act_quant_uses_effective_clipped_weights_for_bias_fold():
    model = _build_model("input")
    model.set_initializer("weight_scale", np.array([0.01, 0.01], dtype=np.float32))
    model.set_initializer("bias_scale", np.array([0.005, 0.005], dtype=np.float32))

    transformed_model, changed = FoldAsymmetricActQuant().apply(model)

    assert changed is False
    assert getCustomOp(transformed_model.graph.node[0]).get_nodeattr("asym_folding") == 1
    np.testing.assert_allclose(
        transformed_model.get_initializer("bias"),
        np.array([6.595, 16.19], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_fold_asymmetric_act_quant_uses_qonnx_weight_rounding_mode():
    model = _build_model("input")
    getCustomOp(model.graph.node[0]).set_nodeattr("w_rounding_mode", "HALF_UP")
    model.set_initializer(
        "weights",
        np.array([0.125, 0.625, 0.875, 1.125], dtype=np.float32).reshape(2, 2, 1, 1),
    )

    transformed_model, changed = FoldAsymmetricActQuant().apply(model)

    assert changed is False
    np.testing.assert_allclose(
        transformed_model.get_initializer("bias"),
        np.array([8.5, 16.625], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_fold_asymmetric_act_quant_uses_qonnx_bias_rounding_mode_for_bitwidth():
    model = _build_model_with_bias_bitwidth("input", bias_bitwidth=4)
    getCustomOp(model.graph.node[0]).set_nodeattr("b_rounding_mode", "CEIL")
    model.set_initializer("weights", np.array([10.0, 0.0, 10.0, 0.0], dtype=np.float32).reshape(2, 2, 1, 1))
    model.set_initializer("weight_scale", np.array([10.0, 10.0], dtype=np.float32))
    model.set_initializer("bias", np.array([9.2, 10.2], dtype=np.float32))
    model.set_initializer("bias_scale", np.array([1.0, 1.0], dtype=np.float32))
    set_custom_tensor_datatype(
        model,
        "input",
        QuantizedTensorType(
            bitwidth=8,
            signed=1,
            scale=0.1,
            zeropt=3,
            narrow=0,
            rounding_mode="ROUND",
        ),
    )

    transformed_model, changed = FoldAsymmetricActQuant().apply(model)

    assert changed is False
    np.testing.assert_allclose(
        transformed_model.get_initializer("bias"),
        np.array([6.2, 7.2], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    assert int(np.asarray(transformed_model.get_initializer("bias_bitwidth")).reshape(()).item()) == 5
