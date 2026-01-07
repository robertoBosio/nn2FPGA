from backend.transformation.fullyconnected_to_conv import FullyConnectedToPointwise
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper
import numpy as np

def test_reshape_pattern():
    # Test the substitution of a Reshape -> Quant -> Gemm pattern with a Pointwise Conv -> Reshape pattern.
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1280, 1, 1]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1000]
    )

    scale_tensor = helper.make_tensor(
        name="scale",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[0.1],
    )
    zero_point_tensor = helper.make_tensor(
        name="zero_point",
        data_type=TensorProto.INT8,
        dims=[1],
        vals=[0],
    )
    bitwidth_tensor = helper.make_tensor(
        name="bitwidth",
        data_type=TensorProto.INT32,
        dims=[1],
        vals=[8],
    )

    quant_node = helper.make_node(
        "Quant",
        inputs=["input", "scale", "zero_point", "bitwidth"],
        outputs=["quantized_input"],
        domain="qonnx.custom_op.general",
        name="quantize_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    reshape_shape_tensor = helper.make_tensor(
        name="reshape_shape",
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[1, -1],  # Reshape to keep the batch size and flatten the rest
    )

    reshape_node = helper.make_node(
        "Reshape",
        inputs=["quantized_input", "reshape_shape"],
        outputs=["reshaped_input"],
        name="reshape_node",
    )

    second_quant_node = helper.make_node(
        "Quant",
        inputs=["reshaped_input", "scale", "zero_point", "bitwidth"],
        outputs=["quantized_reshaped_input"],
        domain="qonnx.custom_op.general",
        name="second_quantize_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    gemm_weights = helper.make_tensor(
        name="gemm_weights",
        data_type=TensorProto.INT8,
        dims=[1000, 1280],
        vals=np.random.randint(-128, 127, size=(1000 * 1280)).astype(np.int8).tolist(),
    )

    gemm_weights_scale = helper.make_tensor(
        name="gemm_weights_scale",
        data_type=TensorProto.FLOAT,
        dims=[1000, 1],
        vals=np.random.rand(1000).astype(np.float32).tolist(),
    )

    gemm_weights_zero_point = helper.make_tensor(
        name="gemm_weights_zero_point",
        data_type=TensorProto.INT8,
        dims=[1000, 1],
        vals=np.zeros(1000, dtype=np.int8).tolist(),
    )
    gemm_weights_bitwidth = helper.make_tensor(
        name="gemm_weights_bitwidth",
        data_type=TensorProto.INT32,
        dims=[],
        vals=[8],
    )

    gemm_weights_quant = helper.make_node(
        "Quant",
        inputs=[
            "gemm_weights",
            "gemm_weights_scale",
            "gemm_weights_zero_point",
            "gemm_weights_bitwidth",
        ],
        outputs=["quantized_gemm_weights"],
        domain="qonnx.custom_op.general",
        name="gemm_weights_quant_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    gemm_bias = helper.make_tensor(
        name="gemm_bias",
        data_type=TensorProto.INT32,
        dims=[1000],
        vals=np.random.randint(-1000, 1000, size=1000).astype(np.int32).tolist(),
    )

    gemm_bias_scale = helper.make_tensor(
        name="gemm_bias_scale",
        data_type=TensorProto.FLOAT,
        dims=[1000],
        vals=np.random.rand(1000).astype(np.float32).tolist(),
    )

    gemm_bias_zero_point = helper.make_tensor(
        name="gemm_bias_zero_point",
        data_type=TensorProto.INT32,
        dims=[1000],
        vals=np.zeros(1000, dtype=np.int32).tolist(),
    )
    gemm_bias_bitwidth = helper.make_tensor(
        name="gemm_bias_bitwidth",
        data_type=TensorProto.INT32,
        dims=[],
        vals=[32],
    )

    gemm_bias_quant = helper.make_node(
        "Quant",
        inputs=[
            "gemm_bias",
            "gemm_bias_scale",
            "gemm_bias_zero_point",
            "gemm_bias_bitwidth",
        ],
        outputs=["quantized_gemm_bias"],
        domain="qonnx.custom_op.general",
        name="gemm_bias_quant_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    gemm_node = helper.make_node(
        "Gemm",
        inputs=[
            "quantized_reshaped_input",
            "quantized_gemm_weights",
            "quantized_gemm_bias",
        ],
        outputs=["output"],
        alpha=1.0,
        beta=1.0,
        transB=1,
        name="gemm_node",
    )

    # Create the graph
    graph = helper.make_graph(
        [
            quant_node,
            reshape_node,
            second_quant_node,
            gemm_weights_quant,
            gemm_bias_quant,
            gemm_node,
        ],
        "test_graph",
        [input_tensor],
        [output_tensor],
        initializer=[
            reshape_shape_tensor,
            gemm_weights,
            gemm_bias,
            scale_tensor,
            zero_point_tensor,
            bitwidth_tensor,
            gemm_weights_scale,
            gemm_weights_zero_point,
            gemm_weights_bitwidth,
            gemm_bias_scale,
            gemm_bias_zero_point,
            gemm_bias_bitwidth,
        ],
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the FullyConnectedToPointwise transformation
    transformed_model = model.transform(FullyConnectedToPointwise())

    assert (
        len(transformed_model.get_nodes_by_op_type("Quant")) == 3
    ), "One Quant node should have been removed."
    assert (
        len(transformed_model.graph.output) == 1
    ), "The output should still be a single tensor after tranformation."
    assert (
        len(transformed_model.graph.input) == 1
    ), "The input should still be a single tensor after tranformation."
    assert (
        len(transformed_model.get_nodes_by_op_type("Conv")) == 1
    ), "One Conv node should have been created."
    assert (
        len(transformed_model.get_nodes_by_op_type("Reshape")) == 2
    ), "Two Reshape nodes should have been created."


def test_flatten_pattern():
    # Test the substitution of a Flatten -> Quant -> Gemm pattern with a Pointwise Conv -> Reshape pattern.
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1280, 1, 1]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1000]
    )

    scale_tensor = helper.make_tensor(
        name="scale",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[0.1],
    )
    zero_point_tensor = helper.make_tensor(
        name="zero_point",
        data_type=TensorProto.INT8,
        dims=[1],
        vals=[0],
    )
    bitwidth_tensor = helper.make_tensor(
        name="bitwidth",
        data_type=TensorProto.INT32,
        dims=[1],
        vals=[8],
    )

    quant_node = helper.make_node(
        "Quant",
        inputs=["input", "scale", "zero_point", "bitwidth"],
        outputs=["quantized_input"],
        domain="qonnx.custom_op.general",
        name="quantize_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    flatten_node = helper.make_node(
        "Flatten",
        inputs=["quantized_input"],
        outputs=["flattened_input"],
        axis=1,
        name="flatten_node",
    )

    second_quant_node = helper.make_node(
        "Quant",
        inputs=["flattened_input", "scale", "zero_point", "bitwidth"],
        outputs=["quantized_flattened_input"],
        domain="qonnx.custom_op.general",
        name="second_quantize_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    gemm_weights = helper.make_tensor(
        name="gemm_weights",
        data_type=TensorProto.INT8,
        dims=[1000, 1280],
        vals=np.random.randint(-128, 127, size=(1000 * 1280)).astype(np.int8).tolist(),
    )

    gemm_weights_scale = helper.make_tensor(
        name="gemm_weights_scale",
        data_type=TensorProto.FLOAT,
        dims=[1000, 1],
        vals=np.random.rand(1000).astype(np.float32).tolist(),
    )

    gemm_weights_zero_point = helper.make_tensor(
        name="gemm_weights_zero_point",
        data_type=TensorProto.INT8,
        dims=[1000, 1],
        vals=np.zeros(1000, dtype=np.int8).tolist(),
    )
    gemm_weights_bitwidth = helper.make_tensor(
        name="gemm_weights_bitwidth",
        data_type=TensorProto.INT32,
        dims=[],
        vals=[8],
    )

    gemm_weights_quant = helper.make_node(
        "Quant",
        inputs=[
            "gemm_weights",
            "gemm_weights_scale",
            "gemm_weights_zero_point",
            "gemm_weights_bitwidth",
        ],
        outputs=["quantized_gemm_weights"],
        domain="qonnx.custom_op.general",
        name="gemm_weights_quant_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    gemm_bias = helper.make_tensor(
        name="gemm_bias",
        data_type=TensorProto.INT32,
        dims=[1000],
        vals=np.random.randint(-1000, 1000, size=1000).astype(np.int32).tolist(),
    )

    gemm_bias_scale = helper.make_tensor(
        name="gemm_bias_scale",
        data_type=TensorProto.FLOAT,
        dims=[1000],
        vals=np.random.rand(1000).astype(np.float32).tolist(),
    )

    gemm_bias_zero_point = helper.make_tensor(
        name="gemm_bias_zero_point",
        data_type=TensorProto.INT32,
        dims=[1000],
        vals=np.zeros(1000, dtype=np.int32).tolist(),
    )
    gemm_bias_bitwidth = helper.make_tensor(
        name="gemm_bias_bitwidth",
        data_type=TensorProto.INT32,
        dims=[],
        vals=[32],
    )

    gemm_bias_quant = helper.make_node(
        "Quant",
        inputs=[
            "gemm_bias",
            "gemm_bias_scale",
            "gemm_bias_zero_point",
            "gemm_bias_bitwidth",
        ],
        outputs=["quantized_gemm_bias"],
        domain="qonnx.custom_op.general",
        name="gemm_bias_quant_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    gemm_node = helper.make_node(
        "Gemm",
        inputs=[
            "quantized_flattened_input",
            "quantized_gemm_weights",
            "quantized_gemm_bias",
        ],
        outputs=["output"],
        alpha=1.0,
        beta=1.0,
        transB=1,
        name="gemm_node",
    )

    # Create the graph
    graph = helper.make_graph(
        [
            quant_node,
            flatten_node,
            second_quant_node,
            gemm_weights_quant,
            gemm_bias_quant,
            gemm_node,
        ],
        "test_graph",
        [input_tensor],
        [output_tensor],
        initializer=[
            gemm_weights,
            gemm_bias,
            scale_tensor,
            zero_point_tensor,
            bitwidth_tensor,
            gemm_weights_scale,
            gemm_weights_zero_point,
            gemm_weights_bitwidth,
            gemm_bias_scale,
            gemm_bias_zero_point,
            gemm_bias_bitwidth,
        ],
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the FullyConnectedToPointwise transformation
    transformed_model = model.transform(FullyConnectedToPointwise())

    assert (
        len(transformed_model.get_nodes_by_op_type("Quant")) == 3
    ), "One Quant node should have been removed."
    assert (
        len(transformed_model.graph.output) == 1
    ), "The output should still be a single tensor after tranformation."
    assert (
        len(transformed_model.graph.input) == 1
    ), "The input should still be a single tensor after tranformation."
    assert (
        len(transformed_model.get_nodes_by_op_type("Conv")) == 1
    ), "One Conv node should have been created."
    assert (
        len(transformed_model.get_nodes_by_op_type("Reshape")) == 2
    ), "Two Reshape nodes should have been created."
    assert (
        len(transformed_model.get_nodes_by_op_type("Flatten")) == 0
    ), "No Flatten node should have been left."