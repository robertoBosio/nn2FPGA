from nn2fpga.compiler.transforms.fold_reshape_into_initializer import (
    FoldReshapeIntoInitializer,
)
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper
import numpy as np

def test_reshape_fold_perchannel_quantization():

    output_tensor = helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1000, 1280, 1, 1],
    )

    weights = helper.make_tensor(
        name="weights",
        data_type=TensorProto.INT8,
        dims=[1000, 1280],
        vals=np.random.randint(-128, 127, size=(1000 * 1280)).astype(np.int8).tolist(),
    )

    weights_scale = helper.make_tensor(
        name="weights_scale",
        data_type=TensorProto.FLOAT,
        dims=[1000, 1],
        vals=np.random.rand(1000).astype(np.float32).tolist(),
    )

    weights_zero_point = helper.make_tensor(
        name="weights_zero_point",
        data_type=TensorProto.INT8,
        dims=[1000, 1],
        vals=np.zeros(1000, dtype=np.int8).tolist(),
    )

    weights_bitwidth = helper.make_tensor(
        name="weights_bitwidth",
        data_type=TensorProto.INT32,
        dims=[],
        vals=[8],
    )

    weights_quant = helper.make_node(
        "Quant",
        inputs=[
            "weights",
            "weights_scale",
            "weights_zero_point",
            "weights_bitwidth",
        ],
        outputs=["quantized_weights"],
        domain="qonnx.custom_op.general",
        name="weights_quant_node",
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
        inputs=["quantized_weights", "reshape_shape"],
        outputs=["output_tensor"],
        name="reshape_node",
    )

    # Create the graph
    graph = helper.make_graph(
        [
            weights_quant,
            reshape_node,
        ],
        "test_graph",
        [],
        [output_tensor],
        initializer=[
            reshape_shape_tensor,
            weights,
            weights_scale,
            weights_zero_point,
            weights_bitwidth,
        ],
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the FoldReshapeIntoInitializer transformation
    transformed_model = model.transform(FoldReshapeIntoInitializer())

    assert (
        len(transformed_model.get_nodes_by_op_type("Quant")) == 1
    ), "One Quant node should have been removed."
    assert (
        len(transformed_model.graph.output) == 1
    ), "The output should still be a single tensor after transformation."
    assert (
        len(transformed_model.graph.input) == 0
    ), "No input should have been added to the graph."
    assert (
        len(transformed_model.get_nodes_by_op_type("Reshape")) == 0
    ), "No Reshape nodes should have been left."

    quant_node = transformed_model.get_nodes_by_op_type("Quant")[0]
    assert (
        len(transformed_model.get_tensor_shape(quant_node.input[0])) == 4
    ), "Weights initializer should have been reshaped to 4D."
    assert (
        len(transformed_model.get_tensor_shape(quant_node.input[1])) == 4
    ), "Scale initializer should have been reshaped to 4D."
    assert (
        len(transformed_model.get_tensor_shape(quant_node.input[2])) == 4
    ), "Zero point initializer should have been reshaped to 4D."

def test_reshape_fold_pertensor_quantization():
    output_tensor = helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=TensorProto.FLOAT,
        shape=[1000, 1280, 1, 1],
    )

    weights = helper.make_tensor(
        name="weights",
        data_type=TensorProto.INT8,
        dims=[1000, 1280],
        vals=np.random.randint(-128, 127, size=(1000 * 1280)).astype(np.int8).tolist(),
    )

    weights_scale = helper.make_tensor(
        name="weights_scale",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[0.1],
    )

    weights_zero_point = helper.make_tensor(
        name="weights_zero_point",
        data_type=TensorProto.INT8,
        dims=[],
        vals=[0],
    )

    weights_bitwidth = helper.make_tensor(
        name="weights_bitwidth",
        data_type=TensorProto.INT32,
        dims=[],
        vals=[8],
    )

    weights_quant = helper.make_node(
        "Quant",
        inputs=[
            "weights",
            "weights_scale",
            "weights_zero_point",
            "weights_bitwidth",
        ],
        outputs=["quantized_weights"],
        domain="qonnx.custom_op.general",
        name="weights_quant_node",
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
        inputs=["quantized_weights", "reshape_shape"],
        outputs=["output_tensor"],
        name="reshape_node",
    )

    # Create the graph
    graph = helper.make_graph(
        [
            weights_quant,
            reshape_node,
        ],
        "test_graph",
        [],
        [output_tensor],
        initializer=[
            reshape_shape_tensor,
            weights,
            weights_scale,
            weights_zero_point,
            weights_bitwidth,
        ],
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the FoldReshapeIntoInitializer transformation
    transformed_model = model.transform(FoldReshapeIntoInitializer())

    assert (
        len(transformed_model.get_nodes_by_op_type("Quant")) == 1
    ), "One Quant node should have been removed."
    assert (
        len(transformed_model.graph.output) == 1
    ), "The output should still be a single tensor after transformation."
    assert (
        len(transformed_model.graph.input) == 0
    ), "No input should have been added to the graph."
    assert (
        len(transformed_model.get_nodes_by_op_type("Reshape")) == 0
    ), "No Reshape nodes should have been left."

    quant_node = transformed_model.get_nodes_by_op_type("Quant")[0]
    assert (
        len(transformed_model.get_tensor_shape(quant_node.input[0])) == 4
    ), "Weights initializer should have been reshaped to 4D."
    assert (
        len(transformed_model.get_tensor_shape(quant_node.input[1])) == 0
    ), "Scale initializer should have been left as a scalar value."
    assert (
        len(transformed_model.get_tensor_shape(quant_node.input[2])) == 0
    ), "Zero point initializer should have been left as a scalar value."
