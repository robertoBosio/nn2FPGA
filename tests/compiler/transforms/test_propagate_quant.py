from nn2fpga.compiler.transforms.propagate_quant import PropagateQuant
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper

def test_simple_forward_propagate_quant():
    # Test the forward propagation in a single chain of nodes.
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

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

    identity_node = helper.make_node(
        "Identity",
        inputs=["quantized_input"],
        outputs=["output"],
        name="identity_node"
    )

    graph = helper.make_graph(
        [quant_node, identity_node],
        "test_graph",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("zero_point", TensorProto.INT8, [1], [0]),
            helper.make_tensor("bitwidth", TensorProto.INT32, [1], [8]),
        ]
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the PropagateQuant transformation
    transformed_model = model.transform(PropagateQuant())

    # Check if the quantization was propagated correctly
    assert len(transformed_model.get_nodes_by_op_type("Quant")) == 2, "The Quant node should have been propagated forward."
    assert len(transformed_model.graph.output) == 1, "The output should still be a single tensor after quantization propagation."
    assert len(transformed_model.graph.input) == 1, "The input should still be a single tensor after quantization propagation."

def test_simple_backward_propagate_quant():
    # Test the backward propagation in a single chain of nodes.
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    identity_node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["identity_output"],
        name="identity_node"
    )

    quant_node = helper.make_node(
        "Quant",
        inputs=["identity_output", "scale", "zero_point", "bitwidth"],
        outputs=["quantized_output"],
        domain="qonnx.custom_op.general",
        name="quantize_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    graph = helper.make_graph(
        [identity_node, quant_node],
        "test_graph",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("zero_point", TensorProto.INT8, [1], [0]),
            helper.make_tensor("bitwidth", TensorProto.INT32, [1], [8]),
        ]
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the PropagateQuant transformation
    transformed_model = model.transform(PropagateQuant())

    # Check if the quantization was propagated correctly
    assert len(transformed_model.get_nodes_by_op_type("Quant")) == 2, "The Quant node should have been propagated backward."
    assert len(transformed_model.graph.output) == 1, "The output should still be a single tensor after quantization propagation."
    assert len(transformed_model.graph.input) == 1, "The input should still be a single tensor after quantization propagation."

def test_propagate_backward_with_multiple_consumers_same_quant():
    # Test propagation with multiple outputs sharing the same quantization parameters.
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor1 = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor2 = helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1, 3, 224, 224])

    identity_node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["id_output"],
        name="identity_node"
    )

    quant_node1 = helper.make_node(
        "Quant",
        inputs=["id_output", "scale", "zero_point", "bitwidth"],
        outputs=["output_tensor1"],
        domain="qonnx.custom_op.general",
        name="quantize_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    quant_node2 = helper.make_node(
        "Quant",
        inputs=["id_output", "scale", "zero_point", "bitwidth"],
        outputs=["output_tensor2"],
        domain="qonnx.custom_op.general",
        name="quantize_node2",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    graph = helper.make_graph(
        [identity_node, quant_node1, quant_node2],
        "test_graph",
        [input_tensor],
        [output_tensor1, output_tensor2],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("zero_point", TensorProto.INT8, [1], [0]),
            helper.make_tensor("bitwidth", TensorProto.INT32, [1], [8]),
        ]
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the PropagateQuant transformation
    transformed_model = model.transform(PropagateQuant())

    # Check if the quantization was propagated correctly
    assert (
        len(transformed_model.get_nodes_by_op_type("Quant")) == 3
    ), "Quant node should have been propagated to the input as both outputs share the same quantization."
    assert len(transformed_model.graph.output) == 2, "The outputs should still be two tensors after quantization propagation."
    assert len(transformed_model.graph.input) == 1, "The input should still be a single tensor after quantization propagation."

def test_propagate_backward_with_multiple_consumers_different_quant():
    # Test propagation with multiple outputs having different quantization parameters.
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor1 = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor2 = helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1, 3, 224, 224])

    identity_node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["id_output"],
        name="identity_node"
    )

    quant_node1 = helper.make_node(
        "Quant",
        inputs=["id_output", "scale1", "zero_point1", "bitwidth1"],
        outputs=["output_tensor1"],
        domain="qonnx.custom_op.general",
        name="quantize_node1",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    quant_node2 = helper.make_node(
        "Quant",
        inputs=["id_output", "scale2", "zero_point2", "bitwidth2"],
        outputs=["output_tensor2"],
        domain="qonnx.custom_op.general",
        name="quantize_node2",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    graph = helper.make_graph(
        [identity_node, quant_node1, quant_node2],
        "test_graph",
        [input_tensor],
        [output_tensor1, output_tensor2],
        initializer=[
            helper.make_tensor("scale1", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("zero_point1", TensorProto.INT8, [1], [0]),
            helper.make_tensor("bitwidth1", TensorProto.INT32, [1], [8]),
            helper.make_tensor("scale2", TensorProto.FLOAT, [1], [0.2]),
            helper.make_tensor("zero_point2", TensorProto.INT8, [1], [1]),
            helper.make_tensor("bitwidth2", TensorProto.INT32, [1], [16]),
        ]
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the PropagateQuant transformation
    transformed_model = model.transform(PropagateQuant())

    # Check if the quantization was not propagated
    assert (
        len(transformed_model.get_nodes_by_op_type("Quant")) == 2
    ), "Quant node should not have been propagated to the input, as outputs have different quantization parameters."
    assert len(transformed_model.graph.output) == 2, "The outputs should still be two tensors after quantization propagation."
    assert len(transformed_model.graph.input) == 1, "The input should still be a single tensor after quantization propagation."

def test_propagate_forward_with_multiple_producers_same_quant():
    # Test propagation with multiple inputs to a single quantization node.
    input_tensor1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3, 224, 224])
    input_tensor2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 6, 224, 224])

    quant_node1 = helper.make_node(
        "Quant",
        inputs=["input1", "scale", "zero_point", "bitwidth"],
        outputs=["output_tensor1"],
        domain="qonnx.custom_op.general",
        name="quantize_node1",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    quant_node2 = helper.make_node(
        "Quant",
        inputs=["input2", "scale", "zero_point", "bitwidth"],
        outputs=["output_tensor2"],
        domain="qonnx.custom_op.general",
        name="quantize_node2",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    concat_node = helper.make_node(
        "Concat",
        inputs=["output_tensor1", "output_tensor2"],
        outputs=["output"],
        axis=1,
        name="concat_node"
    )

    graph = helper.make_graph(
        [quant_node1, quant_node2, concat_node],
        "test_graph",
        [input_tensor1, input_tensor2],
        [output_tensor],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("zero_point", TensorProto.INT8, [1], [0]),
            helper.make_tensor("bitwidth", TensorProto.INT32, [1], [8]),
        ]
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the PropagateQuant transformation
    transformed_model = model.transform(PropagateQuant())

    # Check if the quantization was propagated correctly
    assert (
        len(transformed_model.get_nodes_by_op_type("Quant")) == 3
    ), "Quant node should have been propagated to the output as both inputs share the same quantization."
    assert len(transformed_model.graph.output) == 1, "The output should still be a single tensor after quantization propagation."
    assert len(transformed_model.graph.input) == 2, "The input should still be two tensors after quantization propagation."

def test_propagate_forward_with_multiple_producers_different_quant():
    # Test propagation with multiple inputs to a single quantization node with different quantization parameters.
    input_tensor1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3, 224, 224])
    input_tensor2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 6, 224, 224])

    quant_node1 = helper.make_node(
        "Quant",
        inputs=["input1", "scale1", "zero_point1", "bitwidth1"],
        outputs=["output_tensor1"],
        domain="qonnx.custom_op.general",
        name="quantize_node1",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    quant_node2 = helper.make_node(
        "Quant",
        inputs=["input2", "scale2", "zero_point2", "bitwidth2"],
        outputs=["output_tensor2"],
        domain="qonnx.custom_op.general",
        name="quantize_node2",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    concat_node = helper.make_node(
        "Concat",
        inputs=["output_tensor1", "output_tensor2"],
        outputs=["output"],
        axis=1,
        name="concat_node"
    )

    graph = helper.make_graph(
        [quant_node1, quant_node2, concat_node],
        "test_graph",
        [input_tensor1, input_tensor2],
        [output_tensor],
        initializer=[
            helper.make_tensor("scale1", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("zero_point1", TensorProto.INT8, [1], [0]),
            helper.make_tensor("bitwidth1", TensorProto.INT32, [1], [8]),
            helper.make_tensor("scale2", TensorProto.FLOAT, [1], [0.2]),
            helper.make_tensor("zero_point2", TensorProto.INT8, [1], [1]),
            helper.make_tensor("bitwidth2", TensorProto.INT32, [1], [16]),
        ]
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the PropagateQuant transformation
    transformed_model = model.transform(PropagateQuant())

    # Check if the quantization was not propagated
    assert (
        len(transformed_model.get_nodes_by_op_type("Quant")) == 2
    ), "Quant node should not have been propagated to the output, as inputs have different quantization parameters."
    assert len(transformed_model.graph.output) == 1, "The output should still be a single tensor after quantization propagation."
    assert len(transformed_model.graph.input) == 2, "The input should still be two tensors after quantization propagation."

def test_simple_propagate_backward_quant_with_constant():
    # Test propagation with a constant input.
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 12, 1, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [10, 12])

    quant_node = helper.make_node(
        "Quant",
        inputs=["input_reshaped", "scale", "zero_point", "bitwidth"],
        outputs=["output"],
        domain="qonnx.custom_op.general",
        name="quantize_node",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    reshape_node = helper.make_node(
        "Reshape",
        inputs=["input", "shape"],
        outputs=["input_reshaped"],
        name="reshape_node"
    )

    shape_tensor = helper.make_tensor(
        name="shape",
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[1, -1]
    )

    shape_node = helper.make_node(
        "Constant",
        value=shape_tensor,
        outputs=["shape"],
        inputs=[],
    )

    graph = helper.make_graph(
        [quant_node, reshape_node, shape_node],
        "test_graph",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("zero_point", TensorProto.INT8, [1], [0]),
            helper.make_tensor("bitwidth", TensorProto.INT32, [1], [8]),
        ]
    )

    model = qonnx_make_model(graph, producer_name="test_producer")
    model = ModelWrapper(model)

    # Apply the PropagateQuant transformation
    transformed_model = model.transform(PropagateQuant())

    # Check if the quantization was propagated correctly
    assert (
        len(transformed_model.get_nodes_by_op_type("Quant")) == 2
    ), "Quant node should have been propagated to the output."
    assert len(transformed_model.graph.output) == 1, "The output should still be a single tensor after quantization propagation."
    assert len(transformed_model.graph.input) == 1, "The input should still be a single tensor after quantization propagation."
