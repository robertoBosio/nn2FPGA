from backend.transformation.insert_tensor_duplicator import InsertTensorDuplicator
from backend.core.acceleratorpackage import AcceleratorPackage
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper
import numpy as np

def build_accelerator_package(model: ModelWrapper) -> ModelWrapper:

    index = 0
    ap_input_map = {}
    for i, input in enumerate(model.graph.input):
        ap_input_map[input.name] = {
            "new_name": input.name,
            "index": index,
            "shape": None,
            "quant": None,
            "value": None,
        }
        index += 1

    index = 0 
    ap_output_map = {}
    for i, output in enumerate(model.graph.output):
        ap_output_map[output.name] = {
            "new_name": output.name,
            "index": index,
            "shape": None,
            "quant": None,
            "value": None,
        }
        index += 1

    # Create the accelerator package
    ap = AcceleratorPackage(
        input_map=ap_input_map,
        output_map=ap_output_map,
        board_name=None,
        top_name=None,
        frequency=None,
        hls_version=None,
    )
    model.set_metadata_prop("accelerator_package", ap.to_json())
    return model

def build_quant_node(
    input_name: str,
    output_name: list[str],
    scale_value: float,
    zero_point_value: int,
    bitwidth_value: int,
    node_name: str,
) -> list[helper.NodeProto]:

    scale_name = f"{node_name}_scale"
    scale = helper.make_tensor(
        name=scale_name,
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[scale_value],
    )
    zero_point_name = f"{node_name}_zero_point"
    zero_point = helper.make_tensor(
        name=zero_point_name,
        data_type=TensorProto.INT8,
        dims=[1],
        vals=[zero_point_value],
    )
    bitwidth_name = f"{node_name}_bitwidth"
    bitwidth = helper.make_tensor(
        name=bitwidth_name,
        data_type=TensorProto.INT32,
        dims=[1],
        vals=[bitwidth_value],
    )

    """Helper function to build a Quant node."""
    quant_node = helper.make_node(
        "Quant",
        inputs=[input_name, scale_name, zero_point_name, bitwidth_name],
        outputs=output_name,
        domain="qonnx.custom_op.general",
        name=node_name,
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    return [scale, zero_point, bitwidth, quant_node]

def test_double_tensor_pattern():

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1280, 1, 1]
    )
    output0_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 1280, 1, 1]
    )
    output1_tensor = helper.make_tensor_value_info(
        "output1", TensorProto.FLOAT, [1, 1280, 1, 1]
    )
    quantized_input_tensor = helper.make_tensor_value_info(
        "quantized_input", TensorProto.FLOAT, [1, 1280, 1, 1]
    )

    scale_tensor0, zero_point_tensor0, bitwidth_tensor0, quant_node0 = build_quant_node(
        "input",
        ["quantized_input"],
        0.1,
        0,
        8,
        "quantize_node",
    )

    scale_tensor1, zero_point_tensor1, bitwidth_tensor1, quant_node1 = build_quant_node(
        "quantized_input",
        ["output0"],
        0.1,
        0,
        8,
        "quantize_node_1",
    )

    scale_tensor2, zero_point_tensor2, bitwidth_tensor2, quant_node2 = build_quant_node(
        "quantized_input",
        ["output1"],
        0.1,
        0,
        8,
        "quantize_node_2",
    )

    # Create the graph
    graph = helper.make_graph(
        [
            quant_node0,
            quant_node1,
            quant_node2,
        ],
        "test_graph",
        [input_tensor],
        [output0_tensor, output1_tensor],
        initializer=[
            scale_tensor0,
            zero_point_tensor0,
            bitwidth_tensor0,
            scale_tensor1,
            zero_point_tensor1,
            bitwidth_tensor1,
            scale_tensor2,
            zero_point_tensor2,
            bitwidth_tensor2,
        ],
        value_info=[quantized_input_tensor],
    )

    model = qonnx_make_model(graph, producer_name="test_slice")
    model = ModelWrapper(model)
    model = build_accelerator_package(model)

    # Apply the SlicesToSplitTree transformation
    transformed_model = model.transform(InsertTensorDuplicator())

    # Verify that the transformed model contains Split nodes instead of Slice nodes
    dup_nodes = [n for n in transformed_model.graph.node if n.op_type == "TensorDuplicator"]

    assert len(dup_nodes) == 1, "There should be a TensorDuplicator node after transformation."

def test_triple_tensor_pattern():

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1500, 1, 1]
    )
    output0_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 1500, 1, 1]
    )
    output1_tensor = helper.make_tensor_value_info(
        "output1", TensorProto.FLOAT, [1, 1500, 1, 1]
    )
    output2_tensor = helper.make_tensor_value_info(
        "output2", TensorProto.FLOAT, [1, 1500, 1, 1]
    )
    quant_nodes = []
    quant_initializers = []
    quantized_input_tensor = helper.make_tensor_value_info(
        "quantized_input", TensorProto.FLOAT, [1, 1500, 1, 1]
    )

    scale, zero_point, bitwidth, quant_node = build_quant_node(
        "input",
        ["quantized_input"],
        0.1,
        0,
        8,
        "quantize_node",
    )
    quant_initializers.extend([scale, zero_point, bitwidth])
    quant_nodes.append(quant_node)

    for i in range(3):
        (
            scale_tensor,
            zero_point_tensor,
            bitwidth_tensor,
            quant_node,
        ) = build_quant_node(
            f"quantized_input",
            [f"output{i}"],
            0.1,
            0,
            8,
            f"quantize_node_{i}",
        )
        quant_nodes.append(quant_node)
        quant_initializers.extend([scale_tensor, zero_point_tensor, bitwidth_tensor])

    # Create the graph
    graph = helper.make_graph(
        quant_nodes,
        "test_graph",
        [input_tensor],
        [output0_tensor, output1_tensor, output2_tensor],
        initializer=quant_initializers,
        value_info=[quantized_input_tensor],
    )
    
    model = qonnx_make_model(graph, producer_name="test_slice")
    model = ModelWrapper(model)
    model = build_accelerator_package(model)

    # Apply the SlicesToSplitTree transformation
    transformed_model = model.transform(InsertTensorDuplicator())

    # Verify that the transformed model contains Split nodes instead of Slice nodes
    tensor_duplicator_nodes = [n for n in transformed_model.graph.node if n.op_type == "TensorDuplicator"]

    assert len(tensor_duplicator_nodes) == 2, "There should be two TensorDuplicator nodes after transformation."
    assert len(transformed_model.graph.output) == 3, "There should be three outputs in the transformed model."

def test_triple_tensor_with_output_pattern():

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1500, 1, 1]
    )
    output0_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 1500, 1, 1]
    )
    output1_tensor = helper.make_tensor_value_info(
        "output1", TensorProto.FLOAT, [1, 1500, 1, 1]
    )
    output2_tensor = helper.make_tensor_value_info(
        "quantized_input", TensorProto.FLOAT, [1, 1500, 1, 1]
    )
    quant_nodes = []
    quant_initializers = []

    scale, zero_point, bitwidth, quant_node = build_quant_node(
        "input",
        ["quantized_input"],
        0.1,
        0,
        8,
        "quantize_node",
    )
    quant_initializers.extend([scale, zero_point, bitwidth])
    quant_nodes.append(quant_node)

    for i in range(2):
        (
            scale_tensor,
            zero_point_tensor,
            bitwidth_tensor,
            quant_node,
        ) = build_quant_node(
            f"quantized_input",
            [f"output{i}"],
            0.1,
            0,
            8,
            f"quantize_node_{i}",
        )
        quant_nodes.append(quant_node)
        quant_initializers.extend([scale_tensor, zero_point_tensor, bitwidth_tensor])

    # Create the graph
    graph = helper.make_graph(
        quant_nodes,
        "test_graph",
        [input_tensor],
        [output0_tensor, output1_tensor, output2_tensor],
        initializer=quant_initializers,
    )
    
    model = qonnx_make_model(graph, producer_name="test_slice")
    model = ModelWrapper(model)
    model = build_accelerator_package(model)

    # Apply the SlicesToSplitTree transformation
    transformed_model = model.transform(InsertTensorDuplicator())

    # Verify that the transformed model contains Split nodes instead of Slice nodes
    tensor_duplicator_nodes = [n for n in transformed_model.graph.node if n.op_type == "TensorDuplicator"]

    assert len(tensor_duplicator_nodes) == 2, "There should be two TensorDuplicator nodes after transformation."
    assert len(transformed_model.graph.output) == 3, "There should be three outputs in the transformed model."