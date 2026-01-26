from backend.transformation.slices_to_split_tree import SlicesToSplitTree
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper
import numpy as np

def build_slice_node(
    input_name: str,
    output_name: str,
    starts: list,
    ends: list,
    axes: list,
    steps: list,
    node_name: str,
) -> list[helper.NodeProto]:

    """Helper function to build a Slice node."""
    start_tensor = helper.make_tensor(
        name=f"{node_name}_starts",
        data_type=TensorProto.INT64,
        dims=[len(starts)],
        vals=starts,
    )
    end_tensor = helper.make_tensor(
        name=f"{node_name}_ends",
        data_type=TensorProto.INT64,
        dims=[len(ends)],
        vals=ends,
    )
    axes_tensor = helper.make_tensor(
        name=f"{node_name}_axes",
        data_type=TensorProto.INT64,
        dims=[len(axes)],
        vals=axes,
    )
    steps_tensor = helper.make_tensor(
        name=f"{node_name}_steps",
        data_type=TensorProto.INT64,
        dims=[len(steps)],
        vals=steps,
    )

    start_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{node_name}_starts_out"],
        value=start_tensor,
        name=f"{node_name}_start_const",
    )
    end_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{node_name}_ends_out"],
        value=end_tensor,
        name=f"{node_name}_end_const",
    )
    axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{node_name}_axes_out"],
        value=axes_tensor,
        name=f"{node_name}_axes_const",
    )
    steps_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{node_name}_steps_out"],
        value=steps_tensor,
        name=f"{node_name}_steps_const",
    )

    slice_node = helper.make_node(
        "Slice",
        inputs=[
            input_name,
            f"{node_name}_starts_out",
            f"{node_name}_ends_out",
            f"{node_name}_axes_out",
            f"{node_name}_steps_out",
        ],
        outputs=[output_name],
        name=node_name,
    )

    return [start_node, end_node, axes_node, steps_node, slice_node]

def build_quant_node(
    input_name: str,
    output_name: str,
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
        outputs=[output_name],
        domain="qonnx.custom_op.general",
        name=node_name,
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )

    return [scale, zero_point, bitwidth, quant_node]

def test_double_slice_pattern():

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1280, 1, 1]
    )
    output0_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 640, 1, 1]
    )
    output1_tensor = helper.make_tensor_value_info(
        "output1", TensorProto.FLOAT, [1, 640, 1, 1]
    )

    scale_tensor0, zero_point_tensor0, bitwidth_tensor0, quant_node0 = build_quant_node(
        "input",
        "quantized_input",
        0.1,
        0,
        8,
        "quantize_node",
    )

    start_node0, end_node0, axes_node0, steps_node0, slice_node0 = build_slice_node(
        "quantized_input",
        "sliced_output_0",
        [0],
        [640],
        [1],
        [1],
        "slice_node_0",
    )

    start_node1, end_node1, axes_node1, steps_node1, slice_node1 = build_slice_node(
        "quantized_input",
        "sliced_output_1",
        [640],
        [1280],
        [1],
        [1],
        "slice_node_1",
    )

    scale_tensor1, zero_point_tensor1, bitwidth_tensor1, quant_node1 = build_quant_node(
        "sliced_output_0",
        "output0",
        0.1,
        0,
        8,
        "quantize_node_1",
    )

    scale_tensor2, zero_point_tensor2, bitwidth_tensor2, quant_node2 = build_quant_node(
        "sliced_output_1",
        "output1",
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
            start_node0,
            end_node0,
            axes_node0,
            steps_node0,
            slice_node0,
            start_node1,
            end_node1,
            axes_node1,
            steps_node1,
            slice_node1,
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
    )

    model = qonnx_make_model(graph, producer_name="test_slice")
    model = ModelWrapper(model)

    # Apply the SlicesToSplitTree transformation
    transformed_model = model.transform(SlicesToSplitTree())

    # Verify that the transformed model contains Split nodes instead of Slice nodes
    slice_nodes = [n for n in transformed_model.graph.node if n.op_type == "Slice"]
    split_nodes = [n for n in transformed_model.graph.node if n.op_type == "Split"]

    assert len(slice_nodes) == 0, "There should be no Slice nodes after transformation."
    assert len(split_nodes) == 1, "There should be a Split nodes after transformation."

def test_triple_slice_pattern():

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1500, 1, 1]
    )
    output0_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 500, 1, 1]
    )
    output1_tensor = helper.make_tensor_value_info(
        "output1", TensorProto.FLOAT, [1, 500, 1, 1]
    )
    output2_tensor = helper.make_tensor_value_info(
        "output2", TensorProto.FLOAT, [1, 500, 1, 1]
    )
    quant_nodes = []
    quant_initializers = []

    scale, zero_point, bitwidth, quant_node = build_quant_node(
        "input",
        "quantized_input",
        0.1,
        0,
        8,
        "quantize_node",
    )
    quant_initializers.extend([scale, zero_point, bitwidth])
    quant_nodes.append(quant_node)

    slice_nodes = []
    for i in range(3):
        start = i * 500
        end = (i + 1) * 500
        (
            start_node,
            end_node,
            axes_node,
            steps_node,
            slice_node,
        ) = build_slice_node(
            "quantized_input",
            f"sliced_output_{i}",
            [start],
            [end],
            [1],
            [1],
            f"slice_node_{i}",
        )
        slice_nodes.extend(
            [start_node, end_node, axes_node, steps_node, slice_node]
        )

    for i in range(3):
        (
            scale_tensor,
            zero_point_tensor,
            bitwidth_tensor,
            quant_node,
        ) = build_quant_node(
            f"sliced_output_{i}",
            f"output{i}",
            0.1,
            0,
            8,
            f"quantize_node_{i}",
        )
        quant_nodes.append(quant_node)
        quant_initializers.extend([scale_tensor, zero_point_tensor, bitwidth_tensor])

    # Create the graph
    graph = helper.make_graph(
        slice_nodes + quant_nodes,
        "test_graph",
        [input_tensor],
        [output0_tensor, output1_tensor, output2_tensor],
        initializer=quant_initializers,
    )
    
    model = qonnx_make_model(graph, producer_name="test_slice")
    model = ModelWrapper(model)

    # Apply the SlicesToSplitTree transformation
    transformed_model = model.transform(SlicesToSplitTree())

    # Verify that the transformed model contains Split nodes instead of Slice nodes
    slice_nodes = [n for n in transformed_model.graph.node if n.op_type == "Slice"]
    split_nodes = [n for n in transformed_model.graph.node if n.op_type == "Split"]

    assert len(slice_nodes) == 0, "There should be no Slice nodes after transformation."
    assert len(split_nodes) == 2, "There should be a Split nodes after transformation."
    assert len(transformed_model.graph.output) == 3, "There should be three outputs in the transformed model."
