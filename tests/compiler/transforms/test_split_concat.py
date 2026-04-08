from nn2fpga.compiler.transforms.split_concat import SplitConcat
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
import numpy as np

def build_concat_graph(
    inputs: int,
) -> ModelWrapper:

    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2 * inputs])

    input_tensors = []
    for i in range(inputs):
        input_name = f"input{i+1}"
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 2])
        input_tensors.append(input_tensor)

    concat_node = helper.make_node(
        "Concat",
        inputs=[f"input{i+1}" for i in range(inputs)],
        outputs=["output"],
        name="concat_node",
        axis=1,
        domain=""
    )

    graph = helper.make_graph(
        nodes=[concat_node],
        name="test_graph",
        inputs=input_tensors,
        outputs=[output_tensor],
    )
    model = qonnx_make_model(graph, producer_name="test_model")
    return ModelWrapper(model)

def test_concat3():

    model_wrapper = build_concat_graph(3)
    transformed_model = model_wrapper.transform(SplitConcat())

    concat_nodes = [n for n in transformed_model.graph.node if n.op_type == "Concat"]

    assert len(concat_nodes) == 2

def test_concat4():

    model_wrapper = build_concat_graph(4)
    transformed_model = model_wrapper.transform(SplitConcat())

    concat_nodes = [n for n in transformed_model.graph.node if n.op_type == "Concat"]

    assert len(concat_nodes) == 3

def test_concat5():

    model_wrapper = build_concat_graph(5)
    transformed_model = model_wrapper.transform(SplitConcat())

    concat_nodes = [n for n in transformed_model.graph.node if n.op_type == "Concat"]

    assert len(concat_nodes) == 4

def test_concat6():

    model_wrapper = build_concat_graph(6)
    transformed_model = model_wrapper.transform(SplitConcat())

    concat_nodes = [n for n in transformed_model.graph.node if n.op_type == "Concat"]

    assert len(concat_nodes) == 5

def test_concat7():

    model_wrapper = build_concat_graph(7)
    transformed_model = model_wrapper.transform(SplitConcat())

    concat_nodes = [n for n in transformed_model.graph.node if n.op_type == "Concat"]

    assert len(concat_nodes) == 6

def test_correctness():

    model_wrapper = build_concat_graph(5)
    transformed_model = model_wrapper.transform(SplitConcat())
    transformed_model = transformed_model.transform(InferShapes())

    input_names = [f"input{i+1}" for i in range(5)]
    output_name = "output"

    input_dict = {
        name: np.array([[i, i + 1]], dtype=np.float32)
        for i, name in enumerate(input_names)
    }
    expected_output = np.array([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5]], dtype=np.float32)

    output_dict = oxe.execute_onnx(transformed_model, input_dict)

    np.testing.assert_allclose(output_dict[output_name], expected_output)