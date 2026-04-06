from nn2fpga.compiler.transforms.remove_noop_nodes import RemoveNoopNodes
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from onnx import TensorProto, helper

def test_multiply_by_one():

    # Build a simple model with a Mul node that multiplies by 1
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    const_tensor = helper.make_tensor(
        name="const_one",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[1.0],
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    mul_node = helper.make_node(
        "Mul",
        inputs=["input", "const_one"],
        outputs=["output"],
        name="mul_by_one",
    )

    graph = helper.make_graph(
        nodes=[mul_node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[const_tensor],
    )
    model = qonnx_make_model(graph, producer_name="test_model")
    model_wrapper = ModelWrapper(model)

    # Apply the RemoveNoopNodes transformation
    transformed_model = model_wrapper.transform(RemoveNoopNodes())

    assert len(transformed_model.graph.node) == 0