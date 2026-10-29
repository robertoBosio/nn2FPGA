import numpy as np
import qonnx.core.onnx_exec as oxe

from onnx import TensorProto, helper, numpy_helper
from onnx.helper import get_attribute_value
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.transforms.convert_to_QCDQ import ConvertToQCDQ


def _build_quant_node(input_name: str, output_name: str, node_name: str):
    return helper.make_node(
        "Quant",
        inputs=[input_name, "scale", "zero_point", "bitwidth"],
        outputs=[output_name],
        domain="qonnx.custom_op.general",
        name=node_name,
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )


def _build_quant_initializers(zero_point_value: int):
    return [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [0.5]),
        helper.make_tensor("zero_point", TensorProto.INT8, [], [zero_point_value]),
        helper.make_tensor("bitwidth", TensorProto.INT32, [], [8]),
    ]


def _build_quant_initializers_rank1(zero_point_value: int):
    return [
        helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.5]),
        helper.make_tensor("zero_point", TensorProto.INT8, [1], [zero_point_value]),
        helper.make_tensor("bitwidth", TensorProto.INT32, [], [8]),
    ]


def _build_quant_initializers_rank1_bitwidth(zero_point_value: int):
    return [
        helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.5]),
        helper.make_tensor("zero_point", TensorProto.INT8, [1], [zero_point_value]),
        helper.make_tensor("bitwidth", TensorProto.INT32, [1], [8]),
    ]


def _wrap_model(nodes, inputs, outputs, initializers):
    graph = helper.make_graph(
        nodes=nodes,
        name="test_graph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    return ModelWrapper(qonnx_make_model(graph, producer_name="test_producer"))


def _constant_tensor_by_output(model: ModelWrapper, output_name: str):
    node = model.find_producer(output_name)
    assert node is not None
    assert node.op_type == "Constant"
    value_attr = next(attr for attr in node.attribute if attr.name == "value")
    return get_attribute_value(value_attr)


def test_convert_to_qcdq_dynamic_quant_preserves_nonzero_zero_point():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    quant_node = _build_quant_node("input", "output", "quant_node")

    model = _wrap_model(
        [quant_node],
        [input_tensor],
        [output_tensor],
        _build_quant_initializers(3),
    )

    transformed_model = model.transform(ConvertToQCDQ())

    assert transformed_model.get_nodes_by_op_type("Quant") == []
    assert len(transformed_model.get_nodes_by_op_type("QuantizeLinear")) == 1
    assert len(transformed_model.get_nodes_by_op_type("DequantizeLinear")) == 1

    qlinear_node = transformed_model.get_nodes_by_op_type("QuantizeLinear")[0]
    dqlinear_node = transformed_model.get_nodes_by_op_type("DequantizeLinear")[0]

    zero_point_tensor = _constant_tensor_by_output(transformed_model, qlinear_node.input[2])
    assert zero_point_tensor.data_type == TensorProto.INT8
    np.testing.assert_array_equal(numpy_helper.to_array(zero_point_tensor), np.array(3, dtype=np.int8))

    assert dqlinear_node.input[2] == qlinear_node.input[2]
    assert transformed_model.check_all_tensor_shapes_specified() is True
    assert transformed_model.get_tensor_shape(qlinear_node.input[2]) == []
    assert transformed_model.get_tensor_shape(qlinear_node.output[0]) == [4]

    input_data = np.array([-1.1, -0.6, 0.2, 1.4], dtype=np.float32)
    transformed_model = transformed_model.transform(InferShapes())
    output_dict = oxe.execute_onnx(transformed_model, {"input": input_data})

    expected_output = np.array([-1.0, -0.5, 0.0, 1.5], dtype=np.float32)
    np.testing.assert_allclose(output_dict["output"], expected_output)


def test_convert_to_qcdq_dynamic_quant_accepts_rank1_per_tensor_params():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    quant_node = _build_quant_node("input", "output", "quant_node")

    model = _wrap_model(
        [quant_node],
        [input_tensor],
        [output_tensor],
        _build_quant_initializers_rank1(3),
    )

    transformed_model = model.transform(ConvertToQCDQ())

    assert transformed_model.get_nodes_by_op_type("Quant") == []
    assert len(transformed_model.get_nodes_by_op_type("QuantizeLinear")) == 1
    assert len(transformed_model.get_nodes_by_op_type("DequantizeLinear")) == 1

    qlinear_node = transformed_model.get_nodes_by_op_type("QuantizeLinear")[0]
    dqlinear_node = transformed_model.get_nodes_by_op_type("DequantizeLinear")[0]

    zero_point_tensor = _constant_tensor_by_output(transformed_model, qlinear_node.input[2])
    assert zero_point_tensor.data_type == TensorProto.INT8
    np.testing.assert_array_equal(numpy_helper.to_array(zero_point_tensor), np.array([3], dtype=np.int8))

    assert dqlinear_node.input[2] == qlinear_node.input[2]
    assert transformed_model.check_all_tensor_shapes_specified() is True
    assert transformed_model.get_tensor_shape(qlinear_node.output[0]) == [4]

    input_data = np.array([-1.1, -0.6, 0.2, 1.4], dtype=np.float32)
    transformed_model = transformed_model.transform(InferShapes())
    output_dict = oxe.execute_onnx(transformed_model, {"input": input_data})

    expected_output = np.array([-1.0, -0.5, 0.0, 1.5], dtype=np.float32)
    np.testing.assert_allclose(output_dict["output"], expected_output)


def test_convert_to_qcdq_dynamic_quant_accepts_rank1_bitwidth():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    quant_node = _build_quant_node("input", "output", "quant_node")

    model = _wrap_model(
        [quant_node],
        [input_tensor],
        [output_tensor],
        _build_quant_initializers_rank1_bitwidth(3),
    )

    transformed_model = model.transform(ConvertToQCDQ())

    assert transformed_model.get_nodes_by_op_type("Quant") == []
    assert len(transformed_model.get_nodes_by_op_type("QuantizeLinear")) == 1
    assert len(transformed_model.get_nodes_by_op_type("DequantizeLinear")) == 1


def test_convert_to_qcdq_constant_quant_accepts_rank1_bitwidth():
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    quant_node = _build_quant_node("weights", "output", "quant_node")
    initializers = [
        helper.make_tensor("weights", TensorProto.FLOAT, [4], [-1.1, -0.6, 0.2, 1.4]),
        *_build_quant_initializers_rank1_bitwidth(3),
    ]

    model = _wrap_model([quant_node], [], [output_tensor], initializers)

    transformed_model = model.transform(ConvertToQCDQ())

    assert transformed_model.get_nodes_by_op_type("Quant") == []
    assert transformed_model.get_nodes_by_op_type("QuantizeLinear") == []
    assert len(transformed_model.get_nodes_by_op_type("DequantizeLinear")) == 1


def test_convert_to_qcdq_constant_quant_preserves_nonzero_zero_point():
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    quant_node = _build_quant_node("weights", "output", "quant_node")
    initializers = [
        helper.make_tensor("weights", TensorProto.FLOAT, [4], [-1.1, -0.6, 0.2, 1.4]),
        *_build_quant_initializers(3),
    ]

    model = _wrap_model([quant_node], [], [output_tensor], initializers)

    transformed_model = model.transform(ConvertToQCDQ())

    assert transformed_model.get_nodes_by_op_type("Quant") == []
    assert transformed_model.get_nodes_by_op_type("QuantizeLinear") == []
    assert len(transformed_model.get_nodes_by_op_type("DequantizeLinear")) == 1

    dqlinear_node = transformed_model.get_nodes_by_op_type("DequantizeLinear")[0]
    quantized_tensor = _constant_tensor_by_output(transformed_model, dqlinear_node.input[0])
    zero_point_tensor = _constant_tensor_by_output(transformed_model, dqlinear_node.input[2])

    assert quantized_tensor.data_type == TensorProto.INT8
    assert zero_point_tensor.data_type == TensorProto.INT8
    np.testing.assert_array_equal(
        numpy_helper.to_array(quantized_tensor),
        np.array([1, 2, 3, 6], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        numpy_helper.to_array(zero_point_tensor),
        np.array(3, dtype=np.int8),
    )


def test_convert_to_qcdq_updates_partition_boundary_names():
    input_tensor = helper.make_tensor_value_info(
        "public_input", TensorProto.FLOAT, [1, 2, 3, 4]
    )
    output_tensor = helper.make_tensor_value_info(
        "public_output", TensorProto.FLOAT, [1, 2, 3, 4]
    )
    package = AcceleratorPackage(
        input_map={
            "public_input": {
                "new_name": "global_in",
                "index": 0,
                "shape": [1, 2, 3, 4],
                "layout": "L[0,2,3,1]",
                "quant": "float32",
                "value": None,
                "mode": "axis",
            }
        },
        output_map={
            "public_output": {
                "new_name": "global_out",
                "index": 0,
                "shape": [1, 2, 3, 4],
                "layout": "L[0,2,3,1]",
                "quant": "float32",
                "value": None,
                "mode": "axis",
            }
        },
    )
    partition_node = helper.make_node(
        "nn2fpgaPartition",
        ["public_input"],
        ["public_output"],
        domain="nn2fpga.compiler.custom_op",
        accelerator_package=package.to_json(),
    )
    model = _wrap_model([partition_node], [input_tensor], [output_tensor], [])

    transformed_model = model.transform(ConvertToQCDQ())
    transformed_partition = transformed_model.get_nodes_by_op_type("nn2fpgaPartition")[0]
    transformed_package = AcceleratorPackage.from_json(
        next(
            attribute.s.decode()
            for attribute in transformed_partition.attribute
            if attribute.name == "accelerator_package"
        )
    )

    assert list(transformed_package.input_map) == [transformed_partition.input[0]]
    assert list(transformed_package.output_map) == [transformed_partition.output[0]]
    assert transformed_package.input_map[transformed_partition.input[0]]["new_name"] == "global_in"
    assert transformed_package.output_map[transformed_partition.output[0]]["new_name"] == "global_out"
    assert transformed_package.input_map[transformed_partition.input[0]]["shape"] == [1, 3, 4, 2]
    assert transformed_package.output_map[transformed_partition.output[0]]["shape"] == [1, 3, 4, 2]
