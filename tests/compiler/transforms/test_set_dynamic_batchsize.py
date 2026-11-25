from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.transforms.set_dynamic_batchsize import SetDynamicBatchSize


def test_set_dynamic_batchsize_resolves_internal_output_name():
    input_info = helper.make_tensor_value_info("public_input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("public_output", TensorProto.FLOAT, [1, 4])
    package = AcceleratorPackage(
        output_map={
            "public_output": {
                "new_name": "global_out",
                "index": 0,
                "shape": [1, 4],
                "layout": "L[0,1]",
                "quant": "float32",
                "value": None,
                "mode": "axis",
            }
        }
    )
    partition_node = helper.make_node(
        "nn2fpgaPartition",
        ["public_input"],
        ["public_output"],
        domain="nn2fpga.compiler.custom_op",
        accelerator_package=package.to_json(),
    )
    model = ModelWrapper(qonnx_make_model(helper.make_graph(
        [partition_node], "test_graph", [input_info], [output_info]
    )))

    transformed_model = model.transform(SetDynamicBatchSize(batch_size=3))

    assert transformed_model.get_tensor_shape("public_output") == [3, 4]
