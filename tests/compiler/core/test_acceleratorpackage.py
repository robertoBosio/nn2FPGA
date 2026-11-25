import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.utils.qonnx_patch import patch_qonnx_ops


def test_package_resolves_ports_by_internal_name():
    package = AcceleratorPackage(
        input_map={"public_input": {"new_name": "internal_input"}},
        output_map={"public_output": {"new_name": "internal_output"}},
    )

    assert package.get_input_by_internal_name("internal_input") == (
        "public_input",
        package.input_map["public_input"],
    )
    assert package.get_output_by_internal_name("internal_output") == (
        "public_output",
        package.output_map["public_output"],
    )

    with pytest.raises(KeyError, match="internal tensor 'missing'"):
        package.get_output_by_internal_name("missing")


def test_model_rename_updates_package_internal_name():
    input_info = helper.make_tensor_value_info("internal_input", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("internal_output", TensorProto.FLOAT, [1])
    identity = helper.make_node("Identity", ["internal_input"], ["internal_output"])
    model = ModelWrapper(qonnx_make_model(helper.make_graph(
        [identity], "test_graph", [input_info], [output_info]
    )))
    package = AcceleratorPackage(
        input_map={"public_input": {"new_name": "internal_input"}}
    )
    model.set_metadata_prop("accelerator_package", package.to_json())
    patch_qonnx_ops()

    model.rename_tensor("internal_input", "renamed_input")

    transformed_package = AcceleratorPackage.from_json(
        model.get_metadata_prop("accelerator_package")
    )
    assert transformed_package.input_map["public_input"]["new_name"] == "renamed_input"
