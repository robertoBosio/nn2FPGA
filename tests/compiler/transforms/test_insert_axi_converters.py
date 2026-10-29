from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.transforms.insert_axi_converters import InsertAXIConverters
import nn2fpga.compiler.transforms.insert_axi_converters as insert_axi_converters


def _port(new_name):
    return {
        "new_name": new_name,
        "index": 0,
        "shape": [1, 4],
        "layout": "L[0,1]",
        "quant": "float32",
        "value": None,
        "mode": "axis",
    }


def test_insert_axi_converters_resolves_internal_port_names(monkeypatch):
    input_info = helper.make_tensor_value_info("internal_input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("internal_output", TensorProto.FLOAT, [1, 4])
    identity = helper.make_node("Identity", ["internal_input"], ["internal_output"])
    model = ModelWrapper(qonnx_make_model(helper.make_graph(
        [identity], "test_graph", [input_info], [output_info]
    )))
    package = AcceleratorPackage(
        input_map={"public_input": _port("internal_input")},
        output_map={"public_output": _port("internal_output")},
    )
    model.set_metadata_prop("board_name", "test")
    model.set_metadata_prop("accelerator_package", package.to_json())
    monkeypatch.setattr(
        insert_axi_converters,
        "read_board_info",
        lambda **_: {"axi_bitwidth": 128},
    )

    transformed_model = model.transform(InsertAXIConverters(use_dma=False))
    transformed_package = AcceleratorPackage.from_json(
        transformed_model.get_metadata_prop("accelerator_package")
    )

    assert list(transformed_package.input_map) == ["public_input"]
    assert list(transformed_package.output_map) == ["public_output"]
    assert transformed_package.input_map["public_input"]["mode"] == "m_axi"
    assert transformed_package.output_map["public_output"]["mode"] == "m_axi"
