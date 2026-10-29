from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.core.tensor_type import QuantizedTensorType, set_custom_tensor_datatype
from nn2fpga.compiler.transforms.add_streaming_params import AddStreamingParams
import nn2fpga.compiler.transforms.add_streaming_params as add_streaming_params


def test_add_streaming_params_uses_maxi_without_dma(monkeypatch):
    stream_memory = helper.make_node(
        "StreamingMemory",
        ["weights"],
        ["weights_stream"],
        domain="nn2fpga.compiler.custom_op",
        times=1,
        data_per_word=1,
        mem_shape=[1, 1],
    )
    graph = helper.make_graph(
        [stream_memory],
        "test_graph",
        [],
        [helper.make_tensor_value_info("weights_stream", TensorProto.UINT8, [1, 1])],
        initializer=[helper.make_tensor("weights", TensorProto.UINT8, [1, 1], [1])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_metadata_prop("board_name", "test")
    model.set_metadata_prop("accelerator_package", AcceleratorPackage().to_json())
    set_custom_tensor_datatype(
        model,
        "weights_stream",
        QuantizedTensorType(1.0, 0, 8, False, False, "ROUND"),
    )
    monkeypatch.setattr(
        add_streaming_params,
        "read_board_info",
        lambda **_: {"axi_bitwidth": 128},
    )

    transformed_model = model.transform(AddStreamingParams(use_dma=False))
    package = AcceleratorPackage.from_json(
        transformed_model.get_metadata_prop("accelerator_package")
    )
    converter = transformed_model.get_nodes_by_op_type("AXIToStream")[0]

    assert package.input_map["const_param_stream"]["mode"] == "m_axi"
    assert next(attr.s.decode() for attr in converter.attribute if attr.name == "interface_type") == "m_axi"
