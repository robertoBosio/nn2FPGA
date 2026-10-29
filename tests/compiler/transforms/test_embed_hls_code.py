from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.core.simulation import generate_hls_driver
from nn2fpga.compiler.core.tensor_fifo import TensorFifo, set_custom_tensor_fifo_metadata
from nn2fpga.compiler.transforms.embed_hls_code import generate_hls_code
import nn2fpga.compiler.transforms.embed_hls_code as embed_hls_code


def _model_with_ports():
    input_info = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, [1, 4])
    model = ModelWrapper(qonnx_make_model(helper.make_graph(
        [], "test_graph", [input_info], [output_info]
    )))
    model.set_metadata_prop("top_name", "test_top")
    model.set_metadata_prop("board_name", "test")
    for name in ("global_in", "global_out"):
        set_custom_tensor_fifo_metadata(model, name, TensorFifo(1, "ap_uint<128>"))
    return model


def _port(name, mode):
    return {
        "new_name": name,
        "index": 0,
        "shape": [1, 4],
        "layout": "L[0,1]",
        "quant": "float32",
        "value": None,
        "mode": mode,
    }


def test_generate_hls_code_uses_ap_ctrl_none_for_axis_ports():
    code = generate_hls_code(
        _model_with_ports(),
        AcceleratorPackage(
            input_map={"input": _port("global_in", "axis")},
            output_map={"output": _port("global_out", "axis")},
        ),
    )

    assert "#pragma HLS INTERFACE ap_ctrl_none port=return" in code
    assert "#pragma HLS INTERFACE s_axilite port=return bundle=control" not in code


def test_generate_hls_code_uses_axilite_for_maxi_ports(monkeypatch):
    monkeypatch.setattr(
        embed_hls_code,
        "read_board_info",
        lambda _: {"axi_bitwidth": 128},
    )
    code = generate_hls_code(
        _model_with_ports(),
        AcceleratorPackage(
            input_map={
                "input": {
                    **_port("global_in", "m_axi"),
                    "shape": [1, 17],
                    "quant": "Q[8,0,1.0,0,0,ROUND]",
                }
            },
            output_map={
                "output": {
                    **_port("global_out", "m_axi"),
                    "shape": [1, 17],
                    "quant": "Q[8,0,1.0,0,0,ROUND]",
                }
            },
        ),
    )

    assert "#pragma HLS INTERFACE ap_ctrl_none port=return" not in code
    assert "#pragma HLS INTERFACE s_axilite port=return bundle=control" in code
    assert "#pragma HLS INTERFACE m_axi port=global_in bundle=global_in_bundle depth=2" in code
    assert "#pragma HLS INTERFACE s_axilite port=global_in bundle=control" in code
    assert "#pragma HLS INTERFACE m_axi port=global_out bundle=global_out_bundle depth=2" in code
    assert "#pragma HLS INTERFACE s_axilite port=global_out bundle=control" in code

    driver_code = generate_hls_driver(
        "test_top",
        {
            "input": {
                **_port("global_in", "m_axi"),
                "shape": [1, 17],
                "quant": "Q[8,0,1.0,0,0,ROUND]",
            }
        },
        {
            "output": {
                **_port("global_out", "m_axi"),
                "shape": [1, 17],
                "quant": "Q[8,0,1.0,0,0,ROUND]",
            }
        },
        {},
        128,
    )
    assert "ap_uint<128> global_in[2]" in driver_code
    assert "ap_uint<128> global_out[2]" in driver_code
