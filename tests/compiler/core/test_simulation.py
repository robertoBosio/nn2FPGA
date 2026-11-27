import subprocess
from pathlib import Path

import numpy as np
import pytest

from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.core import simulation


def _port(name, mode="axis"):
    return {
        "new_name": name,
        "index": 0,
        "shape": [1, 4],
        "layout": "L[0,1]",
        "quant": "Q[8,1,1.0,0,0,ROUND]",
        "value": None,
        "mode": mode,
    }


def test_generate_catapult_hls_driver_uses_axis_channels():
    code = simulation.generate_catapult_hls_driver(
        "test_top",
        {"input": _port("global_in")},
        {"output": _port("global_out")},
        {},
        128,
    )

    assert "catapult/utils/testbench_utils.hpp" in code
    assert "ac_channel<AxisWord<128>>& global_in" in code
    assert "ac_channel<AxisWord<128>>& global_out" in code
    assert "int main(int argc, char** argv)" in code
    assert "test_top(global_in, global_out);" in code
    assert "npy_to_catapult_stream<128, ac_int<8, true>, char>" in code
    assert "catapult_stream_to_npy<128, ac_int<8, true>, char>" in code
    assert "hls::stream" not in code
    assert "ap_axiu" not in code


def test_generate_catapult_cosim_driver_uses_scverify_and_checks_sidebands():
    code = simulation.generate_catapult_cosim_driver(
        "test_top",
        {"input": _port("global_in")},
        {"output": _port("global_out")},
        {},
        128,
        ["/tmp/global_in.npy"],
        ["/tmp/expected_global_out.npy"],
        ["/tmp/actual_global_out.npy"],
    )

    assert "mc_scverify.h" in code
    assert "CCS_MAIN(int argc, char** argv)" in code
    assert code.count("CCS_DESIGN(test_top)(global_in, global_out);") == 1
    assert "catapult_stream_to_npy_and_compare" in code
    assert "/tmp/expected_global_out.npy" in code
    assert "/tmp/actual_global_out.npy" in code


def test_dump_catapult_cosim_tcl_uses_scverify_flow():
    tcl = simulation.dump_catapult_cosim_tcl(
        "test_top", "250", "/tmp/sim", "ncsim"
    )

    assert "flow package option set /SCVerify/USE_NCSIM true" in tcl
    assert "-I/tmp/sim/compat" in tcl
    assert "directive set -CLOCKS {clk {-CLOCK_PERIOD 4.00}}" in tcl
    assert "directive set /test_top -FIFO_DEPTH 2" in tcl
    assert "PIPELINE_INIT_INTERVAL" not in tcl
    assert "flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_ncsim.mk {} SIMTOOL=ncsim" in tcl


def test_catapult_testbench_helpers_use_unsigned_slice_offsets():
    helper = (
        Path(__file__).resolve().parents[3]
        / "nn2fpga/hw/library/include/catapult/utils/testbench_utils.hpp"
    ).read_text()

    assert "set_slc(static_cast<unsigned>(lane * TData::width)" in helper
    assert helper.count("static_cast<unsigned>(lane * TData::width)") == 3
    assert "Invalid AXIS sidebands" in helper


def test_generate_catapult_hls_driver_rejects_non_axis_ports():
    with pytest.raises(ValueError, match="only axis ports"):
        simulation.generate_catapult_hls_driver(
            "test_top", {"input": _port("global_in", "m_axi")}, {}, {}, 128
        )


def test_generate_catapult_hls_driver_rejects_buffers():
    with pytest.raises(ValueError, match="external memory buffers"):
        simulation.generate_catapult_hls_driver(
            "test_top", {"input": _port("global_in")}, {}, {"buffer": {}}, 128
        )


def test_catapult_simulation_runs_native_cpp_driver(tmp_path, monkeypatch):
    package = AcceleratorPackage(
        hls_code_b64="dm9pZCB0ZXN0X3RvcCgpIHt9",
        input_map={"input": _port("global_in")},
        output_map={"output": _port("global_out")},
        work_dir=str(tmp_path),
        board_name="test",
        top_name="test_top",
        frequency="250",
        hls_version="2025.2",
        hls_backend="catapult",
        simulation="csim",
    )
    commands = []

    monkeypatch.setenv("CATAPULT_HOME", "/opt/catapult")
    monkeypatch.setattr(
        simulation, "read_board_info", lambda _: {"axi_bitwidth": 128}
    )

    def run(command, **kwargs):
        commands.append(command)
        if command[0].endswith("/csim"):
            np.save(tmp_path / "sim" / "global_out.npy", np.array([[1, 2, 3, 4]], dtype=np.int8))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(simulation.subprocess, "run", run)
    context = simulation.simulate(
        package.to_json(), {"input": np.array([[1, 2, 3, 4]], dtype=np.int8)}
    )

    compile_command, execute_command = commands
    assert compile_command[:3] == ["g++", "-std=c++11", "-DNN2FPGA_CATAPULT"]
    assert "-I/opt/catapult/shared/include" in compile_command
    assert "/workspace/nn2FPGA/deps/cnpy/cnpy.cpp" in compile_command
    assert execute_command[0].endswith("/sim/csim")
    assert len(execute_command) == 3
    assert np.array_equal(
        context["output"], np.array([[1, 2, 3, 4]], dtype=np.int8)
    )


def test_catapult_cosimulation_is_not_supported(tmp_path):
    package = AcceleratorPackage(
        hls_code_b64="dm9pZCB0ZXN0X3RvcCgpIHt9",
        input_map={"input": _port("global_in")},
        output_map={"output": _port("global_out")},
        work_dir=str(tmp_path),
        board_name="test",
        top_name="test_top",
        frequency="250",
        hls_version="2025.2",
        hls_backend="catapult",
        simulation="cosim",
    )
    with pytest.raises(NotImplementedError, match="RTL co-simulation is not supported"):
        simulation.simulate(package.to_json(), {})
