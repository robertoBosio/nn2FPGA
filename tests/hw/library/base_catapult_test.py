import os
import subprocess
from abc import ABC, abstractmethod

import pytest


class BaseCatapultTest(ABC):
    """Shared helpers for Catapult HLS library tests."""

    @abstractmethod
    def generate_config_file(self, config_dict) -> str:
        ...

    @property
    @abstractmethod
    def unit_filename(self) -> str:
        ...

    def generate_hls_script(
        self, workdir: str, verify: bool = False, simulator: str = "ncsim"
    ) -> str:
        include_dir = "/workspace/nn2FPGA/nn2fpga/hw/library/include"
        testbench = (
            f"/workspace/nn2FPGA/nn2fpga/hw/library/testbench/"
            f"Unit{self.unit_filename}.cpp"
        )
        scverify_options = (
            [f"flow package option set /SCVerify/USE_{simulator.upper()} true"]
            if verify
            else []
        )
        lines = [
            "project new",
            "flow package require /SCVerify",
            *scverify_options,
            "solution options set Input/CppStandard c++11",
            (
                "solution options set Input/CompilerFlags "
                f"\"-DNN2FPGA_CATAPULT -I{include_dir} -I{workdir} "
                "-I/usr/include/x86_64-linux-gnu\""
            ),
            f"solution file add {testbench} -type C++",
            "solution library add mgc_Xilinx-ZYNQ-uplus-2LV_beh -- "
            "-rtlsyntool Vivado -manufacturer Xilinx -family ZYNQ-uplus -speed -2LV",
            "solution library add Xilinx_RAMS",
            "directive set -CLOCKS {clk {-CLOCK_PERIOD 3.33}}",
            "go compile",
            "go libraries",
            "go assembly",
            "directive set /wrap_run/core/main -PIPELINE_INIT_INTERVAL 1",
            "directive set /wrap_run/core/main -PIPELINE_STALL_MODE bubble",
            "go extract",
        ]
        if verify:
            lines.append(
                "flow run /SCVerify/launch_make "
                f"./scverify/Verify_rtl_v_{simulator}.mk {{}} SIMTOOL={simulator}"
            )
        lines.append("quit")
        return "\n".join(lines)

    @pytest.fixture(autouse=True)
    def workdir(self, tmp_path):
        self._workdir = str(tmp_path)

    def run(
        self,
        config_dict: dict,
        hls_steps: str,
        workdir: str | None = None,
        clean: bool = True,
    ):
        workdir = workdir or self._workdir
        os.makedirs(workdir, exist_ok=True)
        steps = {step.strip() for step in hls_steps.split(",")}
        simulator = os.environ.get("SCVERIFY_SIMULATOR", "ncsim")
        if "cosim" in steps and simulator not in {"msim", "ncsim", "vcs"}:
            raise ValueError("SCVERIFY_SIMULATOR must be one of: msim, ncsim, vcs")
        config_path = os.path.join(workdir, "test_config.hpp")
        tcl_path = os.path.join(workdir, "script.tcl")

        with open(config_path, "w") as f:
            f.write(self.generate_config_file(config_dict))
        with open(tcl_path, "w") as f:
            f.write(
                self.generate_hls_script(
                    workdir, verify="cosim" in steps, simulator=simulator
                )
            )
        include_dir = "/workspace/nn2FPGA/nn2fpga/hw/library/include"
        testbench = (
            f"/workspace/nn2FPGA/nn2fpga/hw/library/testbench/"
            f"Unit{self.unit_filename}.cpp"
        )

        if "csim" in steps:
            catapult_home = os.environ.get("CATAPULT_HOME", os.environ.get("MGC_HOME"))
            if not catapult_home:
                raise EnvironmentError("CATAPULT_HOME or MGC_HOME must be set for Catapult tests.")
            executable = os.path.join(workdir, "csim")
            result = subprocess.run(
                [
                    "g++",
                    "-std=c++11",
                    "-DNN2FPGA_CATAPULT",
                    f"-I{catapult_home}/shared/include",
                    f"-I{include_dir}",
                    f"-I{workdir}",
                    testbench,
                    "-o",
                    executable,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert result.returncode == 0, f"C simulation build failed: {result.stderr}"
            result = subprocess.run(
                [executable], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            assert result.returncode == 0, f"C simulation failed: {result.stderr}"
            assert "Passed." in result.stdout, f"Test did not pass: {result.stdout}"

        if "csynth" in steps or "cosim" in steps:
            catapult_home = os.environ.get("CATAPULT_HOME", os.environ.get("MGC_HOME"))
            if not catapult_home:
                raise EnvironmentError("CATAPULT_HOME or MGC_HOME must be set for Catapult tests.")
            result = subprocess.run(
                [f"{catapult_home}/bin/catapult", "-product", "hls", "-shell", "-file", tcl_path],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert result.returncode == 0, f"Catapult failed: {result.stderr}\n{result.stdout}"
            assert "Completed transformation 'extract'" in result.stdout, (
                f"RTL extraction did not complete: {result.stdout}"
            )
            if "cosim" in steps:
                makefile = os.path.join(
                    workdir,
                    "Catapult",
                    "wrap_run.v1",
                    "scverify",
                    f"Verify_rtl_v_{simulator}.mk",
                )
                assert os.path.isfile(makefile), f"SCVerify makefile was not generated: {makefile}"
