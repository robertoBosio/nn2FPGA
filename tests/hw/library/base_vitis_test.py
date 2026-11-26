import os
import shutil
import subprocess
from abc import ABC, abstractmethod
import numpy as np
from onnx import TensorProto
import pytest

PROJECT_NAME = "proj_unit_test"
FILE_DIR = "/workspace/nn2FPGA/nn2fpga/hw/library"

class BaseVitisTest(ABC):
    """
    Shared helpers for Vitis HLS tests.
    Subclasses must implement generate_config_file(config_dict) -> str (C++ header text).
    """

    @abstractmethod
    def generate_config_file(self, config_dict, **kwargs) -> str:
        ...

    @property
    @abstractmethod
    def operator_filename(self) -> str | list[str]:
        """The filename of the operator under test."""
        ...
    
    @property
    @abstractmethod
    def unit_filename(self) -> str:
        """The filename of the unit test for the operator."""
        ...

    @staticmethod
    def get_tensorproto_dtype(datawidth, is_unsigned):
        if datawidth == 8:
            return TensorProto.UINT8 if is_unsigned else TensorProto.INT8
        elif datawidth == 16:
            return TensorProto.UINT16 if is_unsigned else TensorProto.INT16
        elif datawidth == 32:
            return TensorProto.UINT32 if is_unsigned else TensorProto.INT32
        else:
            raise ValueError(f"Unsupported datawidth: {datawidth}")

    @staticmethod
    def get_numpy_dtype(datawidth, is_unsigned):
        if datawidth == 8:
            return np.uint8 if is_unsigned else np.int8
        elif datawidth == 16:
            return np.uint16 if is_unsigned else np.int16
        elif datawidth == 32:
            return np.uint32 if is_unsigned else np.int32
        else:
            raise ValueError(f"Unsupported datawidth: {datawidth}")

    def generate_hls_script(self, steps: str, workdir: str) -> str:
        op_filenames = self.operator_filename
        un_filename = self.unit_filename

        # allow single string or list of strings
        if isinstance(op_filenames, str):
            op_filenames = [op_filenames]

        steps_list = [s.strip() for s in steps.split(",")]
        do_csim   = "csim"   in steps_list
        do_csynth = (
            "csynth" in steps_list or "cosim" in steps_list or "export" in steps_list
        )
        do_cosim  = "cosim"  in steps_list
        do_export = "export" in steps_list

        lines = [
            f'open_project -reset "{PROJECT_NAME}"',
            'open_solution -reset solution0',
            *(
                f'add_files {FILE_DIR}/include/vitis/{op_name}.hpp -cflags "-I/workspace/nn2FPGA/nn2fpga/hw/library/include -I{workdir}"' for op_name in op_filenames
            ),
            f'add_files {FILE_DIR}/testbench/Unit{un_filename}.cpp -cflags "-I/workspace/nn2FPGA/nn2fpga/hw/library/include -I{workdir}"',
            f'add_files -tb {FILE_DIR}/testbench/Unit{un_filename}.cpp -cflags "-I/workspace/nn2FPGA/nn2fpga/hw/library/include -I{workdir}"',
            "set_top wrap_run",
            "set_part xck26-sfvc784-2LV-c",
            "create_clock -period 3.33ns",
            "config_compile -pipeline_style frp",
            "",
        ]
        if do_csim:
            lines.append("csim_design -argv csim")
        if do_csynth:
            lines.append("csynth_design")
        if do_cosim:
            lines.append("cosim_design -argv cosim")
        if do_export:
            lines.append(f'export_design -flow impl')
        lines.append("exit")
        return "\n".join(lines)

    @staticmethod
    def runhls(tcl_file: str, workdir: str):
        xilinx_version = os.environ.get("XILINX_VERSION", "0")
        if float(xilinx_version) < 2025.1:
            cmd = ["vitis_hls", "-f", tcl_file]
        else:
            cmd = ["vitis-run", "--mode", "hls", "--tcl", tcl_file]
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workdir,
        )

    @pytest.fixture(autouse=True)
    def workdir(self, tmp_path):
        self._workdir = str(tmp_path)

    def run(self, config_dict: dict, steps: str, workdir: str | None = None, clean: bool = True, **kwargs):
        workdir = workdir or self._workdir
        os.makedirs(workdir, exist_ok=True)
        # write config header
        testconfig_path = os.path.join(workdir, "test_config.hpp")
        with open(testconfig_path, "w") as f:
            f.write(self.generate_config_file(config_dict, **kwargs))

        # write script
        tcl_path = os.path.join(workdir, "script.tcl")
        with open(tcl_path, "w") as f:
            f.write(self.generate_hls_script(steps, workdir))

        try:
            result = self.runhls(tcl_path, workdir)
            assert result.returncode == 0, f"HLS failed: {result.stderr}"

            steps_l = [s.strip() for s in steps.split(",")]
            if "csim" in steps_l or "cosim" in steps_l:
                assert "passed" in result.stdout.lower(), f"Test did not pass: {result.stdout}"
            if "csynth" in steps_l or "cosim" in steps_l:
                assert "All loop constraints were satisfied" in result.stdout, \
                    f"Loop constraints not satisfied: {result.stdout}"
        finally:
            if clean:
                for path in (tcl_path, testconfig_path):
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                shutil.rmtree(os.path.join(workdir, PROJECT_NAME), ignore_errors=True)
                for filename in os.listdir(workdir):
                    if filename.startswith("vitis_hls") and filename.endswith(".log"):
                        try:
                            os.remove(os.path.join(workdir, filename))
                        except OSError:
                            pass
