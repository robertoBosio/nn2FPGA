import csnake
import pytest
from .base_catapult_test import BaseCatapultTest
from .base_vitis_test import BaseVitisTest

class StreamToAXITestData:

    @property
    def operator_filename(self) -> str:
        return "StreamToAXI"
    
    @property
    def unit_filename(self) -> str:
        return "StreamToAXI"

    def generate_config_file(self, config_dict):

        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        catapult = self.backend == "catapult"
        cwr.include("<ac_int.h>" if catapult else "<ap_int.h>")
        if not catapult:
            cwr.include("vitis/DequantQuant.hpp")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            elif isinstance(value, str):
                continue # Skip string values (like IN_TYPE) as they are not needed in the C++ config.
            else:   
                cwr.add_line(f"const int {key} = {value};")
        in_datatype = config_dict.get("IN_TYPE", "int")
        if catapult:
            cwr.add_line("using TInput = ac_int<8, true>;")
            cwr.add_line(f"using TOutput = ac_int<{config_dict['AXI_DATAWIDTH']}, true>;")
        else:
            cwr.add_line(f"typedef {in_datatype} TInput;")
            cwr.add_line(f"typedef ap_int<{config_dict['AXI_DATAWIDTH']}> TOutput;")
        cwr.add_line(
            f"typedef DequantQuantEqual<TInput> Quantizer;"
        )
        if catapult:
            cwr.add_line(f"using TOutputWord = AxisWord<{config_dict['AXI_DATAWIDTH']}>;")
        else:
            cwr.add_line(
                f"typedef ap_axiu<{config_dict['AXI_DATAWIDTH']}, 0, 0, 0> TOutputWord;"
            )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_axi128_par2(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 128,
            "IN_TYPE": "ap_int<8>",
            "DIM1": 4,
            "DIM0": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 1,
            "DIM2_UNROLL": 2,
            "DATA_PER_WORD": 16,
            "ITER": 32,
            "PIPELINE_DEPTH": 4,
        }
        self.run(config_dict, hls_steps)


    def test_axi64_par6(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 64,
            "IN_TYPE": "ap_int<8>",
            "DIM1": 4,
            "DIM0": 4,
            "DIM2": 3,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 3,
            "DATA_PER_WORD": 8,
            "ITER": 8,
            "PIPELINE_DEPTH": 1,
        }
        self.run(config_dict, hls_steps)


    def test_axi64_par3(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 64,
            "IN_TYPE": "ap_int<8>",
            "DIM1": 4,
            "DIM0": 4,
            "DIM2": 3,
            "DIM1_UNROLL": 1,
            "DIM2_UNROLL": 3,
            "DATA_PER_WORD": 8,
            "ITER": 16,
            "PIPELINE_DEPTH": 2,
        }
        self.run(config_dict, hls_steps)

    def test_axi128_par2_padding(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 128,
            "IN_TYPE": "ap_int<8>",
            "DIM1": 1,
            "DIM0": 1,
            "DIM2": 1000,
            "DIM1_UNROLL": 1,
            "DIM2_UNROLL": 10,
            "DATA_PER_WORD": 16,
            "ITER": 101,
            "PIPELINE_DEPTH": 4,
        }
        self.run(config_dict, hls_steps)
    
    def test_axi128_par1_float(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 128,
            "IN_TYPE": "ap_float<32, 8>",
            "DIM1": 4,
            "DIM0": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 1,
            "DIM2_UNROLL": 1,
            "DATA_PER_WORD": 4,
            "ITER": 64,
            "PIPELINE_DEPTH": 4,
        }
        self.run(config_dict, hls_steps)


class TestVitisStreamToAXI(StreamToAXITestData, BaseVitisTest):
    backend = "vitis"


class TestCatapultStreamToAXI(StreamToAXITestData, BaseCatapultTest):
    backend = "catapult"

    @property
    def unit_filename(self) -> str:
        return "StreamToAXI"

    def test_axi128_par1_float(self, hls_steps):
        pytest.skip("Catapult float transport is not supported")
