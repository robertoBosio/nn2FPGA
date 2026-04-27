import csnake
from .base_hls_test import BaseHLSTest

class TestAXIToStream(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "AXIToStream"
    
    @property
    def unit_filename(self) -> str:
        return "AXIToStream"

    def generate_config_file(self, config_dict):

        # Dump the tensors in a hpp file
        cwr = csnake.CodeWriter()
        cwr.include("<cstdint>")
        cwr.include("<array>")
        cwr.include("<ap_int.h>")
        cwr.include("DequantQuant.hpp")
        cwr.add_line("namespace test_config {")
        cwr.indent()
        for key, value in config_dict.items():
            if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
                cwr.add_line(f"const float {key} = {value}f;")
            else:   
                cwr.add_line(f"const int {key} = {value};")
        cwr.add_line(f"typedef ap_int<{config_dict['AXI_DATAWIDTH']}> TInput;")
        cwr.add_line(f"typedef ap_int<{config_dict['OUT_DATAWIDTH']}> TOutput;")
        cwr.add_line(
            f"typedef DequantQuantPo2<0, TOutput, TOutput> Quantizer;"
        )
        cwr.add_line(
            f"typedef ap_axiu<{config_dict['AXI_DATAWIDTH']}, 0, 0, 0> TInputWord;"
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_axi128_par2(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 128,
            "OUT_DATAWIDTH": 8,
            "DIM1": 4,
            "DIM0": 4,
            "DIM2": 4,
            "DIM1_UNROLL": 1,
            "DIM2_UNROLL": 2,
            "DATA_PER_WORD": 16,
            "PIPELINE_DEPTH": 4,
        }
        self.run(config_dict, hls_steps)

    def test_axi64_par6(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 64,
            "OUT_DATAWIDTH": 8,
            "DIM1": 4,
            "DIM0": 4,
            "DIM2": 3,
            "DIM1_UNROLL": 2,
            "DIM2_UNROLL": 3,
            "DATA_PER_WORD": 8,
            "PIPELINE_DEPTH": 1,
        }
        self.run(config_dict, hls_steps)

    def test_axi64_par3(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 64,
            "OUT_DATAWIDTH": 8,
            "DIM1": 4,
            "DIM0": 4,
            "DIM2": 3,
            "DIM1_UNROLL": 1,
            "DIM2_UNROLL": 3,
            "DATA_PER_WORD": 8,
            "PIPELINE_DEPTH": 2,
        }
        self.run(config_dict, hls_steps)
