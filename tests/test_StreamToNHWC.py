import csnake
from .base_hls_test import BaseHLSTest

class TestStreamToNHWC(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "StreamToNHWC"

    def generate_config_file(self, config_dict):

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
        cwr.add_line(f"typedef ap_int<{config_dict['IN_DATAWIDTH']}> TInput;")
        cwr.add_line(f"typedef ap_int<{config_dict['AXI_DATAWIDTH']}> TOutput;")
        cwr.add_line(
            f"typedef DequantQuantPo2<0, TInput, TInput> Quantizer;"
        )
        cwr.add_line(
            f"typedef ap_axiu<{config_dict['AXI_DATAWIDTH']}, 0, 0, 0> TOutputWord;"
        )
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_axi128_par2(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 128,
            "IN_DATAWIDTH": 8,
            "WIDTH": 4,
            "HEIGHT": 4,
            "CH": 4,
            "IN_W_PAR": 1,
            "IN_CH_PAR": 2,
            "DATA_PER_WORD": 16,
            "PIPELINE_DEPTH": 4,
        }
        self.run(config_dict, hls_steps)

    def test_axi64_par6(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 64,
            "IN_DATAWIDTH": 8,
            "WIDTH": 4,
            "HEIGHT": 4,
            "CH": 3,
            "IN_W_PAR": 2,
            "IN_CH_PAR": 3,
            "DATA_PER_WORD": 8,
            "PIPELINE_DEPTH": 1,
        }
        self.run(config_dict, hls_steps)

    def test_axi64_par3(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 64,
            "IN_DATAWIDTH": 8,
            "WIDTH": 4,
            "HEIGHT": 4,
            "CH": 3,
            "IN_W_PAR": 1,
            "IN_CH_PAR": 3,
            "DATA_PER_WORD": 8,
            "PIPELINE_DEPTH": 2,
        }
        self.run(config_dict, hls_steps)

    def test_axi128_par2_padding(self, hls_steps):
        config_dict = {
            "AXI_DATAWIDTH": 128,
            "IN_DATAWIDTH": 8,
            "WIDTH": 1,
            "HEIGHT": 1,
            "CH": 1000,
            "IN_W_PAR": 1,
            "IN_CH_PAR": 2,
            "DATA_PER_WORD": 16,
            "PIPELINE_DEPTH": 4,
        }
        self.run(config_dict, hls_steps)
