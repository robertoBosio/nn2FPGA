import csnake
from .base_hls_test import BaseHLSTest

class TestTensorDuplicator(BaseHLSTest):

    @property
    def operator_filename(self) -> str:
        return "TensorDuplicator"

    @property
    def unit_filename(self) -> str:
        return "TensorDuplicator"

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
        cwr.add_line(f"typedef ap_int<{config_dict['DATAWIDTH']}> TInput;")
        cwr.add_line(f"using TWord = std::array<TInput, {config_dict['CH_PAR']}>;")
        cwr.dedent()
        cwr.add_line("}")
        return cwr.code

    def test_2copies(self, hls_steps):
        config_dict = {
            "DATAWIDTH": 8,
            "IN_HEIGHT": 4,
            "IN_WIDTH": 4,
            "IN_CH": 4,
            "CH_PAR": 2,
            "W_PAR": 2,
            "PIPELINE_DEPTH": 5,
        }
        self.run(
            config_dict,
            hls_steps,
        )
